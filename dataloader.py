# TCGA tile input pipeline backed by Parquet shards. Each shard is a parquet
# file of `{path: string, jpeg: binary}` rows. We open the shards via pyarrow
# directly (NOT `datasets.load_dataset`, which copies into ~/.cache) so the
# ~120 GB of shards are mmap'd in place with zero duplication. Random access
# is resolved by per-shard ParquetFile.read_row_group; prepare.py packs each
# shard with PARQUET_ROW_GROUP_SIZE=64 rows/group so reading one row group is
# ~2 MB and __getitem__ is ~2-3 ms incl. JPEG decode.
#
# Patients (not tiles) are hashed by TCGA barcode and the bottom `val_fraction`
# of the hash space is held out from training; train.py instantiates the dataset
# twice (`is_train=True` for the training loop, `is_train=False` for the
# lightweight DINO/JEPA/KDE validation pass), so the held-out patient slice
# stays cleanly out-of-distribution from optimization.
#
# Augmentation per view: RandomResizedCrop -> optional HEDJitter -> horizontal/
# vertical flips -> right-angle rotation -> ColorJitter -> occasional
# grayscale/blur -> Normalize.
#
# This file is the *pretraining* input pipeline only. The downstream probes
# (probe.py) do not import anything from here.
#
# Optional metadata guidance (cfg.metadata.enabled): looks up one or more TCGA clinical/genomic
# covariates (e.g. cancer subtype, imaging scanner) per patient barcode from
# metadata/fino_meta.json -- already-published, already-vetted public metadata (see
# metadata/README.md), not derived here.
# train.py uses this as an auxiliary classification target; -1 means "no label for this patient".

import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


HED_FROM_RGB = torch.tensor(
    [
        [1.87798274, -1.00767869, -0.55611582],
        [-0.06590806, 1.13473037, -0.1355218],
        [-0.60190736, -0.48041419, 1.57358807],
    ],
    dtype=torch.float32,
)
RGB_FROM_HED = torch.tensor(
    [
        [0.65, 0.7, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=torch.float32,
)
LOG_1E6 = float(np.log(1e-6))
TILE_SIZE = 224


# Patients (not tiles) are the split unit so train/val never share a case.
def patient_in_val(patient_id, seed, val_fraction):
    key = f"{seed}:{patient_id}".encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64
    return value < float(val_fraction)


# Path entries start with the SVS stem (TCGA-XX-XXXX-...); the first three dash parts are the patient barcode.
def patient_id_from_relpath(rel):
    return "-".join(rel.split("/", 1)[0].split("-")[:3])


# Stain-space jitter; the stain augmentation hook for pretraining tiles. When sigma_hi > sigma the
# per-image perturbation strength is itself drawn ~ U(sigma, sigma_hi) (RandStainNA-style: random
# stain-augmentation strength gives heavier, more realistic scanner/stain diversity, which improves
# cross-site robustness). sigma_hi <= sigma (the default) reproduces the fixed-strength behaviour exactly.
class HEDJitter(nn.Module):
    # Store conversion matrices as buffers so transforms move with the module dtype/device if needed.
    def __init__(self, sigma, sigma_hi=None):
        super().__init__()
        self.sigma = sigma
        self.sigma_hi = sigma_hi if (sigma_hi and sigma_hi > sigma) else None
        self.register_buffer("hed_from_rgb", HED_FROM_RGB)
        self.register_buffer("rgb_from_hed", RGB_FROM_HED)

    # Perturb HED channels, then convert back to RGB while the crop is still in [0, 1].
    def forward(self, x):
        sigma = self.sigma if self.sigma_hi is None else float(torch.empty(()).uniform_(self.sigma, self.sigma_hi))
        rgb = x.permute(1, 2, 0).clamp_min(1e-6)
        hed = (torch.log(rgb) / LOG_1E6) @ self.hed_from_rgb.to(dtype=x.dtype)
        hed = hed.clamp_min(0.0)
        shift = torch.randn((1, 1, 3), dtype=x.dtype) * sigma
        scale = 1.0 + torch.randn((1, 1, 3), dtype=x.dtype) * sigma
        hed = hed * scale + shift
        log_rgb = -(hed * (-LOG_1E6)) @ self.rgb_from_hed.to(dtype=x.dtype)
        return torch.exp(log_rgb).clamp_(0.0, 1.0).permute(2, 0, 1)


# Map-style TCGA tile dataset that emits global/local multi-view stacks for train.py.
class TCGATileDataset(Dataset):
    # Glob shards, build a (shard_idx, row_in_shard) index over the requested patient
    # split, and configure augmentations. `is_train=True` keeps the (1 - val_fraction)
    # majority of patient ids; `is_train=False` keeps the held-out `val_fraction` slice.
    def __init__(self, cfg, is_train=True):
        data = cfg["data"]
        train = cfg["train"]
        self.tissue_thresh = float(data["tissue_thresh"]) if is_train else 0.0
        dataset_dir = Path(data["dataset_dir"])
        self.shards = sorted(dataset_dir.glob("shard-*.parquet"))
        if not self.shards:
            raise FileNotFoundError(
                f"No parquet shards (shard-*.parquet) under {dataset_dir}. Run "
                f"`python prepare.py {cfg['config_path']} download=True` to fetch them from "
                f"the medarc/nanopath HF dataset before training."
            )
        if int(train["global_size"]) > TILE_SIZE:
            raise ValueError(f"global_size must be <= {TILE_SIZE}, got global_size={train['global_size']}")
        # Lazy ParquetFile handles, opened on first __getitem__ in each worker
        # so fork-children own their own file positions.
        self._readers = [None] * len(self.shards)
        # Pull just the path column from each shard once to build the train index;
        # the JPEG bytes column stays on disk until __getitem__.
        in_split_shard = []
        in_split_row = []
        for shard_idx, shard_path in enumerate(self.shards):
            paths = pq.read_table(str(shard_path), columns=["path"], memory_map=True)["path"].to_pylist()
            for row_idx, p in enumerate(paths):
                # XOR with is_train: training keeps tiles where patient_in_val is False,
                # validation keeps the complement.
                if patient_in_val(patient_id_from_relpath(p), data["split_seed"], data["val_fraction"]) != is_train:
                    in_split_shard.append(shard_idx)
                    in_split_row.append(row_idx)
        if not in_split_shard:
            raise ValueError(f"no {'train' if is_train else 'val'} tiles found in {dataset_dir}; check val_fraction={data['val_fraction']}")
        # Two parallel int32 arrays (~32 MB total for 4M tiles) shared COW across DataLoader fork-workers.
        self.shard_of = np.asarray(in_split_shard, dtype=np.int32)
        self.row_of = np.asarray(in_split_row, dtype=np.int32)
        # Optional semantic curation (train only): restrict the index to the deduplicated subset
        # in data.curation_manifest (produced by curate.py). Spends the fixed 1M-presentation
        # budget on more-informative, less-redundant tiles. Intersect rather than trust the
        # manifest blindly so a stale manifest (built at a different split) fails loud on emptiness.
        if is_train and data.get("curation_manifest"):
            m = np.load(data["curation_manifest"])
            keep = set(zip(m["shard_of"].tolist(), m["row_of"].tolist()))
            sel = np.array([(int(s), int(r)) in keep for s, r in zip(self.shard_of, self.row_of)], dtype=bool)
            if not sel.any():
                raise ValueError(f"curation_manifest {data['curation_manifest']} matched 0 tiles; rebuild it for split_seed={data['split_seed']}")
            print(f"[data] curation: {int(sel.sum())}/{len(sel)} tiles kept from {data['curation_manifest']}", flush=True)
            self.shard_of, self.row_of = self.shard_of[sel], self.row_of[sel]
        # Metadata guidance: barcode -> value maps loaded once and shared copy-on-write across
        # DataLoader fork-workers (same sharing pattern as shard_of/row_of). cfg.metadata has two
        # lists of [factor_name, sign] pairs: `discrete` (categorical -> class-id, cross-entropy)
        # and `continuous` (numeric vector -> z-scored regression). sign is only used by train.py
        # (M+ encourage / M- suppress via GradScale), not here.
        metadata_cfg = cfg.get("metadata") or {}
        self.metadata_enabled = bool(metadata_cfg.get("enabled"))
        if self.metadata_enabled:
            meta = json.loads((dataset_dir / "fino_meta.json").read_text())
            self.metadata_discrete = [name for name, _ in metadata_cfg.get("discrete", [])]
            self.metadata_continuous = [name for name, _ in metadata_cfg.get("continuous", [])]
            self.discrete_labels = {name: meta["discrete"][name] for name in self.metadata_discrete}
            self.continuous_values = {name: meta["continuous"][name] for name in self.metadata_continuous}
            self.continuous_dims = {name: meta["cont_dim"][name] for name in self.metadata_continuous}
        mean, std = data["mean"], data["std"]
        self.global_views = int(train["global_views"])
        self.local_views = int(train["local_views"])
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        # Tissue has no canonical orientation (unlike natural images), so a 90-degree-multiple
        # rotation is a genuine invariance of the data, not just a trick: it's lossless/exact on
        # a square crop (no interpolation, no border artifacts), unlike an arbitrary-angle
        # rotation would be. Standard torchvision v2 primitives only.
        random_right_angle = v2.RandomChoice([v2.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)])
        # Global crops carry the high-context view used by the DINO/JEPA objectives.
        self.global_aug = v2.Compose(
            [
                v2.RandomResizedCrop(train["global_size"], scale=tuple(data["global_crop_scale"]), antialias=True),
                *([HEDJitter(data["hed_jitter"], data.get("hed_jitter_hi"))] if data["hed_jitter"] > 0 else []),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                random_right_angle,
                v2.ColorJitter(data["color_jitter"], data["color_jitter"], data["color_jitter_saturation"], 0.0),
                v2.RandomGrayscale(p=0.1),
                v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 1.8))], p=0.35),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        # Local crops force the encoder to align small tissue regions with the global context.
        self.local_aug = v2.Compose(
            [
                v2.RandomResizedCrop(train["local_size"], scale=tuple(data["local_crop_scale"]), antialias=True),
                *([HEDJitter(data["hed_jitter"], data.get("hed_jitter_hi"))] if data["hed_jitter"] > 0 else []),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                random_right_angle,
                v2.ColorJitter(data["color_jitter"], data["color_jitter"], data["color_jitter_saturation"], 0.0),
                v2.RandomGrayscale(p=0.1),
                v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 1.8))], p=0.35),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    # Dataset length is the number of tiles in this train/val split.
    def __len__(self):
        return int(self.shard_of.shape[0])

    # Read one JPEG row, decode, apply augmentations, and return train.py fields.
    def __getitem__(self, idx):
        idx = int(idx)
        for _ in range(9):
            shard_idx = int(self.shard_of[idx])
            row_idx = int(self.row_of[idx])
            reader = self._readers[shard_idx]
            if reader is None:
                reader = pq.ParquetFile(str(self.shards[shard_idx]), memory_map=True)
                self._readers[shard_idx] = reader
            # Each shard has uniform-size row groups (PARQUET_ROW_GROUP_SIZE in
            # prepare.py); reading one group is ~2 MB and ~2-3 ms incl. JPEG decode.
            rg_size = reader.metadata.row_group(0).num_rows
            rg_idx = row_idx // rg_size
            row_in_rg = row_idx % rg_size
            table = reader.read_row_group(rg_idx, columns=["path", "jpeg"])
            rel = table["path"][row_in_rg].as_py()
            jpeg_bytes = table["jpeg"][row_in_rg].as_py()
            with Image.open(io.BytesIO(jpeg_bytes)) as img:
                tile = self.to_tensor(img.convert("RGB"))
            if self.tissue_thresh <= 0:
                break
            sat = (tile.amax(0) - tile.amin(0)) / (tile.amax(0) + 1e-6)
            if float((sat > 0.07).float().mean()) >= self.tissue_thresh:
                break
            idx = random.randint(0, self.shard_of.shape[0] - 1)
        slide_stem = rel.split("/", 1)[0]
        patient_id = "-".join(slide_stem.split("-")[:3])
        slide_key = int.from_bytes(hashlib.blake2b(slide_stem.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        patient_key = int.from_bytes(hashlib.blake2b(patient_id.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        # Augmentations are stochastic per view; reproducibility comes from worker seeds.
        global_views = torch.stack([self.global_aug(tile) for _ in range(self.global_views)])
        local_views = torch.stack([self.local_aug(tile) for _ in range(self.local_views)])
        # Discrete: one class id per factor (-1 = no label for this patient). Continuous: one
        # value vector per factor keyed by name (nan-filled if missing). train.py masks missing
        # entries out of each factor's loss. Fixed factor order so train.py can index by position.
        out = {
            "global_views": global_views,
            "local_views": local_views,
            "sample_idx": torch.tensor(int(idx), dtype=torch.int64),
            "slide_id": torch.tensor(slide_key, dtype=torch.int64),
            "patient_id": torch.tensor(patient_key, dtype=torch.int64),
        }
        if self.metadata_enabled:
            disc = [self.discrete_labels[name].get(patient_id, -1) for name in self.metadata_discrete] or [-1]
            out["metadata_labels"] = torch.tensor(disc, dtype=torch.int64)
            for name in self.metadata_continuous:
                v = self.continuous_values[name].get(patient_id)
                v = [float("nan")] * self.continuous_dims[name] if v is None else (v if isinstance(v, list) else [v])
                out[f"metacont_{name}"] = torch.tensor(v, dtype=torch.float32)
        else:
            out["metadata_labels"] = torch.tensor([-1], dtype=torch.int64)
        return out
