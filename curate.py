# curate.py -- SemDeDup-style semantic data curation for nanopath (preprocessing, rules-legal:
# uses a non-pathology frozen DINOv2 for curation only, generated before the capped run and not
# counted against training FLOP/sample caps). Embeds every training tile with frozen DINOv2,
# k-means clusters the L2-normalised CLS embeddings, then within each cluster greedily drops
# semantic near-duplicates (cosine similarity above `tau` to an already-kept tile). Writes a
# manifest of kept (shard_of, row_of) pairs; dataloader.py restricts the training index to it
# when data.curation_manifest is set. The point: spend the fixed 1M tile-presentation budget on
# a deduplicated, diverse subset so each presentation carries more unique information.
#
# Usage (on a GPU node):  python curate.py configs/main.yaml
#   optional overrides:   max_shards=N (test on a subset), tau=0.95 (dedup threshold),
#                         k_clusters=10000, out=<manifest path>, batch_size=256, num_workers=16

import io
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from dataloader import patient_id_from_relpath, patient_in_val
from model import DinoV2ViT, load_dinov2_pretrained


# Minimal read-only tile dataset: one resized/normalised tile + its global index, no augmentation.
class EmbedTiles(Dataset):
    def __init__(self, shards, shard_of, row_of, mean, std):
        self.shards, self.shard_of, self.row_of = shards, shard_of, row_of
        self._readers = [None] * len(shards)
        self.tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                              v2.Resize((224, 224), antialias=True), v2.Normalize(mean=mean, std=std)])

    def __len__(self): return int(self.shard_of.shape[0])

    def __getitem__(self, i):
        si, ri = int(self.shard_of[i]), int(self.row_of[i])
        if self._readers[si] is None:
            self._readers[si] = pq.ParquetFile(str(self.shards[si]), memory_map=True)
        rg = self._readers[si].metadata.row_group(0).num_rows
        table = self._readers[si].read_row_group(ri // rg, columns=["jpeg"])
        with Image.open(io.BytesIO(table["jpeg"][ri % rg].as_py())) as img:
            return self.tf(img.convert("RGB")), i


def main():
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    ov = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    data = cfg["data"]
    dataset_dir = Path(data["dataset_dir"])
    shards = sorted(dataset_dir.glob("shard-*.parquet"))[: int(ov.get("max_shards", 10**9))]
    tau, k_clusters = float(ov.get("tau", 0.95)), int(ov.get("k_clusters", 10000))
    out = Path(ov.get("out", dataset_dir / "curation_manifest.npz"))
    device = torch.device("cuda")

    # Full training index (same patient split as dataloader.py so the manifest is drop-in).
    shard_of, row_of = [], []
    for si, sp in enumerate(shards):
        paths = pq.read_table(str(sp), columns=["path"], memory_map=True)["path"].to_pylist()
        for ri, p in enumerate(paths):
            if not patient_in_val(patient_id_from_relpath(p), data["split_seed"], data["val_fraction"]):
                shard_of.append(si); row_of.append(ri)
    shard_of, row_of = np.asarray(shard_of, np.int32), np.asarray(row_of, np.int32)
    n = len(shard_of)
    print(f"[curate] train tiles: {n}  shards: {len(shards)}", flush=True)

    # Frozen DINOv2 CLS embeddings (bf16 forward), L2-normalised, stored fp16.
    model = load_dinov2_pretrained(DinoV2ViT(variant=cfg["model"]["type"])).to(device).eval()
    loader = DataLoader(EmbedTiles(shards, shard_of, row_of, data["mean"], data["std"]),
                        batch_size=int(ov.get("batch_size", 256)), num_workers=int(ov.get("num_workers", 16)),
                        pin_memory=True, prefetch_factor=4, persistent_workers=False)
    emb = np.zeros((n, model.embed_dim), np.float16)
    done = 0
    for x, idx in loader:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            f = F.normalize(model(x.to(device, non_blocking=True))["x_norm_clstoken"].float(), dim=-1)
        emb[idx.numpy()] = f.cpu().numpy().astype(np.float16)
        done += len(idx)
        if done % 200000 < int(ov.get("batch_size", 256)):
            print(f"[curate] embedded {done}/{n}", flush=True)

    # Cluster once, then SemDeDup inside each cluster: keep tiles ordered by proximity to the
    # centroid, dropping any whose cosine similarity to an already-kept tile exceeds tau.
    print(f"[curate] clustering into {k_clusters} clusters", flush=True)
    embf = emb.astype(np.float32)
    labels = MiniBatchKMeans(n_clusters=k_clusters, batch_size=8192, n_init=3, max_iter=100, random_state=0).fit_predict(embf)
    keep = np.zeros(n, bool)
    for c in range(k_clusters):
        members = np.where(labels == c)[0]
        if len(members) <= 1:
            keep[members] = True
            continue
        vecs = embf[members]  # (M, D), unit-norm
        order = np.argsort(-(vecs @ vecs.mean(0)))  # local indices, closest-to-centroid first
        sim = vecs @ vecs.T  # (M, M) cosine; per-cluster matrix stays small
        kept_local = []
        for li in order:
            if not kept_local or sim[li, kept_local].max() < tau:
                kept_local.append(li)
        keep[members[kept_local]] = True
    n_keep = int(keep.sum())
    print(f"[curate] kept {n_keep}/{n} tiles ({100*n_keep/n:.1f}%) at tau={tau}", flush=True)
    np.savez(out, shard_of=shard_of[keep], row_of=row_of[keep])
    print(f"[curate] wrote manifest -> {out}", flush=True)


if __name__ == "__main__":
    main()
