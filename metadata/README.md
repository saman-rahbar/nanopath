# metadata

`fino_meta.json` — a single discrete factor (`subtype`: TCGA cancer subtype, 40 classes,
barcode -> class id) used as an optional auxiliary classification target on the CLS token
(`metadata.enabled` in `configs/main.yaml`). Public TCGA clinical metadata, not derived from
`probe.py` or `benchmarking/`.

Trimmed from the full multi-factor FINO metadata artifact published on the `block-strided-cls`
branch of this repo (22 discrete + 11 continuous factors, ~80MB) down to just the one factor
actually used here, to keep this branch's diff minimal. See that branch's `metadata/README.md`
for the full factor list and provenance of the original artifact.
