# Model weights

Weights are **not committed**. This directory holds files fetched from
elsewhere, and one of them carries a licence that makes redistribution
inadvisable. `.gitignore` excludes `*.ckpt`, `*.safetensors` and `*.bin`.

## TabPFN-2.5

A tabular foundation model from Prior Labs, evaluated as a candidate
classifier. See `reports/tabpfn_comparison.json` for the outcome.

### Fetching the checkpoint

```bash
pip install tabpfn
python - <<'PY'
import urllib.request, os
os.makedirs("models/tabpfn", exist_ok=True)
urllib.request.urlretrieve(
    "https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/main/"
    "tabpfn-v2.5-classifier-v2.5_default.ckpt",
    "models/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt")
PY
```

The HuggingFace repo is public and ungated (`gated: false`), so no
token is required for the download itself. The `tabpfn` package's own
downloader additionally opens a browser for licence acceptance; passing
`model_path` explicitly bypasses that flow, but **the licence still
applies** — see below.

### Pin the version

`pip install tabpfn` does **not** give you v2.5. The package default
moved across releases (v6 → 2.5, v7 → 2.6, v8+ → TabPFN-3), so an
unpinned install silently benchmarks a different model:

```python
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier(
    model_path="models/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt",
    device="cuda",
    categorical_features_indices=cat_idx,
)
```

### Hardware

Prior Labs recommend A100/H100-class GPUs. TabPFN holds the full
training context in VRAM for every forward pass, at cost quadratic in
rows — so **training-set size, not prediction batch size, is the
binding constraint**. On an 8 GB laptop GPU this dataset exhausts
memory even with 500-row prediction batches. Reducing the batch does
not help; only reducing training rows does.

### Licence — read before using outputs

Weights are released under the **TabPFN-2.5 License v1.1**, which is
non-commercial only:

> You may only access, use, Distribute, or create Derivatives of the
> TABPFN-2.5 Model or Derivatives for Non-Commercial Purposes.

The restriction covers **outputs** — predictions and probabilities —
not just the weights. Academic research and internal benchmarking are
explicitly permitted, so use in this project and its paper is fine.
Deploying SmartLend commercially with TabPFN in the loop would not be.

This belongs in the limitations section of any write-up that reports
TabPFN results.

### Citing it correctly

The widely-cited *Nature* paper describes **TabPFNv2** (10,000 samples,
500 features), **not** v2.5 (50,000 × 2,000). They are different
models. Cite:

- **Method lineage** — Hollmann, N., Müller, S., Purucker, L.,
  Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., &
  Hutter, F. (2025). Accurate predictions on small data with a tabular
  foundation model. *Nature*, 637(8045), 319–326.
  https://doi.org/10.1038/s41586-024-08328-6
- **v2.5 specifically** — Grinsztajn, L., et al. (2025). *TabPFN-2.5:
  Advancing the State of the Art in Tabular Foundation Models.*
  arXiv:2511.08667 — **a preprint, not peer-reviewed.**

Attributing a v2.5 capability claim to the *Nature* paper is a factual
error and an easy one to make.
