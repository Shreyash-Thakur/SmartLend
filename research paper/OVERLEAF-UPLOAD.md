# How to build the paper on Overleaf (step by step)

Everything the paper needs lives in this folder. The last Overleaf attempt
rendered with missing graphs because the figure PDFs were never uploaded; this
checklist exists so that cannot happen again.

## What the paper is

`creddefer_rait2027.tex` — "CredDefer: Capacity-Constrained, Uncertainty-Aware
Human Review for Credit Risk Decisions" (IEEE conference format, authors:
Shreyash Thakur, Shweta Karandikar, Yuvaan Shah, Bhakti Palkar). Core result:
a permutation-test audit that caught the production deferral rule routing the
model's MOST confident cases to human review, the diagnosis (calibration
offset between the two scores), and the measured repair that now runs as the
production default.

## Files to upload — exact names

**1. The main file (project root, do not rename):**

```
creddefer_rait2027.tex
```

**2. The figures — create a folder named exactly `figures` inside the Overleaf
project, then upload these 6 PDFs INTO that folder (exact names, PDF versions,
not the PNGs):**

```
figures/fig_architecture.pdf
figures/fig_audit_null.pdf
figures/fig_uncertainty_deciles.pdf
figures/fig_risk_coverage.pdf
figures/fig_cost_frontier.pdf
figures/fig_fold_deltas.pdf
```

That is the complete list — the tex references these six and nothing else
(`grep includegraphics creddefer_rait2027.tex` to verify). Two more figures
exist in this folder (`fig_fairness.pdf`, `fig_learning_curve.pdf`) but are
NOT referenced by the current draft: do not upload them unless you first add
them to the tex. The `.png` files are quick-look previews for GitHub only;
Overleaf does not need them.

## Overleaf steps

1. overleaf.com → New Project → **Blank Project** (name it CredDefer).
2. Delete the auto-created `main.tex`.
3. Upload `creddefer_rait2027.tex` (top-left upload icon).
4. Menu → Settings → **Main document** = `creddefer_rait2027.tex`,
   **Compiler** = pdfLaTeX (default), TeX Live 2023 or newer.
5. New Folder → name it exactly `figures` (lowercase).
6. With the `figures` folder selected, upload the 6 PDFs listed above.
   If a figure lands in the project root by mistake, drag it into `figures/`.
7. Recompile. The PDF should be ~8 pages with 6 figures and no
   "File not found" boxes.

**If a figure shows as a missing-file box:** the file is either not inside the
`figures` folder or its name does not match exactly (case-sensitive).

## Rules that bind any edits to the tex

- **No em dashes anywhere in the paper.** Use commas, colons or parentheses.
  `python dedash.py` (in this folder) checks/strips them; `python check_tex.py`
  runs basic sanity checks.
- **Every number in the paper traces to a committed `reports/*.json`** in this
  repository. Do not adjust a number in the tex without the report backing it.
- Figures are generated, never hand-edited: `figures/make_figures.py`
  regenerates all of them from the committed reports (run from the repo root
  with the project `.venv`).
- Search the tex for `VERIFY` comments before submission: the venue is
  undecided (RAIT 2027 was missed; shortlist and live CFPs are in
  `docs/RESEARCH-SCOPE-IDEAS.md`), and the author block is currently non-blind
  by choice — re-check against the chosen venue's blinding rules.
- `litreview_sources.txt` holds the sources behind the related-work section.

## Continuing the research

Read `docs/RESEARCH-SCOPE-IDEAS.md` first: it has the novelty check on this
paper's core claim (it survives, with three near-neighbor papers that must be
cited and distinguished), the ranked list of additions worth making, and the
regulatory claims that still need primary-source verification.
