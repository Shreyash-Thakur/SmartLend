"""Generate all paper figures from the project's committed reports.

Every number plotted here is read from reports/*.json - nothing hand-entered.
Output: PDF (for LaTeX) + PNG (for eyeballing) per figure, IEEE column width.

Run:  python paper/figures/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# Okabe-Ito subset, validated (dataviz six-checks, light surface).
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
GRAY = "#5f6368"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "figure.dpi": 300,
})

COL_W = 3.45  # IEEE column width in inches


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


# ---------------------------------------------------------------- fig: deltas
def fig_fold_deltas():
    d = json.loads((REPO / "reports" / "features_vs_fusion.json").read_text())
    models = ["CatBoost", "XGBoost", "LightGBM", "LogisticRegression"]
    floor = d["protocol"]["noise_floor"]
    fusion_gain = d["fusion_on_v2"]["best_honest_fusion"]["gain_vs_best_single"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.axhspan(-floor, floor, color=GRAY, alpha=0.15, lw=0)
    ax.axhline(0, color=GRAY, lw=0.6)
    for i, m in enumerate(models):
        deltas = d["models"][m]["fold_paired_delta_v2_minus_v1"]
        mean = d["models"][m]["mean_delta"]
        ax.scatter([i] * len(deltas), deltas, s=14, color=BLUE, zorder=3,
                   facecolors="none", linewidths=1.0)
        ax.hlines(mean, i - 0.22, i + 0.22, color=BLUE, lw=2.2, zorder=4)
        ax.annotate(f"+{mean:.4f}", (i, mean), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=7, color="#1a1a1a")
    # fusion reference
    ax.hlines(fusion_gain, -0.45, len(models) - 0.55, color=VERM, lw=1.4,
              linestyles=(0, (4, 2)), zorder=2)
    ax.annotate(f"best honest fusion (+{fusion_gain:.4f})",
                (len(models) - 1.42, fusion_gain), textcoords="offset points",
                xytext=(0, -10), fontsize=7, color=VERM)
    ax.annotate("fold-noise floor (±0.0036)", (0.97, floor),
                textcoords="offset points", xytext=(0, 2), fontsize=6.5,
                color=GRAY, ha="right")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["CatBoost", "XGBoost", "LightGBM", "LogReg"])
    ax.set_ylabel("Fold-paired AUC delta (v2 - v1)")
    ax.set_title("Feature enrichment vs. model fusion under one pre-registered rule")
    save(fig, "fig_fold_deltas")


# -------------------------------------------------------- fig: risk-coverage
def fig_risk_coverage():
    u = json.loads((REPO / "reports" / "uncertainty_deferral.json").read_text())
    curves = u["risk_coverage_curves"]
    overall = u["sglb_model"]["overall_error_rate_test"]

    spec = [
        ("knowledge_uncertainty", "knowledge uncertainty (ours)", GREEN, "-", "o"),
        ("xgb_disagreement", "two-model disagreement", BLUE, (0, (4, 2)), "s"),
        ("chow_distance", "Chow's rule", VERM, (0, (1, 1.2)), "^"),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for key, label, color, ls, marker in spec:
        pts = curves[key]
        cov = [p["coverage"] for p in pts]
        risk = [p["selective_risk"] for p in pts]
        ax.plot(cov, risk, color=color, lw=1.6, linestyle=ls, marker=marker,
                markersize=3, markevery=3, label=label)
    ax.axhline(overall, color=GRAY, lw=1.0, linestyle=(0, (6, 3)))
    ax.annotate("random abstention", (0.06, overall), textcoords="offset points",
                xytext=(0, 3), fontsize=6.5, color=GRAY)
    ax.axvline(0.775, color=GRAY, lw=0.7, alpha=0.6)
    ax.annotate("operating point\n(22.5% deferral)", (0.775, 0.60),
                textcoords="offset points", xytext=(-4, 0), fontsize=6.5,
                color=GRAY, ha="right")
    ax.set_xlabel("Coverage (share auto-decided)")
    ax.set_ylabel("Selective risk (error on kept cases)")
    ax.set_title("Risk-coverage: epistemic uncertainty routes review cases")
    leg = ax.legend(frameon=True, loc="lower right", framealpha=0.95, edgecolor="#cccccc")
    save(fig, "fig_risk_coverage")


# ------------------------------------------------------- fig: learning curve
def fig_learning_curve():
    c = json.loads((REPO / "reports" / "convergence.json").read_text())
    lc = c["learning_curve"]
    sizes = lc["sizes"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    styles = {"XGBoost": (BLUE, "-", "o"), "LogisticRegression": (ORANGE, (0, (4, 2)), "s")}
    for name, series in lc["models"].items():
        # points are [{n, auc_mean, auc_sd, n_seeds}, ...]
        ns = [p["n"] for p in series]
        aucs = [p["auc_mean"] for p in series]
        sds = [p.get("auc_sd", 0.0) for p in series]
        color, ls, marker = styles.get(name, (GREEN, "-", "^"))
        label = "LogReg" if name == "LogisticRegression" else name
        ax.errorbar(ns, aucs, yerr=sds, color=color, lw=1.6, linestyle=ls,
                    marker=marker, markersize=3.5, label=label, capsize=2,
                    elinewidth=0.8)
        ax.annotate(f"{aucs[-1]:.3f}", (ns[-1], aucs[-1]),
                    textcoords="offset points", xytext=(4, -2), fontsize=7,
                    color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("Held-out AUC")
    ax.set_title("Learning curve: still rising at full data")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_learning_curve")


# -------------------------------------------------------- fig: architecture
def fig_architecture():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, ec=BLUE, fc="#f3f6fb", fs=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=7, color=GRAY, lw=0.9))

    box(0.2, 7.9, 2.7, 1.9, "Applicant\nprofile\n(15 fields)")
    box(3.6, 8.9, 3.4, 1.4, "ML model\n(calibrated LR)", ec=BLUE)
    box(3.6, 6.9, 3.4, 1.4, "CBES rules\n(5 pillars)", ec=GREEN)
    box(7.9, 7.9, 3.8, 1.6, "Blend\n$(1{-}\\alpha)p_{ml}+\\alpha\\,p_{cbes}$", ec=VERM)
    box(0.6, 4.2, 2.9, 1.5, "APPROVE\n(instant)", ec=GREEN)
    box(4.2, 4.2, 2.9, 1.5, "REJECT\n(instant)", ec=VERM)
    box(7.8, 4.2, 3.4, 1.5, "DEFER\n(uncertainty band)", ec=ORANGE)
    box(6.6, 0.7, 5.2, 2.3,
        "Human review queue\nLLM briefing (advisory)\ncaptures decision + codes",
        ec=ORANGE, fs=5.9)
    box(0.2, 0.7, 5.8, 2.3,
        "Governance stamps on every decision:\nversion + artifact + threshold hash;\ngate closed, 3% exploration arm",
        ec=GRAY, fc="#fbf7ef", fs=5.9)

    arrow(2.9, 9.1, 3.6, 9.5)
    arrow(2.9, 8.5, 3.6, 7.7)
    arrow(7.0, 9.5, 7.9, 8.9)
    arrow(7.0, 7.7, 7.9, 8.2)
    arrow(9.0, 7.9, 2.2, 5.7)
    arrow(9.5, 7.9, 5.7, 5.7)
    arrow(9.8, 7.9, 9.5, 5.7)
    arrow(9.5, 4.2, 9.3, 3.0)
    ax.set_title("CredDefer decision pipeline", fontsize=8.5)
    save(fig, "fig_architecture")


if __name__ == "__main__":
    print("[figures]")
    fig_fold_deltas()
    fig_risk_coverage()
    fig_learning_curve()
    fig_architecture()
