"""Model comparison figure: ROC curves + AUC bars, for the defense.

Curves are computed from backend/artifacts/prediction_outputs.csv (307,511
out-of-fold rows; y_true == 1 means GOOD, probabilities are approval
probabilities, so P(default) = 1 - prob). Bar values not in that file are
read from the reports named next to each one.

Run: .venv/bin/python docs/figures/make_model_auc_comparison.py
Writes docs/figures/model_auc_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("model_auc_comparison.png")

# Dark surface + text tokens, categorical slots from the dataviz palette.
SURFACE = "#1a1a19"
TEXT = "#f0efec"
TEXT_2 = "#c3c2b7"
MUTED = "#8a8980"
GRID = "#33332f"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = (
    "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9")


def roc(y_default: np.ndarray, score: np.ndarray, points: int = 400):
    order = np.argsort(-score, kind="mergesort")
    y = y_default[order]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()])
    idx = np.unique(np.linspace(0, len(tpr) - 1, points).astype(int))
    return fpr[idx], tpr[idx]


def auc(y_default: np.ndarray, score: np.ndarray) -> float:
    r = pd.Series(score).rank().to_numpy()
    n1 = y_default.sum()
    n0 = len(y_default) - n1
    return float((r[y_default == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    df = pd.read_csv(ROOT / "backend/artifacts/prediction_outputs.csv")
    y_default = (df["y_true"] == 0).astype(int).to_numpy()

    curves = [  # (column, label, colour, width)
        ("prob_XGBoost", "XGBoost", BLUE, 2.6),
        ("prob_CatBoost", "CatBoost", AQUA, 1.6),
        ("prob_LightGBM", "LightGBM", VIOLET, 1.6),
        ("prob_Logistic Regression", "Logistic Regression", YELLOW, 1.8),
        ("prob_Random Forest", "Random Forest", MAGENTA, 1.8),
        ("prob_CBES", "CBES (rule score)", ORANGE, 1.8),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": TEXT, "axes.labelcolor": TEXT_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig = plt.figure(figsize=(18, 9.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.62,
                          left=0.06, right=0.97, top=0.83, bottom=0.17)

    # ---- left: ROC curves -------------------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color=MUTED, lw=1.4)
    ax.text(0.86, 0.75, "coin flip (AUC 0.50)", color=MUTED, rotation=38,
            fontsize=11, ha="center")

    for col, label, colour, width in curves:
        score = 1.0 - df[col].to_numpy()  # P(default)
        fpr, tpr = roc(y_default, score)
        a = auc(y_default, score)
        if col == "prob_XGBoost":
            ax.fill_between(fpr, tpr, 0, color=BLUE, alpha=0.13, lw=0)
        ax.plot(fpr, tpr, color=colour, lw=width,
                label=f"{label}  {a:.4f}", zorder=3 if col == "prob_XGBoost" else 2)

    ax.text(0.10, 0.04, "shaded = AUC of XGBoost\n(0.765 of the box)",
            color=TEXT_2, fontsize=10.5, ha="left")
    ax.annotate("XGBoost, CatBoost, LightGBM lie\non top of each other: same mistakes",
                xy=(0.18, 0.565), xytext=(0.02, 0.88), color=TEXT, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=1))
    ax.annotate("Logistic Regression is hidden\nunder Random Forest (0.7378 vs 0.7376)",
                xy=(0.30, 0.66), xytext=(0.30, 0.30), color=TEXT, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=1))
    ax.annotate("CBES: barely above\nthe coin flip", xy=(0.66, 0.74),
                xytext=(0.70, 0.50), color=TEXT, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Good customers wrongly flagged as risky  (false-positive rate)")
    ax.set_ylabel("Defaulters caught  (true-positive rate)")
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=11,
                    title="ROC-AUC (out-of-fold, 307,511 rows)",
                    title_fontsize=11, labelcolor=TEXT)
    leg.get_title().set_color(TEXT_2)
    ax.set_title("ROC curves: the higher and further left, the better",
                 color=TEXT, fontsize=14, loc="left", pad=12)

    # ---- right: every model's AUC ----------------------------------------
    bars = [  # (label, auc, sample note, role)
        ("CatBoost, FE-v3 (345 features)", 0.7868, "5-fold OOF · features_v3_side_tables.json", "best"),
        ("CatBoost, FE-v2 (319 features)", 0.7757, "5-fold OOF · features_vs_fusion.json", "research"),
        ("XGBoost", auc(y_default, 1 - df["prob_XGBoost"].to_numpy()), "5-fold OOF · prediction_outputs.csv", "research"),
        ("CatBoost", auc(y_default, 1 - df["prob_CatBoost"].to_numpy()), "5-fold OOF", "research"),
        ("LightGBM", auc(y_default, 1 - df["prob_LightGBM"].to_numpy()), "5-fold OOF", "research"),
        ("Logistic Regression (research)", auc(y_default, 1 - df["prob_Logistic Regression"].to_numpy()), "5-fold OOF", "research"),
        ("Random Forest", auc(y_default, 1 - df["prob_Random Forest"].to_numpy()), "5-fold OOF", "research"),
        ("TabPFN-2.5 (no training)", 0.7284, "8,000 rows · tabpfn_quick_run.json", "research"),
        ("LIVE: calibrated Logistic Regression", 0.6919, "15 form fields · 61,503 test rows", "live"),
        ("Google TabFM (no training)", 0.6911, "8,000 rows · tabfm_quick_run.json", "research"),
        ("CBES rule score", auc(y_default, 1 - df["prob_CBES"].to_numpy()), "no training · 307,511 rows", "research"),
    ]
    colour_for = {"best": BLUE, "live": ORANGE, "research": "#5b5a55"}

    bx = fig.add_subplot(gs[1])
    bx.set_facecolor(SURFACE)
    ypos = np.arange(len(bars))[::-1]
    base = 0.5
    for yp, (label, a, note, role) in zip(ypos, bars):
        bx.barh(yp, a - base, left=base, height=0.62, color=colour_for[role], lw=0)
        bx.text(a + 0.003, yp, f"{a:.4f}", va="center", color=TEXT, fontsize=12,
                fontweight="bold" if role != "research" else "normal")
        bx.text(base - 0.004, yp + 0.02, label, va="center", ha="right", color=TEXT,
                fontsize=12, fontweight="bold" if role != "research" else "normal")
        bx.text(base - 0.004, yp - 0.33, note, va="center", ha="right",
                color=MUTED, fontsize=9)

    # the v1 boosted trees sit inside the fold noise of each other
    y_top, y_bot = ypos[2] + 0.42, ypos[4] - 0.42
    bx.plot([0.807, 0.807], [y_bot, y_top], color=TEXT_2, lw=1.2)
    bx.text(0.811, (y_top + y_bot) / 2, "within ±0.0036\nnoise: a tie",
            va="center", color=TEXT_2, fontsize=10)

    bx.set_xlim(base, 0.86)
    bx.set_ylim(-0.7, len(bars) - 0.3)
    bx.set_yticks([])
    bx.set_xlabel("ROC-AUC  (axis starts at 0.50 = coin flip)")
    bx.grid(axis="x", color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    for s in bx.spines.values():
        s.set_visible(False)
    bx.set_title("Every model we scored", color=TEXT, fontsize=14, loc="left", pad=12)

    # ---- title + why-selected notes ---------------------------------------
    fig.text(0.06, 0.94, "Which model, and why", fontsize=22, fontweight="bold", color=TEXT)
    fig.text(0.06, 0.895,
             "Home Credit, 307,511 applicants, 8.07% defaulted. AUC = chance the model "
             "ranks a random defaulter as riskier than a random good customer.",
             fontsize=12.5, color=TEXT_2)

    fig.patches.append(plt.Rectangle((0.06, 0.035), 0.012, 0.03, color=BLUE,
                                     transform=fig.transFigure))
    fig.text(0.078, 0.05,
             "Best research model: CatBoost FE-v3. Same model family as the rest; "
             "the gain came from better features (0.7603 → 0.7757 → 0.7868), not a new algorithm.",
             fontsize=12, color=TEXT, va="center")
    fig.patches.append(plt.Rectangle((0.06, 0.085), 0.012, 0.03, color=ORANGE,
                                     transform=fig.transFigure))
    fig.text(0.078, 0.10,
             "Live model: calibrated Logistic Regression. The form only collects 15 fields; "
             "it is calibrated (0.9 means 90% repay), fast, and SHAP is exact for it.",
             fontsize=12, color=TEXT, va="center")

    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
