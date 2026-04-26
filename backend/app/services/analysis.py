import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from pathlib import Path
from typing import Dict, List, Any

# Configure paths
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

def _ensure_dir():
    """Ensure artifacts/plots directory exists."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_roc_curves(y_true: np.ndarray, model_probs_dict: Dict[str, np.ndarray], save_path: Path | None = None) -> None:
    """Generate ROC curves for all evaluated models."""
    _ensure_dir()
    plt.figure(figsize=(10, 8))
    
    for model_name, y_prob in model_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {model_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves Comparison Across Models", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    out_path = save_path or (PLOTS_DIR / "roc_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_pml_vs_pcbes_scatter(p_ml: np.ndarray, p_cbes: np.ndarray, decisions: np.ndarray | None = None, save_path: Path | None = None) -> None:
    """Scatter plot illustrating structural alignment or disagreement between ML and CBES paradigms."""
    _ensure_dir()
    plt.figure(figsize=(9, 8))
    
    if decisions is not None:
        # We can hue by decision if provided
        sns.scatterplot(x=p_ml, y=p_cbes, hue=decisions, palette="Set1", alpha=0.6, s=50)
    else:
        sns.scatterplot(x=p_ml, y=p_cbes, alpha=0.5, color="teal", s=50)
        
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Agreement")
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Machine Learning Probability (p_ml)", fontsize=12)
    plt.ylabel("CBES Prior Probability (p_cbes)", fontsize=12)
    plt.title("ML vs CBES Decision Terrain", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    out_path = save_path or (PLOTS_DIR / "pml_vs_pcbes_scatter.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_disagreement_histogram(p_ml: np.ndarray, p_cbes: np.ndarray, tau_d: float, save_path: Path | None = None) -> None:
    """Histogram of Disagreement (D) tracking the density mapping against TAU_D threshold."""
    _ensure_dir()
    plt.figure(figsize=(10, 6))
    
    D = np.abs(p_ml - p_cbes)
    sns.histplot(D, bins=40, kde=True, color="indigo", alpha=0.5)
    
    plt.axvline(tau_d, color='red', linestyle='dashed', linewidth=2, label=f'TAU_D Barrier ({tau_d})')
    
    plt.xlabel("Disagreement Magnitude |p_ml - p_cbes|", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Model Disagreement", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    out_path = save_path or (PLOTS_DIR / "disagreement_histogram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_deferral_vs_accuracy(curve_data: List[Dict[str, float]], save_path: Path | None = None) -> None:
    """Plots the relationship mapping deferral rate scaling vs accuracy tradeoffs during calibration."""
    _ensure_dir()
    if not curve_data:
        return
        
    df_curve = pd.DataFrame(curve_data)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Disagreement Tolerance (tau_d)', fontsize=12)
    ax1.set_ylabel('Deferral Rate', color=color, fontsize=12)
    ax1.plot(df_curve['tau_d'], df_curve['deferral_rate'], color=color, lw=2, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(0.22, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(0.28, color='gray', linestyle=':', alpha=0.5)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Non-Deferred Accuracy', color=color, fontsize=12)
    ax2.plot(df_curve['tau_d'], df_curve['non_deferred_accuracy'], color=color, lw=2, marker='s', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Calibration Tradeoffs: Abstention Rate vs Maintained Accuracy", fontsize=14)
    fig.tight_layout()
    plt.grid(alpha=0.3)
    
    out_path = save_path or (PLOTS_DIR / "deferral_vs_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_confidence_distribution(confidences: np.ndarray, save_path: Path | None = None) -> None:
    """Visualizes confidence volume mapping highlighting low-trust zones."""
    _ensure_dir()
    plt.figure(figsize=(10, 6))
    
    sns.histplot(confidences, bins=30, kde=True, fill=True, color="darkorange", alpha=0.6)
    
    plt.axvline(0.15, color='red', linestyle='dashed', linewidth=2, label='Low Confidence Cutoff (0.15)')
    plt.axvline(0.55, color='gray', linestyle=':', linewidth=1.5, label='Medium Confidence (0.55)')
    plt.axvline(0.75, color='green', linestyle=':', linewidth=1.5, label='High Confidence (0.75)')
    
    plt.xlim(0, 1)
    plt.xlabel("Confidence Score", fontsize=12)
    plt.ylabel("Application Count", fontsize=12)
    plt.title("System-Wide Decision Confidence Spread", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    out_path = save_path or (PLOTS_DIR / "confidence_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_cbes_radar_chart(cbes_breakdown: Dict[str, float], title: str = "CBES Evaluation Radar", save_path: Path | None = None) -> None:
    """Renders a 5Cs spider/radar chart displaying an applicant's component performance."""
    _ensure_dir()
    
    labels = list(cbes_breakdown.keys())
    stats = list(cbes_breakdown.values())
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    
    # Close the loop
    stats += stats[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.fill(angles, stats, color='teal', alpha=0.25)
    ax.plot(angles, stats, color='teal', linewidth=2)
    
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", size=8)
    ax.set_ylim(0, 1)
    
    ax.set_xticks(angles[:-1])
    # Capitalize labels cleanly
    ax.set_xticklabels([label.upper() for label in labels], fontsize=10, weight="bold")
    
    plt.title(title, size=15, color='black', y=1.1)
    
    out_path = save_path or (PLOTS_DIR / "cbes_radar_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
