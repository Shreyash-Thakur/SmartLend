"""Which v2 features carry the +0.015? Appends the answer to the report.

Fits one early-stopped LightGBM on the full v2 matrix (same params as the
experiment's folds) and appends the top-25 gain importances to
reports/features_vs_fusion.json under `top_features_lightgbm_gain`, marking
which are ENGINEERED (APP_/BURO_ prefixes) vs native columns — the number a
reviewer asks for right after seeing the headline.

Run:  python -m research.analysis.v2_feature_importances
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import train_test_split

from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "reports" / "features_vs_fusion.json"

ENGINEERED_PREFIXES = ("APP_", "BURO_")


def main() -> None:
    X, y, _ = build_feature_matrix()
    X_fit, X_es, y_fit, y_es = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y)
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=63,
        random_state=42, n_jobs=-1, verbose=-1, importance_type="gain")
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])

    total = float(model.feature_importances_.sum())
    ranked = sorted(zip(X.columns, model.feature_importances_),
                    key=lambda kv: kv[1], reverse=True)
    top = [
        {
            "feature": name,
            "gain_share": round(float(gain) / total, 4),
            "engineered": name.startswith(ENGINEERED_PREFIXES),
        }
        for name, gain in ranked[:25]
    ]
    engineered_share_top25 = round(sum(t["gain_share"] for t in top if t["engineered"]), 4)

    results = json.loads(RESULTS.read_text())
    results["top_features_lightgbm_gain"] = {
        "note": "single early-stopped LightGBM on the full v2 matrix, gain importance",
        "engineered_gain_share_within_top25": engineered_share_top25,
        "top_25": top,
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"engineered features carry {engineered_share_top25:.0%} of top-25 gain")
    for t in top[:15]:
        tag = "ENGINEERED" if t["engineered"] else "native"
        print(f"  {t['gain_share']:6.2%}  {t['feature']}  [{tag}]")


if __name__ == "__main__":
    main()
