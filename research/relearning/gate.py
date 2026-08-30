"""Gate evaluator for the relearning loop.

Answers one question from real data: *is it safe to start retraining on
human-reviewed deferred cases yet?*

The authoritative definition of the gate is
``docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md`` section 3
("Gate to open the loop"). All four conditions must hold:

1. Deferral rule beats random at isolating hard cases (CoDoC pattern: separate
   accuracy on the auto-decided and deferred subsets against a no-deferral
   baseline).
2. Observed override/defer rate sits within the AUC-implied natural-rate bound
   (Tasche, "Bounds for rating override rates").
3. The exploration arm has accumulated enough un-selected labels for an
   unbiased evaluation set.
4. The retraining design explicitly models the missingness/selection mechanism
   (cf. RMT-Net) and reviewer bias/consistency (Madras et al.), rather than
   treating human decisions as ground truth.

The danger being guarded against is the runaway feedback loop (Ensign et al.):
retraining on the cases a broken router selected reinforces the router's bias.
So the default verdict is refusal - the loop stays shut unless every condition
passes.

LABEL SEMANTICS (getting this wrong inverts every conclusion):
  * ``y_true == 1``  -> APPROVE-worthy, i.e. a GOOD customer who did NOT default.
  * ``best_model_prob`` / ``prob_*`` are APPROVAL probabilities (P(good)).
  * ``final_decision`` is one of APPROVE / REJECT / DEFER.

This module is read-only with respect to ``backend/``: it reads the prediction
artifact and writes only to ``reports/``.

Run: ``python -m research.relearning.gate``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = REPO_ROOT / "backend" / "artifacts" / "prediction_outputs.csv"
DEFAULT_REPORT = REPO_ROOT / "reports" / "relearning_gate.json"

DEFER = "DEFER"
RANDOM_TRIALS = 200
RANDOM_SEED = 20260830

# Condition 1 decision rule: how many standard deviations below the random-router
# baseline the actual router must sit before we call it "better than random at
# isolating hard cases".
Z_THRESHOLD = 2.0

# Condition 3: how many un-selected (exploration-arm) labels would be needed
# before an unbiased evaluation set exists. Sized so that a subset AUC has a
# usable confidence interval at this dataset's ~8% bad rate.
MIN_EXPLORATION_LABELS = 1000

# Condition 4: paths that would hold a written retraining design, and the
# mechanisms such a design must demonstrably address.
RETRAINING_DESIGN_CANDIDATES = (
    Path("docs") / "retraining-design.md",
    Path("docs") / "superpowers" / "specs" / "retraining-design.md",
    Path("research") / "relearning" / "retraining_design.md",
)
RETRAINING_DESIGN_REQUIRED_TOPICS = (
    "selection mechanism / missingness model (RMT-Net)",
    "reviewer bias and consistency model (Madras et al.)",
    "explicit rejection of human_decision-as-ground-truth",
)

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"


@dataclass
class ConditionResult:
    """One gate condition's verdict plus the numbers behind it."""

    number: int
    name: str
    status: str
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.number,
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "metrics": _jsonable(self.metrics),
            "notes": self.notes,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


# --------------------------------------------------------------------------
# subset metrics
# --------------------------------------------------------------------------


def _predicted_good(frame: pd.DataFrame) -> np.ndarray:
    """Predicted label under a NO-DEFERRAL baseline.

    Every case is forced to a hard approve/reject at the engine's own approval
    threshold, including the ones the router actually deferred. This is the
    counterfactual the CoDoC pattern compares against: what the model would have
    done unaided.
    """
    prob = frame["best_model_prob"].to_numpy(dtype=float)
    if "approval_threshold" in frame.columns:
        threshold = frame["approval_threshold"].to_numpy(dtype=float)
    else:
        threshold = np.full(len(frame), 0.5)
    return (prob >= threshold).astype(int)


def subset_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Size, good-share, balance distance, accuracy and AUC for one subset."""
    n = int(len(frame))
    if n == 0:
        return {
            "n": 0,
            "good_share": None,
            "balance_distance": None,
            "accuracy": None,
            "auc": None,
        }

    y = frame["y_true"].to_numpy(dtype=int)
    good_share = float(y.mean())
    accuracy = float((_predicted_good(frame) == y).mean())

    # AUC needs both classes present in the subset.
    auc: float | None
    if len(np.unique(y)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y, frame["best_model_prob"].to_numpy(dtype=float)))

    return {
        "n": n,
        "good_share": good_share,
        # 0.0 == perfectly balanced 50/50 (maximally hard); 0.5 == all one class.
        "balance_distance": float(abs(good_share - 0.5)),
        "accuracy": accuracy,
        "auc": auc,
    }


def _random_router_baseline(
    frame: pd.DataFrame, n_defer: int, trials: int, seed: int
) -> dict[str, Any]:
    """Distribution of deferred-subset statistics under a UNIFORM random router.

    A random router that defers the same volume is the null hypothesis. If the
    real router isolates hard cases it must produce a deferred pile that is
    *more* balanced (lower ``balance_distance``) and on which the model is
    *less* accurate than random selection would give.
    """
    rng = np.random.default_rng(seed)
    n = len(frame)
    y = frame["y_true"].to_numpy(dtype=int)
    predicted = _predicted_good(frame)
    prob = frame["best_model_prob"].to_numpy(dtype=float)

    good_shares = np.empty(trials)
    balance = np.empty(trials)
    accuracies = np.empty(trials)
    aucs = np.full(trials, np.nan)

    for i in range(trials):
        idx = rng.choice(n, size=n_defer, replace=False)
        ys = y[idx]
        share = float(ys.mean())
        good_shares[i] = share
        balance[i] = abs(share - 0.5)
        accuracies[i] = float((predicted[idx] == ys).mean())
        if len(np.unique(ys)) == 2:
            aucs[i] = float(roc_auc_score(ys, prob[idx]))

    def _dist(values: np.ndarray) -> dict[str, Any]:
        clean = values[~np.isnan(values)]
        if clean.size == 0:
            return {"mean": None, "std": None, "p05": None, "p95": None}
        return {
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
            "p05": float(np.percentile(clean, 5)),
            "p95": float(np.percentile(clean, 95)),
        }

    return {
        "trials": trials,
        "seed": seed,
        "good_share": _dist(good_shares),
        "balance_distance": _dist(balance),
        "accuracy": _dist(accuracies),
        "auc": _dist(aucs),
    }


def _z_score(observed: float | None, dist: dict[str, Any]) -> float | None:
    """Standard deviations of ``observed`` above the random baseline mean."""
    if observed is None or dist.get("mean") is None:
        return None
    std = dist.get("std") or 0.0
    if std <= 0:
        return None
    return float((observed - dist["mean"]) / std)


# --------------------------------------------------------------------------
# Condition 1
# --------------------------------------------------------------------------


def evaluate_condition_1(
    frame: pd.DataFrame, trials: int = RANDOM_TRIALS, seed: int = RANDOM_SEED
) -> ConditionResult:
    """Deferral rule beats random at isolating hard cases.

    A working router hands the HARD cases to humans. Hard means: near the
    decision boundary, so the good/bad mix in the deferred pile should be closer
    to 50/50 than the auto-decided pile, and the model should be measurably
    *less* accurate there. We test both against a random router that defers the
    same number of cases.
    """
    name = "Deferral rule beats random at isolating hard cases"
    deferred_mask = frame["final_decision"].astype(str).str.upper() == DEFER
    deferred = frame.loc[deferred_mask]
    auto = frame.loc[~deferred_mask]

    if len(deferred) == 0:
        return ConditionResult(
            1,
            name,
            FAIL,
            "no deferred cases in the artifact - nothing to validate the router on",
            {"n_total": int(len(frame)), "n_deferred": 0},
        )
    if len(auto) == 0:
        return ConditionResult(
            1,
            name,
            FAIL,
            "every case was deferred - there is no auto-decided subset to compare against",
            {"n_total": int(len(frame)), "n_deferred": int(len(deferred))},
        )

    deferred_stats = subset_metrics(deferred)
    auto_stats = subset_metrics(auto)
    overall_stats = subset_metrics(frame)
    baseline = _random_router_baseline(frame, len(deferred), trials, seed)

    balance_z = _z_score(deferred_stats["balance_distance"], baseline["balance_distance"])
    accuracy_z = _z_score(deferred_stats["accuracy"], baseline["accuracy"])

    # Direction of the two tests. Both must point at "harder than random".
    harder_mix = (
        balance_z is not None and balance_z <= -Z_THRESHOLD
    )  # closer to 50/50 than random
    harder_for_model = (
        accuracy_z is not None and accuracy_z <= -Z_THRESHOLD
    )  # model does worse there than random selection

    metrics = {
        "auto_decided": auto_stats,
        "deferred": deferred_stats,
        "overall_no_deferral_baseline": overall_stats,
        "defer_rate": float(len(deferred) / len(frame)),
        "random_router_baseline": baseline,
        "balance_distance_z": balance_z,
        "accuracy_z": accuracy_z,
        "z_threshold": -Z_THRESHOLD,
        "deferred_minus_auto_good_share": (
            deferred_stats["good_share"] - auto_stats["good_share"]
        ),
        "deferred_minus_auto_accuracy": (
            deferred_stats["accuracy"] - auto_stats["accuracy"]
        ),
        "harder_mix_than_random": bool(harder_mix),
        "harder_for_model_than_random": bool(harder_for_model),
    }

    notes = [
        "y_true == 1 means GOOD (approve-worthy, did not default).",
        "Accuracy is measured against the no-deferral baseline: every case forced "
        "to approve/reject at the engine's own approval_threshold (CoDoC pattern).",
        "balance_distance = |good_share - 0.5|; lower means a harder, more "
        "boundary-adjacent pile.",
        f"Random-router null: {trials} uniform samples of the same deferral volume.",
        "Caveat: balance_distance is a folded statistic, so its z-score is only "
        "interpretable when the population good-share is away from 50%. Here it is "
        "~92% good, so the folding is immaterial.",
    ]

    if harder_mix and harder_for_model:
        status = PASS
        reason = (
            f"deferred pile is {abs(balance_z):.1f} sd closer to 50/50 and the model is "
            f"{abs(accuracy_z):.1f} sd less accurate on it than a random router would give - "
            "the router is isolating genuinely hard cases"
        )
    else:
        status = FAIL
        parts = []
        if balance_z is not None and balance_z > 0:
            parts.append(
                f"deferred pile is {balance_z:+.1f} sd MORE lopsided than random "
                f"({deferred_stats['good_share']:.2%} good vs "
                f"{auto_stats['good_share']:.2%} good in the auto-decided pile) - "
                "the router defers EASIER cases"
            )
        elif not harder_mix:
            parts.append(
                f"deferred-pile balance is only {balance_z if balance_z is not None else float('nan'):+.1f} sd "
                "from random - not distinguishable from random selection"
            )
        if accuracy_z is not None and accuracy_z > 0:
            parts.append(
                f"the model is {accuracy_z:+.1f} sd MORE accurate on the deferred pile "
                "than random selection would give"
            )
        elif not harder_for_model:
            parts.append(
                "model accuracy on the deferred pile is not meaningfully below the random baseline"
            )
        reason = "; ".join(parts)

    return ConditionResult(1, name, status, reason, metrics, notes)


# --------------------------------------------------------------------------
# Condition 2
# --------------------------------------------------------------------------


def auc_implied_natural_rate_bound(
    auc: float, good_rate: float
) -> dict[str, Any]:
    """Approximate the Tasche-style natural override/defer rate bound from AUC.

    ASSUMPTIONS (stated explicitly because this is an approximation, not
    Tasche's exact construction):

    * Binormal, equal-variance score distributions. Score | good ~ N(delta, 1),
      score | bad ~ N(0, 1). Under that model AUC = Phi(delta / sqrt(2)), so
      delta = sqrt(2) * Phi^-1(AUC). This is the standard AUC -> separation
      inversion; it is exact only if the scores really are equal-variance
      normal, which is why the empirical error rate is reported alongside as a
      sanity check.
    * The "natural" rate of genuinely ambiguous cases is the Bayes error rate
      of that binormal problem at the observed class prior: the irreducible
      share of cases the model gets wrong however it is thresholded. A referral
      mechanism exists to catch those cases.
    * A referral rule cannot perfectly target errors, so it is allowed to spend
      more than one referral per error. We take an upper bound of
      2 x Bayes error - a generous 50%-precision allowance. Anything above that
      is referring cases the model already handles correctly, which is exactly
      the signal Tasche's bounds are meant to raise.
    * The lower bound is the Bayes error rate itself: referring fewer cases than
      the model gets wrong means the reviewer capacity cannot cover the errors.

    Returns the delta, the Bayes error, and the [lower, upper] bound.
    """
    auc = float(np.clip(auc, 1e-6, 1 - 1e-6))
    good_rate = float(np.clip(good_rate, 1e-6, 1 - 1e-6))
    bad_rate = 1.0 - good_rate

    delta = float(np.sqrt(2.0) * norm.ppf(auc))

    if delta <= 0:
        # No discriminatory power: the best possible rule is to call everything
        # the majority class, and the error rate is the minority share.
        bayes_error = float(min(good_rate, bad_rate))
    else:
        # Likelihood-ratio threshold where prior-weighted densities cross.
        t = delta / 2.0 + np.log(bad_rate / good_rate) / delta
        bayes_error = float(
            bad_rate * (1.0 - norm.cdf(t)) + good_rate * norm.cdf(t - delta)
        )

    return {
        "auc": auc,
        "good_rate": good_rate,
        "binormal_separation_delta": delta,
        "bayes_error_rate": bayes_error,
        "lower_bound": bayes_error,
        "upper_bound": float(min(1.0, 2.0 * bayes_error)),
        "assumptions": [
            "binormal equal-variance scores; delta = sqrt(2) * Phi^-1(AUC)",
            "natural ambiguous-case rate == Bayes error rate at the observed prior",
            "upper bound = 2 x Bayes error (allows 50% referral precision)",
            "lower bound = Bayes error (referrals must at least cover the errors)",
        ],
    }


def evaluate_condition_2(frame: pd.DataFrame) -> ConditionResult:
    """Observed defer rate sits within the AUC-implied natural-rate bound."""
    name = "Observed defer/override rate within the AUC-implied natural-rate bound"
    deferred_mask = frame["final_decision"].astype(str).str.upper() == DEFER
    observed_rate = float(deferred_mask.mean())

    y = frame["y_true"].to_numpy(dtype=int)
    prob = frame["best_model_prob"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        return ConditionResult(
            2,
            name,
            UNKNOWN,
            "only one outcome class present - AUC is undefined, so no bound can be computed",
            {"observed_defer_rate": observed_rate},
        )

    auc = float(roc_auc_score(y, prob))
    good_rate = float(y.mean())
    bound = auc_implied_natural_rate_bound(auc, good_rate)

    # Empirical cross-check on the modelling assumption: how often the model is
    # actually wrong under the no-deferral baseline.
    empirical_error = float((_predicted_good(frame) != y).mean())

    within = bound["lower_bound"] <= observed_rate <= bound["upper_bound"]
    status = PASS if within else FAIL
    if within:
        reason = (
            f"observed defer rate {observed_rate:.2%} lies inside the AUC-implied band "
            f"[{bound['lower_bound']:.2%}, {bound['upper_bound']:.2%}] at AUC {auc:.3f}"
        )
    elif observed_rate > bound["upper_bound"]:
        reason = (
            f"observed defer rate {observed_rate:.2%} is {observed_rate / max(bound['upper_bound'], 1e-9):.1f}x "
            f"the upper bound {bound['upper_bound']:.2%} implied by AUC {auc:.3f} - the router is "
            "referring far more cases than the model actually gets wrong, so most referrals "
            "are cases the model already handles"
        )
    else:
        reason = (
            f"observed defer rate {observed_rate:.2%} is below the lower bound "
            f"{bound['lower_bound']:.2%} - reviewer capacity cannot cover the model's error mass"
        )

    metrics = {
        "observed_defer_rate": observed_rate,
        "n_deferred": int(deferred_mask.sum()),
        "n_total": int(len(frame)),
        "model_auc": auc,
        "empirical_error_rate_no_deferral": empirical_error,
        "bound": bound,
    }
    notes = [
        "Approximation of Tasche, 'Bounds for rating override rates' (arXiv:1203.2287); "
        "not his exact construction.",
        "Assumptions are listed in metrics.bound.assumptions and in the docstring of "
        "auc_implied_natural_rate_bound().",
        "empirical_error_rate_no_deferral is reported so the binormal assumption can be "
        "sanity-checked against the data.",
    ]
    return ConditionResult(2, name, status, reason, metrics, notes)


# --------------------------------------------------------------------------
# Condition 3
# --------------------------------------------------------------------------


def evaluate_condition_3(exploration_labels: int = 0) -> ConditionResult:
    """Exploration arm has accumulated enough un-selected labels.

    The exploration arm (spec section 3, ``exploration_flag``) routes a random 2-5% of
    would-be-auto-decided applications to human review anyway. Those are the
    only labels NOT chosen by the router, hence the only escape hatch from the
    selective-labels trap (Lakkaraju et al., KDD 2017).

    The ``deferred_review`` capture table is being built separately and is
    deliberately not read or created here.
    """
    name = "Exploration arm has enough un-selected labels"
    if exploration_labels <= 0:
        return ConditionResult(
            3,
            name,
            FAIL,
            "no exploration arm data collected yet",
            {
                "exploration_labels": 0,
                "required_labels": MIN_EXPLORATION_LABELS,
            },
            notes=[
                "What is needed: the deferred_review capture table live in production "
                "with exploration_flag set on a random 2-5% of would-be-auto-decided "
                "applications;",
                f"at least {MIN_EXPLORATION_LABELS} of those exploration rows carrying a "
                "non-censored realized_outcome;",
                "spanning enough time for outcomes to season, and recorded against a "
                "single engine_version / threshold_artifact_hash so a fixed router's "
                "labels can be separated from a broken router's.",
            ],
        )

    enough = exploration_labels >= MIN_EXPLORATION_LABELS
    return ConditionResult(
        3,
        name,
        PASS if enough else FAIL,
        (
            f"{exploration_labels} un-selected exploration labels available "
            f"(need {MIN_EXPLORATION_LABELS})"
        ),
        {
            "exploration_labels": int(exploration_labels),
            "required_labels": MIN_EXPLORATION_LABELS,
        },
    )


# --------------------------------------------------------------------------
# Condition 4
# --------------------------------------------------------------------------


def evaluate_condition_4(repo_root: Path = REPO_ROOT) -> ConditionResult:
    """Retraining design models selection/missingness and reviewer bias.

    This is a design-artifact condition, not a statistic: it asks whether a
    written retraining design exists that models the selection mechanism
    (RMT-Net) and reviewer bias/consistency (Madras et al.) instead of treating
    ``human_decision`` as ground truth. It is checked by looking for such a
    document; it is never inferred from the data, because no data can prove a
    design exists.
    """
    name = "Retraining design models selection mechanism and reviewer bias"
    found = [
        str(candidate)
        for candidate in RETRAINING_DESIGN_CANDIDATES
        if (repo_root / candidate).is_file()
    ]
    metrics = {
        "searched_paths": [str(p) for p in RETRAINING_DESIGN_CANDIDATES],
        "found_paths": found,
        "required_topics": list(RETRAINING_DESIGN_REQUIRED_TOPICS),
    }
    if not found:
        return ConditionResult(
            4,
            name,
            FAIL,
            "no retraining design document exists - nothing models the selection "
            "mechanism or reviewer bias, and the spec forbids writing one that "
            "treats human decisions as ground truth",
            metrics,
            notes=[
                "Not computable from prediction_outputs.csv: this is a design artifact, "
                "not a statistic. Reported FAIL rather than UNKNOWN because absence of "
                "the document is itself conclusive.",
                "To satisfy it, a written design must cover: "
                + "; ".join(RETRAINING_DESIGN_REQUIRED_TOPICS)
                + ".",
                "Per spec section 3 this design should not be built until conditions 1-3 hold.",
            ],
        )

    return ConditionResult(
        4,
        name,
        UNKNOWN,
        f"retraining design document(s) found at {found} - automated review cannot "
        "confirm they adequately model the selection mechanism and reviewer bias; "
        "human sign-off required",
        metrics,
        notes=[
            "Presence of a document is necessary but not sufficient; this evaluator "
            "deliberately does not grade prose as PASS."
        ],
    )


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def load_predictions(path: Path = DEFAULT_PREDICTIONS) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"y_true", "best_model_prob", "final_decision"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def evaluate_gate(
    frame: pd.DataFrame,
    *,
    exploration_labels: int = 0,
    trials: int = RANDOM_TRIALS,
    seed: int = RANDOM_SEED,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run all four conditions and produce the overall verdict."""
    conditions = [
        evaluate_condition_1(frame, trials=trials, seed=seed),
        evaluate_condition_2(frame),
        evaluate_condition_3(exploration_labels=exploration_labels),
        evaluate_condition_4(repo_root=repo_root),
    ]
    all_pass = all(c.passed for c in conditions)
    failing = [c.number for c in conditions if not c.passed]
    return {
        "spec": "docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md section 3",
        "label_semantics": {
            "y_true==1": "GOOD customer (approve-worthy, did not default)",
            "best_model_prob": "approval probability, P(good)",
        },
        "n_rows": int(len(frame)),
        "conditions": [c.to_dict() for c in conditions],
        "failing_conditions": failing,
        "all_conditions_pass": bool(all_pass),
        "verdict": "SAFE TO OPEN THE LOOP" if all_pass else "DO NOT OPEN THE LOOP",
    }


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def render_report(result: dict[str, Any]) -> str:
    """Human-readable report, one block per condition."""
    lines: list[str] = []
    w = 78
    lines.append("=" * w)
    lines.append("RELEARNING LOOP - GATE EVALUATION")
    lines.append(f"Spec: {result['spec']}")
    lines.append(f"Rows evaluated: {result['n_rows']:,}")
    lines.append("Labels: y_true==1 means GOOD (did not default); probabilities are P(approve).")
    lines.append("=" * w)

    for cond in result["conditions"]:
        m = cond["metrics"]
        lines.append("")
        lines.append(f"[{cond['status']}] CONDITION {cond['condition']} - {cond['name']}")
        lines.append("-" * w)
        lines.append(f"  Reason: {cond['reason']}")

        if cond["condition"] == 1 and m.get("deferred"):
            auto, deferred, base = m["auto_decided"], m["deferred"], m["overall_no_deferral_baseline"]
            lines.append("")
            lines.append(f"  {'subset':<18}{'n':>8}{'good share':>14}{'|share-0.5|':>14}{'accuracy':>12}{'AUC':>10}")
            for label, s in (
                ("auto-decided", auto),
                ("deferred", deferred),
                ("all (no-defer)", base),
            ):
                lines.append(
                    f"  {label:<18}{s['n']:>8,}"
                    f"{_fmt(s['good_share'], '.4%'):>14}"
                    f"{_fmt(s['balance_distance'], '.4f'):>14}"
                    f"{_fmt(s['accuracy'], '.4f'):>12}"
                    f"{_fmt(s['auc'], '.4f'):>10}"
                )
            rb = m["random_router_baseline"]
            lines.append("")
            lines.append(
                f"  Random-router null ({rb['trials']} draws of {deferred['n']:,} cases, seed {rb['seed']}):"
            )
            for key, label in (
                ("good_share", "good share"),
                ("balance_distance", "|share-0.5|"),
                ("accuracy", "accuracy"),
                ("auc", "AUC"),
            ):
                d = rb[key]
                lines.append(
                    f"    {label:<13} mean={_fmt(d['mean'])}  sd={_fmt(d['std'], '.5f')}  "
                    f"[p05 {_fmt(d['p05'])}, p95 {_fmt(d['p95'])}]"
                )
            lines.append("")
            lines.append(
                f"    balance_distance z = {_fmt(m['balance_distance_z'], '+.2f')} "
                f"(need <= {m['z_threshold']:+.2f}; negative = harder than random)"
            )
            lines.append(
                f"    accuracy         z = {_fmt(m['accuracy_z'], '+.2f')} "
                f"(need <= {m['z_threshold']:+.2f}; negative = model struggles there)"
            )
            lines.append(
                f"    deferred - auto good share = {_fmt(m['deferred_minus_auto_good_share'], '+.4%')}"
                "   (positive = the router defers EASIER cases)"
            )
            lines.append(
                f"    deferred - auto accuracy   = {_fmt(m['deferred_minus_auto_accuracy'], '+.4f')}"
            )

        if cond["condition"] == 2 and m.get("bound"):
            b = m["bound"]
            lines.append("")
            lines.append(f"    observed defer rate      = {_fmt(m['observed_defer_rate'], '.4%')} ({m['n_deferred']:,}/{m['n_total']:,})")
            lines.append(f"    model AUC                = {_fmt(m['model_auc'], '.4f')}")
            lines.append(f"    binormal separation delta= {_fmt(b['binormal_separation_delta'], '.4f')}")
            lines.append(f"    Bayes error rate         = {_fmt(b['bayes_error_rate'], '.4%')}")
            lines.append(f"    empirical error (no-def) = {_fmt(m['empirical_error_rate_no_deferral'], '.4%')}")
            lines.append(
                f"    allowed band             = [{_fmt(b['lower_bound'], '.4%')}, {_fmt(b['upper_bound'], '.4%')}]"
            )
            lines.append("    assumptions:")
            for a in b["assumptions"]:
                lines.append(f"      - {a}")

        if cond["condition"] == 3:
            lines.append("")
            lines.append(
                f"    exploration labels available = {m['exploration_labels']} "
                f"(required {m['required_labels']})"
            )

        if cond["condition"] == 4:
            lines.append("")
            lines.append(f"    searched: {', '.join(m['searched_paths'])}")
            lines.append(f"    found:    {', '.join(m['found_paths']) or '(none)'}")

        if cond["notes"]:
            lines.append("")
            for note in cond["notes"]:
                lines.append(f"    note: {note}")

    lines.append("")
    lines.append("=" * w)
    lines.append(f"OVERALL VERDICT: {result['verdict']}")
    if not result["all_conditions_pass"]:
        failing = ", ".join(str(n) for n in result["failing_conditions"])
        lines.append(f"Failing conditions: {failing}")
        lines.append(
            "No retraining on deferred-case labels is permitted. Retraining on cases a "
            "broken router selected is the runaway feedback loop (Ensign et al.)."
        )
    lines.append("=" * w)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--trials", type=int, default=RANDOM_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--exploration-labels",
        type=int,
        default=0,
        help="count of un-selected exploration-arm labels available (default 0: none captured yet)",
    )
    args = parser.parse_args(argv)

    frame = load_predictions(args.predictions)
    result = evaluate_gate(
        frame,
        exploration_labels=args.exploration_labels,
        trials=args.trials,
        seed=args.seed,
    )
    print(render_report(result))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nMachine-readable report written to {args.out}")

    # Non-zero exit while the loop must stay shut, so CI can depend on it.
    return 0 if result["all_conditions_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
