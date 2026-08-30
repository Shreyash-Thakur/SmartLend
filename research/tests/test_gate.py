"""Tests for the relearning-loop gate evaluator.

The central risk this file guards against is a LABEL INVERSION bug: if
``y_true == 1`` were read as "defaulted" instead of "good", condition 1 would
happily bless a router that hands humans the easiest cases. So the core tests
build two synthetic routers whose correct verdict is known by construction --
one that defers genuinely hard (boundary) cases, one that defers easy ones --
and assert the gate separates them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from research.relearning.gate import (
    FAIL,
    PASS,
    UNKNOWN,
    auc_implied_natural_rate_bound,
    evaluate_condition_1,
    evaluate_condition_2,
    evaluate_condition_3,
    evaluate_condition_4,
    evaluate_gate,
    main,
    render_report,
    subset_metrics,
)

THRESHOLD = 0.5


def _frame(probs, y_true, decisions) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "best_model_prob": np.asarray(probs, dtype=float),
            "final_decision": list(decisions),
            "approval_threshold": THRESHOLD,
        }
    )


def _synthetic_population(n: int = 4000, seed: int = 7):
    """Population where 'hard' is defined by construction.

    * 40% of rows sit at the decision boundary (prob ~ 0.5, outcome a coin flip
      -- the model genuinely cannot tell). These are the HARD cases.
    * 60% sit at the extremes where the model is always right, and are mostly
      good, so the population as a whole is lopsided (~74% good) the way a real
      credit book is. That lopsidedness is what makes the balanced deferred pile
      detectable against a random router.

    y_true == 1 means GOOD.
    """
    rng = np.random.default_rng(seed)
    n_hard = int(round(0.4 * n))
    n_easy = n - n_hard
    n_easy_good = int(round(0.9 * n_easy))

    hard_prob = rng.uniform(0.45, 0.55, n_hard)
    hard_y = rng.integers(0, 2, n_hard)  # ~50/50: genuinely ambiguous

    easy_prob = np.concatenate(
        [rng.uniform(0.9, 0.99, n_easy_good), rng.uniform(0.01, 0.1, n_easy - n_easy_good)]
    )
    easy_y = (easy_prob >= THRESHOLD).astype(int)  # model always right on these

    probs = np.concatenate([hard_prob, easy_prob])
    y = np.concatenate([hard_y, easy_y])
    is_hard = np.concatenate([np.ones(n_hard, dtype=bool), np.zeros(n_easy, dtype=bool)])
    return probs, y, is_hard


def _decisions(probs: np.ndarray, defer_mask: np.ndarray) -> list[str]:
    out = []
    for prob, defer in zip(probs, defer_mask):
        if defer:
            out.append("DEFER")
        else:
            out.append("APPROVE" if prob >= THRESHOLD else "REJECT")
    return out


@pytest.fixture()
def good_router() -> pd.DataFrame:
    """Defers exactly the boundary cases. Must PASS condition 1."""
    probs, y, is_hard = _synthetic_population()
    return _frame(probs, y, _decisions(probs, is_hard))


@pytest.fixture()
def bad_router() -> pd.DataFrame:
    """Defers exactly the easy cases -- the failure mode the real router has."""
    probs, y, is_hard = _synthetic_population()
    return _frame(probs, y, _decisions(probs, ~is_hard))


# ---------------------------------------------------------------------------
# condition 1 -- the label-inversion guard
# ---------------------------------------------------------------------------


def test_good_router_passes_condition_1(good_router):
    result = evaluate_condition_1(good_router, trials=50, seed=1)
    assert result.status == PASS, result.reason
    m = result.metrics
    # The deferred pile is the balanced/ambiguous one.
    assert m["deferred"]["balance_distance"] < m["auto_decided"]["balance_distance"]
    # And the model does worse there than on the auto-decided pile.
    assert m["deferred"]["accuracy"] < m["auto_decided"]["accuracy"]
    assert m["balance_distance_z"] < -2.0
    assert m["accuracy_z"] < -2.0


def test_bad_router_fails_condition_1(bad_router):
    result = evaluate_condition_1(bad_router, trials=50, seed=1)
    assert result.status == FAIL
    m = result.metrics
    # Deferred pile is the LOPSIDED, easy one -- both z-scores point the wrong way.
    assert m["balance_distance_z"] > 2.0
    assert m["accuracy_z"] > 2.0
    assert m["deferred"]["accuracy"] > m["auto_decided"]["accuracy"]
    assert "EASIER" in result.reason


def test_condition_1_is_not_symmetric_under_label_flip(good_router):
    """Flipping y_true must turn the good router's verdict into nonsense.

    If the evaluator were label-agnostic (the inversion bug), this would still
    pass and the test would fail -- which is the point.
    """
    flipped = good_router.copy()
    flipped["y_true"] = 1 - flipped["y_true"]
    good = evaluate_condition_1(good_router, trials=50, seed=1)
    bad = evaluate_condition_1(flipped, trials=50, seed=1)
    assert good.status == PASS
    # With labels flipped the model is wrong on every easy case, so the
    # auto-decided pile becomes the one it fails on: no longer a valid router.
    assert bad.metrics["accuracy_z"] > good.metrics["accuracy_z"]


def test_random_router_fails_condition_1():
    """A router that defers uniformly at random must not pass."""
    probs, y, _ = _synthetic_population(seed=11)
    rng = np.random.default_rng(3)
    defer_mask = rng.random(len(probs)) < 0.4
    frame = _frame(probs, y, _decisions(probs, defer_mask))
    result = evaluate_condition_1(frame, trials=100, seed=5)
    assert result.status == FAIL
    assert abs(result.metrics["balance_distance_z"]) < 3.0


def test_condition_1_handles_no_deferrals():
    probs, y, _ = _synthetic_population(n=200, seed=2)
    frame = _frame(probs, y, _decisions(probs, np.zeros(len(probs), dtype=bool)))
    result = evaluate_condition_1(frame, trials=10, seed=1)
    assert result.status == FAIL
    assert "no deferred cases" in result.reason


def test_condition_1_handles_all_deferrals():
    probs, y, _ = _synthetic_population(n=200, seed=2)
    frame = _frame(probs, y, ["DEFER"] * len(probs))
    result = evaluate_condition_1(frame, trials=10, seed=1)
    assert result.status == FAIL
    assert "no auto-decided subset" in result.reason


def test_subset_metrics_label_semantics():
    """good_share counts y_true == 1, and accuracy uses the approval threshold."""
    frame = _frame([0.9, 0.8, 0.2, 0.1], [1, 0, 1, 0], ["APPROVE"] * 2 + ["REJECT"] * 2)
    m = subset_metrics(frame)
    assert m["n"] == 4
    assert m["good_share"] == 0.5
    assert m["balance_distance"] == 0.0
    # Correct on rows 0 and 3, wrong on 1 and 2.
    assert m["accuracy"] == 0.5


def test_subset_metrics_auc_none_when_single_class():
    frame = _frame([0.9, 0.8], [1, 1], ["APPROVE"] * 2)
    assert subset_metrics(frame)["auc"] is None


# ---------------------------------------------------------------------------
# condition 2
# ---------------------------------------------------------------------------


def test_bound_shrinks_as_auc_improves():
    weak = auc_implied_natural_rate_bound(0.60, 0.92)
    strong = auc_implied_natural_rate_bound(0.90, 0.92)
    assert strong["bayes_error_rate"] < weak["bayes_error_rate"]
    assert strong["upper_bound"] < weak["upper_bound"]
    assert weak["upper_bound"] == pytest.approx(2 * weak["bayes_error_rate"])


def test_bound_for_realistic_auc_is_far_below_a_50_percent_defer_rate():
    bound = auc_implied_natural_rate_bound(0.767, 0.92)
    assert bound["upper_bound"] < 0.25


def test_condition_2_fails_on_excessive_defer_rate():
    rng = np.random.default_rng(4)
    n = 2000
    y = (rng.random(n) < 0.92).astype(int)
    probs = np.clip(rng.normal(0.5, 0.15, n) + 0.25 * y, 0.01, 0.99)
    decisions = ["DEFER"] * (n // 2) + ["APPROVE"] * (n - n // 2)
    frame = _frame(probs, y, decisions)
    result = evaluate_condition_2(frame)
    assert result.status == FAIL
    assert result.metrics["observed_defer_rate"] == pytest.approx(0.5)
    assert result.metrics["observed_defer_rate"] > result.metrics["bound"]["upper_bound"]


def test_condition_2_passes_when_defer_rate_sits_in_band():
    rng = np.random.default_rng(9)
    n = 4000
    y = (rng.random(n) < 0.92).astype(int)
    probs = np.clip(rng.normal(0.5, 0.15, n) + 0.25 * y, 0.01, 0.99)
    bound = auc_implied_natural_rate_bound(roc_auc_score(y, probs), float(y.mean()))
    target = (bound["lower_bound"] + bound["upper_bound"]) / 2
    n_defer = int(round(target * n))
    decisions = ["DEFER"] * n_defer + ["APPROVE"] * (n - n_defer)
    result = evaluate_condition_2(_frame(probs, y, decisions))
    assert result.status == PASS


# ---------------------------------------------------------------------------
# conditions 3 and 4
# ---------------------------------------------------------------------------


def test_condition_3_fails_with_no_exploration_data():
    result = evaluate_condition_3(exploration_labels=0)
    assert result.status == FAIL
    assert result.reason == "no exploration arm data collected yet"
    assert result.metrics["required_labels"] > 0
    assert result.notes


def test_condition_3_fails_with_too_few_labels():
    assert evaluate_condition_3(exploration_labels=10).status == FAIL


def test_condition_3_passes_with_enough_labels():
    result = evaluate_condition_3(exploration_labels=100_000)
    assert result.status == PASS


def test_condition_4_fails_without_a_design_document(tmp_path):
    result = evaluate_condition_4(repo_root=tmp_path)
    assert result.status == FAIL
    assert "no retraining design document" in result.reason


def test_condition_4_is_unknown_not_pass_when_a_document_exists(tmp_path):
    doc = tmp_path / "docs" / "retraining-design.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("RMT-Net selection model, Madras reviewer bias.", encoding="utf-8")
    result = evaluate_condition_4(repo_root=tmp_path)
    # Presence of prose is never auto-graded as PASS.
    assert result.status == UNKNOWN
    assert not result.passed


# ---------------------------------------------------------------------------
# overall verdict + reporting
# ---------------------------------------------------------------------------


def test_verdict_is_refusal_when_any_condition_fails(good_router, tmp_path):
    result = evaluate_gate(good_router, exploration_labels=0, trials=20, repo_root=tmp_path)
    assert result["verdict"] == "DO NOT OPEN THE LOOP"
    assert result["all_conditions_pass"] is False
    assert 3 in result["failing_conditions"]
    assert len(result["conditions"]) == 4


def test_report_is_ascii_and_json_serialisable(good_router, tmp_path):
    result = evaluate_gate(good_router, trials=20, repo_root=tmp_path)
    text = render_report(result)
    assert text.isascii()
    assert "OVERALL VERDICT" in text
    for i in (1, 2, 3, 4):
        assert f"CONDITION {i}" in text
    json.dumps(result)  # must not raise


def test_main_writes_json_and_exits_nonzero_while_gate_is_shut(tmp_path, bad_router, capsys):
    csv = tmp_path / "predictions.csv"
    bad_router.to_csv(csv, index=False)
    out = tmp_path / "reports" / "relearning_gate.json"
    code = main(["--predictions", str(csv), "--out", str(out), "--trials", "20"])
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "DO NOT OPEN THE LOOP"
    assert [c["condition"] for c in payload["conditions"]] == [1, 2, 3, 4]
    assert payload["conditions"][0]["status"] == FAIL
    assert "DO NOT OPEN THE LOOP" in capsys.readouterr().out
