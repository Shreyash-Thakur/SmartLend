"""Missingness profiler.

The premise: for this project missingness is not noise to be removed, it is the
object of study. Imputing it destroys the fairness signal (thin-file applicants
have systematically more missing fields) and very likely destroys a proxy for the
low-overlap region where positivity fails.

So "cleaning" here means *classifying* missingness, not eliminating it. Four
mechanical detectors, each testable:

  sentinel     a single value holds anomalous mass at a distribution extreme
  structural   missingness is perfectly predicted by another column
  block        columns share a near-identical missingness pattern
  informative  the missingness indicator ALONE predicts the target

The last one is the decisive test: if `is_missing` predicts default on its own,
the missingness carries signal and must never be imputed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Missingness(str, Enum):
    COMPLETE = "complete"
    SENTINEL_CODED = "sentinel_coded"
    STRUCTURAL = "structural"
    INFORMATIVE = "informative"
    MAR_TRACE = "mar_trace"
    MAR_CANDIDATE = "mar_candidate"


RECOMMENDATION: dict[Missingness, str] = {
    Missingness.COMPLETE: "No action.",
    Missingness.SENTINEL_CODED: (
        "Replace sentinel with NaN and add an indicator. Never leave the literal "
        "value in place; never impute over it."
    ),
    Missingness.STRUCTURAL: (
        "Missingness IS the information. Add an indicator and do not impute — the "
        "predictor column already encodes the value."
    ),
    Missingness.INFORMATIVE: (
        "Missingness predicts the target. Add an indicator, let tree models handle "
        "NaN natively, and include this column in the fairness analysis."
    ),
    Missingness.MAR_TRACE: (
        "Negligible missingness. Safe to impute (median/mode); log the decision."
    ),
    Missingness.MAR_CANDIDATE: (
        "No signal detected in the missingness. Add an indicator anyway (cheap), "
        "then impute or use native NaN handling."
    ),
}


@dataclass(frozen=True)
class ColumnProfile:
    column: str
    dtype: str
    n_rows: int
    missing_count: int
    missing_rate: float
    n_unique: int
    sentinel_values: tuple[float, ...]
    sentinel_rate: float
    structural_predictor: str | None
    block_id: int | None
    target_auc: float | None
    rate_when_missing: float | None
    rate_when_present: float | None
    classification: Missingness

    @property
    def recommendation(self) -> str:
        return RECOMMENDATION[self.classification]

    @property
    def effective_missing_rate(self) -> float:
        """Missingness after sentinels are converted to NaN."""
        return self.missing_rate + self.sentinel_rate


def detect_sentinels(
    series: pd.Series,
    *,
    min_count: int = 2,
    gap_ratio: float = 1.0,
    min_unique: int = 10,
    max_cluster: int = 3,
) -> tuple[float, ...]:
    """Find coded values masquerading as numbers.

    Detection is based on *gap structure*, not on IQR. An earlier IQR-based
    version silently failed on real data: credit delinquency counts are
    zero-inflated, so q1 == median == q3 == 0, the IQR is zero, and no
    outlyingness test is possible. That is precisely where sentinels live
    (Give-Me-Some-Credit codes 96/98 in its late-payment columns).

    A sentinel cluster is therefore identified as: the smaller side of the
    largest gap in the sorted unique values, where the gap exceeds the bulk's
    entire span and every member is a *repeated* value. Repetition is what
    separates a code from an outlier — a genuine tail value appears once.

    Contiguous codes (96 AND 98) are caught together, which peeling extremes
    one at a time cannot do, since the gap between 96 and 98 is small.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return ()

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or clean.nunique() < min_unique:
        return ()

    counts = clean.value_counts()
    uniques = np.sort(counts.index.to_numpy(dtype="float64"))
    diffs = np.diff(uniques)
    if diffs.size == 0:
        return ()

    split = int(np.argmax(diffs))
    gap = float(diffs[split])
    left, right = uniques[: split + 1], uniques[split + 1 :]

    # The sentinel cluster is the smaller side of the largest gap.
    cluster, bulk = (right, left) if len(right) <= len(left) else (left, right)
    if len(cluster) > max_cluster or len(bulk) == 0:
        return ()

    span = float(bulk.max() - bulk.min())
    if gap <= gap_ratio * span:
        return ()

    # Every member must be repeated; a one-off extreme is an outlier, not a code.
    if any(int(counts.get(value, 0)) < min_count for value in cluster):
        return ()

    return tuple(float(v) for v in cluster)


def detect_structural(
    df: pd.DataFrame,
    column: str,
    *,
    max_predictor_cardinality: int = 50,
    candidates: list[str] | None = None,
) -> str | None:
    """Return a column whose value perfectly determines `column`'s missingness.

    The motivating case is Home Credit's OWN_CAR_AGE, which is missing exactly
    when FLAG_OWN_CAR == 'N'. There the missingness is not a gap — it is a fact
    about the applicant, already encoded elsewhere.

    Only low-cardinality predictors are considered: a high-cardinality column can
    partition the rows finely enough to "predict" anything, which would be
    overfitting rather than structure.
    """
    is_missing = df[column].isna()
    # Nothing to explain if missingness does not vary.
    if is_missing.nunique() < 2:
        return None

    pool = candidates if candidates is not None else list(df.columns)
    for predictor in pool:
        if predictor == column:
            continue
        values = df[predictor]
        if values.isna().any():
            # A predictor with its own gaps cannot cleanly determine anything.
            continue
        if values.nunique(dropna=False) > max_predictor_cardinality:
            continue
        # Perfect prediction <=> missingness is constant within every group.
        if is_missing.groupby(values, observed=True).nunique().max() == 1:
            return predictor
    return None


def detect_blocks(
    df: pd.DataFrame,
    columns: list[str],
    *,
    threshold: float = 0.99,
) -> dict[str, int]:
    """Group columns that go missing together.

    Home Credit has ~47 building-characteristic columns (`*_AVG`, `*_MODE`,
    `*_MEDI`) that are missing as a unit. Emitting 47 near-identical indicators
    adds collinearity and no information, so they collapse to one block.

    Returns column -> block id, only for columns in a block of size >= 2.
    """
    missing = df[columns].isna()
    varying = [c for c in columns if missing[c].nunique() > 1]
    if len(varying) < 2:
        return {}

    correlation = missing[varying].astype(float).corr()

    # Connected components over the "near-identical pattern" graph.
    parent = {c: c for c in varying}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(varying):
        for b in varying[i + 1 :]:
            value = correlation.loc[a, b]
            if pd.notna(value) and value >= threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for column in varying:
        groups.setdefault(find(column), []).append(column)

    blocks: dict[str, int] = {}
    for block_id, members in enumerate(
        sorted((m for m in groups.values() if len(m) >= 2), key=lambda m: -len(m))
    ):
        for column in members:
            blocks[column] = block_id
    return blocks


def score_informativeness(
    is_missing: pd.Series, target: pd.Series
) -> tuple[float | None, float | None, float | None]:
    """AUC of the missingness indicator alone, plus per-group positive rates.

    This is the test that decides whether missingness may be imputed away. If
    `is_missing` discriminates the target on its own, it carries signal.
    """
    mask = target.notna()
    y = target[mask]
    x = is_missing[mask].astype(int)
    if y.nunique() < 2 or x.nunique() < 2:
        return None, None, None

    rate_missing = float(y[x == 1].mean())
    rate_present = float(y[x == 0].mean())

    # Closed-form AUC for a binary score; avoids a sklearn dependency here.
    p = float((x[y == 1] == 1).mean())  # P(missing | positive)
    q = float((x[y == 0] == 1).mean())  # P(missing | negative)
    auc = p * (1.0 - q) + 0.5 * (p * q + (1.0 - p) * (1.0 - q))
    return auc, rate_missing, rate_present


def profile_frame(
    df: pd.DataFrame,
    *,
    target: str | pd.Series | None = None,
    sample_size: int | None = 50_000,
    random_state: int = 42,
    mar_trace_rate: float = 0.001,
    informative_auc_delta: float = 0.02,
    structural_candidates: list[str] | None = None,
) -> list[ColumnProfile]:
    """Profile every column's missingness and classify its mechanism.

    Missing counts are computed on the FULL frame; the expensive structural and
    block detectors run on a sample, since both answer structural questions that
    do not need 300k rows.
    """
    target_series: pd.Series | None
    if isinstance(target, str):
        target_series = df[target]
    else:
        target_series = target

    feature_columns = [c for c in df.columns if not (isinstance(target, str) and c == target)]

    work = df
    if sample_size is not None and len(df) > sample_size:
        work = df.sample(sample_size, random_state=random_state)

    candidate_pool = (
        structural_candidates
        if structural_candidates is not None
        else [c for c in feature_columns if work[c].nunique(dropna=False) <= 50]
    )

    missing_columns = [c for c in feature_columns if df[c].isna().any()]
    blocks = detect_blocks(work, [c for c in missing_columns if c in work.columns])

    profiles: list[ColumnProfile] = []
    n_rows = len(df)

    for column in feature_columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        missing_rate = missing_count / n_rows if n_rows else 0.0

        sentinels = detect_sentinels(series)
        sentinel_rate = (
            float(series.isin(sentinels).sum()) / n_rows if sentinels and n_rows else 0.0
        )

        structural = (
            detect_structural(work, column, candidates=candidate_pool)
            if missing_count and column in work.columns
            else None
        )

        auc = rate_missing = rate_present = None
        if target_series is not None and missing_count:
            auc, rate_missing, rate_present = score_informativeness(
                series.isna(), target_series
            )

        profiles.append(
            ColumnProfile(
                column=column,
                dtype=str(series.dtype),
                n_rows=n_rows,
                missing_count=missing_count,
                missing_rate=missing_rate,
                n_unique=int(series.nunique(dropna=True)),
                sentinel_values=sentinels,
                sentinel_rate=sentinel_rate,
                structural_predictor=structural,
                block_id=blocks.get(column),
                target_auc=auc,
                rate_when_missing=rate_missing,
                rate_when_present=rate_present,
                classification=_classify(
                    missing_rate=missing_rate,
                    sentinels=sentinels,
                    structural=structural,
                    auc=auc,
                    mar_trace_rate=mar_trace_rate,
                    informative_auc_delta=informative_auc_delta,
                ),
            )
        )
    return profiles


def _classify(
    *,
    missing_rate: float,
    sentinels: tuple[float, ...],
    structural: str | None,
    auc: float | None,
    mar_trace_rate: float,
    informative_auc_delta: float,
) -> Missingness:
    """Priority order matters: sentinels must be fixed before anything else is
    even measurable, and structure beats statistics."""
    if sentinels:
        return Missingness.SENTINEL_CODED
    if missing_rate == 0.0:
        return Missingness.COMPLETE
    if structural is not None:
        return Missingness.STRUCTURAL
    if auc is not None and abs(auc - 0.5) >= informative_auc_delta:
        return Missingness.INFORMATIVE
    if missing_rate < mar_trace_rate:
        return Missingness.MAR_TRACE
    return Missingness.MAR_CANDIDATE


def to_frame(profiles: list[ColumnProfile]) -> pd.DataFrame:
    """Report table, ordered by how much attention each column needs."""
    rows = [
        {
            "column": p.column,
            "dtype": p.dtype,
            "missing_rate": round(p.missing_rate, 4),
            "sentinel_rate": round(p.sentinel_rate, 4),
            "effective_missing_rate": round(p.effective_missing_rate, 4),
            "sentinels": ", ".join(str(v) for v in p.sentinel_values),
            "structural_predictor": p.structural_predictor or "",
            "block_id": "" if p.block_id is None else p.block_id,
            "target_auc": None if p.target_auc is None else round(p.target_auc, 4),
            "rate_when_missing": (
                None if p.rate_when_missing is None else round(p.rate_when_missing, 4)
            ),
            "rate_when_present": (
                None if p.rate_when_present is None else round(p.rate_when_present, 4)
            ),
            "classification": p.classification.value,
            "recommendation": p.recommendation,
        }
        for p in profiles
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    severity = {
        Missingness.SENTINEL_CODED.value: 0,
        Missingness.INFORMATIVE.value: 1,
        Missingness.STRUCTURAL.value: 2,
        Missingness.MAR_CANDIDATE.value: 3,
        Missingness.MAR_TRACE.value: 4,
        Missingness.COMPLETE.value: 5,
    }
    frame["_severity"] = frame["classification"].map(severity)
    frame = frame.sort_values(
        ["_severity", "effective_missing_rate"], ascending=[True, False]
    ).drop(columns="_severity")
    return frame.reset_index(drop=True)


def apply_sentinels(df: pd.DataFrame, profiles: list[ColumnProfile]) -> pd.DataFrame:
    """Convert detected sentinels to NaN. Returns a copy.

    This is the only transformation the profiler performs, because leaving a
    sentinel in place corrupts every downstream statistic. Indicator creation is
    deliberately left to the feature pipeline, where it can be versioned.
    """
    out = df.copy()
    for profile in profiles:
        if profile.sentinel_values and profile.column in out.columns:
            out[profile.column] = out[profile.column].replace(
                list(profile.sentinel_values), np.nan
            )
    return out


def add_missingness_indicators(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    *,
    include: tuple[Missingness, ...] = (
        Missingness.SENTINEL_CODED,
        Missingness.STRUCTURAL,
        Missingness.INFORMATIVE,
        Missingness.MAR_CANDIDATE,
    ),
    collapse_blocks: bool = True,
) -> pd.DataFrame:
    """Add `<column>_isna` indicators, collapsing co-missing blocks to one.

    Block collapsing matters on Home Credit: 47 building columns share one
    missingness pattern, so 47 indicators would be perfectly collinear.
    """
    out = df.copy()
    selected = [p for p in profiles if p.classification in include]

    emitted_blocks: set[int] = set()
    for profile in selected:
        if profile.column not in out.columns:
            continue
        if collapse_blocks and profile.block_id is not None:
            if profile.block_id in emitted_blocks:
                continue
            emitted_blocks.add(profile.block_id)
            out[f"block{profile.block_id}_isna"] = out[profile.column].isna().astype("int8")
            continue
        out[f"{profile.column}_isna"] = out[profile.column].isna().astype("int8")
    return out
