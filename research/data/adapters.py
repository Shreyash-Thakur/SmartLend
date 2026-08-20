"""Dataset -> canonical mapping engine.

Every dataset resolves into the same bundle shape:

  core      the ~12 canonical fields CBES consumes (portable, comparable)
  native    the dataset's own features, categoricals INTACT (what ML consumes)
  target    1 = default
  selected  1 = approved by historical policy, where observable

The `native` half is why this layer exists. The legacy pipeline used
`select_dtypes(include=["number"])`, which silently discarded every categorical
feature — gender, education, employment type, region. That both handicapped the
model and made CODE_GENDER fairness work impossible. Categoricals are preserved
here and typed, so downstream encoders can make an explicit choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.data.canonical import (
    CORE_NAMES,
    SELECTED,
    TARGET,
    Availability,
    DatasetSpec,
)


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    core: pd.DataFrame
    native: pd.DataFrame
    target: pd.Series
    selected: pd.Series | None
    spec: DatasetSpec

    @property
    def n_rows(self) -> int:
        return len(self.core)

    def categorical_columns(self) -> list[str]:
        return [
            c
            for c in self.native.columns
            if isinstance(self.native[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(self.native[c])
            or pd.api.types.is_string_dtype(self.native[c])
        ]

    def numeric_columns(self) -> list[str]:
        return [c for c in self.native.columns if pd.api.types.is_numeric_dtype(self.native[c])]

    def drop_unresolved_target(self) -> "DatasetBundle":
        """Drop rows whose outcome is not yet resolved.

        Lending Club needs this: Current / In Grace Period / Late loans have no
        final outcome, and coercing them to 0 is a common and serious error.
        """
        keep = self.target.notna()
        if bool(keep.all()):
            return self
        return DatasetBundle(
            name=self.name,
            core=self.core.loc[keep].reset_index(drop=True),
            native=self.native.loc[keep].reset_index(drop=True),
            target=self.target.loc[keep].reset_index(drop=True).astype("int8"),
            selected=(
                None
                if self.selected is None
                else self.selected.loc[keep].reset_index(drop=True)
            ),
            spec=self.spec,
        )


def _resolve(
    source: str | object, df: pd.DataFrame, label: str
) -> pd.Series:
    if callable(source):
        return source(df)
    if isinstance(source, str):
        if source not in df.columns:
            raise KeyError(f"{label} column {source!r} not present in frame")
        return df[source]
    raise TypeError(f"{label} must be a column name or callable, got {type(source)!r}")


def build_bundle(spec: DatasetSpec, df: pd.DataFrame) -> DatasetBundle:
    """Map a raw frame into canonical space.

    Nothing is imputed. A canonical field the dataset cannot supply becomes an
    all-NaN column, so downstream code sees an honest gap rather than a
    fabricated value.
    """
    mapping = spec.by_canonical()

    core = pd.DataFrame(index=df.index)
    for name in CORE_NAMES:
        field = mapping.get(name)
        resolved = field.resolve(df) if field is not None else None
        core[name] = pd.Series(pd.NA, index=df.index, dtype="object") if resolved is None else resolved

    target = _resolve(spec.target, df, "target").rename(TARGET)

    selected = None
    if spec.selected is not None:
        selected = _resolve(spec.selected, df, "selected").rename(SELECTED)

    # Native features: everything except the target and selection columns, with
    # dtypes left alone so categoricals survive.
    drop = {c for c in (spec.target, spec.selected) if isinstance(c, str)}
    native = df.drop(columns=[c for c in drop if c in df.columns]).copy()

    return DatasetBundle(
        name=spec.name,
        core=core,
        native=native,
        target=target,
        selected=selected,
        spec=spec,
    )


def coverage_report(spec: DatasetSpec) -> pd.DataFrame:
    """Which canonical fields this dataset supplies, and how.

    This table is what makes cross-dataset claims auditable: it states up front
    where a comparison rests on a proxy or a derivation rather than a real column.
    """
    mapping = spec.by_canonical()
    rows = []
    for name in CORE_NAMES:
        field = mapping.get(name)
        rows.append(
            {
                "canonical": name,
                "availability": (
                    field.availability.value if field else Availability.ABSENT.value
                ),
                "source": (
                    ""
                    if field is None or field.source is None
                    else (field.source if isinstance(field.source, str) else "<derived>")
                ),
                "notes": field.notes if field else "not mapped",
            }
        )
    return pd.DataFrame(rows)


def validate_spec(spec: DatasetSpec, df: pd.DataFrame) -> list[str]:
    """Check a spec against a real frame. Returns human-readable problems.

    Specs for datasets that are not downloaded yet are written from
    documentation, so they must be validated against the actual columns before
    any result computed through them is trusted.
    """
    problems: list[str] = []
    mapping = spec.by_canonical()

    for name in CORE_NAMES:
        field = mapping.get(name)
        if field is None:
            problems.append(f"{name}: not mapped in spec")
            continue
        if field.availability is Availability.ABSENT:
            continue
        if isinstance(field.source, str) and field.source not in df.columns:
            problems.append(f"{name}: source column {field.source!r} missing from frame")
        elif callable(field.source):
            try:
                field.source(df)
            except Exception as exc:  # noqa: BLE001 - report, don't crash validation
                problems.append(f"{name}: derivation failed ({type(exc).__name__}: {exc})")

    for label, source in (("target", spec.target), ("selected", spec.selected)):
        if source is None:
            continue
        try:
            _resolve(source, df, label)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{label}: {type(exc).__name__}: {exc}")

    required_gaps = spec.missing_required()
    if required_gaps:
        problems.append(
            "required canonical fields unavailable: " + ", ".join(required_gaps)
        )
    return problems
