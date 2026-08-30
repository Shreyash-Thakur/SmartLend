from __future__ import annotations

from datetime import datetime, timezone
import uuid

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database import Base


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"app-{uuid.uuid4().hex[:12]}")
    applicant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    input_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    ml_prob: Mapped[float] = mapped_column(Float, nullable=False)
    cbes_prob: Mapped[float] = mapped_column(Float, nullable=False)
    final_decision: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    documents: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class DeferredReview(Base):
    """One row per decision routed to a human reviewer (DEFER, or an
    exploration-arm sample of a would-be-auto decision).

    CAPTURE-ONLY TABLE. Per
    docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md
    section 3 ("Explicitly do not build yet"), no code anywhere reads
    `human_decision` as a training label, and there is no retraining trigger
    — scheduled, volume-based, or otherwise. Writing this table changes
    nothing about model behavior; that is the point. The gate conditions
    that must all hold before any relearning pass may consume these rows are
    listed in the same spec section.
    """

    __tablename__ = "deferred_reviews"

    # --- keys -------------------------------------------------------------
    review_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: f"rev-{uuid.uuid4().hex[:12]}"
    )
    application_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # --- full engine state at defer time ----------------------------------
    decision_reason: Mapped[str] = mapped_column(String, nullable=False)
    p_ml: Mapped[float] = mapped_column(Float, nullable=False)
    p_cbes: Mapped[float] = mapped_column(Float, nullable=False)
    p_blend: Mapped[float] = mapped_column(Float, nullable=False)
    disagreement: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    t_approve: Mapped[float] = mapped_column(Float, nullable=False)
    t_reject: Mapped[float] = mapped_column(Float, nullable=False)
    cbes_breakdown_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    # --- provenance: separates a fixed router's rows from a broken one's ---
    engine_version: Mapped[str] = mapped_column(String, nullable=False)
    threshold_artifact_hash: Mapped[str] = mapped_column(String, nullable=False)
    t_base: Mapped[float] = mapped_column(Float, nullable=False)

    # --- reviewer metadata (reviewer-attention-degradation check) ----------
    reviewer_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_spent_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- human decision (NEVER a training label; see class docstring) ------
    human_decision: Mapped[str | None] = mapped_column(String, nullable=True)
    human_reason_codes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    human_free_text: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- override-rate monitoring (SR 11-7) --------------------------------
    agreed_with_engine: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    override_direction: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewer_confidence: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- disparate-deferral check ------------------------------------------
    applicant_segment_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # --- outcome; explicitly encodes MNAR-ness -----------------------------
    realized_outcome: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outcome_observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    outcome_censored: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # --- control arm --------------------------------------------------------
    exploration_flag: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
