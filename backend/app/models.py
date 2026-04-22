from __future__ import annotations

from datetime import datetime, timezone
import uuid

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


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
