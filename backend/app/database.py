from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from backend.app import config as _config  # noqa: F401  # loads .env before the URL is read

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "smartlend.db"
LEGACY_DB_PATH = APP_ROOT / "smartlend.db"

# Preserve historical records from old backend-local DB path when upgrading.
if not DB_PATH.exists() and LEGACY_DB_PATH.exists():
    LEGACY_DB_PATH.replace(DB_PATH)

# SQLite on disk by default (zero-setup dev); a Postgres URL via
# SMARTLEND_DATABASE_URL for Docker Compose and RDS. The SQLite-specific
# helpers below are dialect-guarded, so both engines share this module.
SQLALCHEMY_DATABASE_URL = (
    os.environ.get("SMARTLEND_DATABASE_URL", "").strip() or f"sqlite:///{DB_PATH}"
)

# check_same_thread is a SQLite-only pysqlite argument; other drivers reject it.
_connect_args = (
    {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=_connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from backend.app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _configure_sqlite_durability()
    _ensure_documents_column()
    _seed_customer_profiles()


def _seed_customer_profiles() -> None:
    """Load the committed customer sample into `customer_profiles`.

    Imported lazily to keep `database` free of service-layer imports. Idempotent
    (a populated table is left alone) and never raises: the seed is demo data,
    and its absence must not stop the API from starting.
    """
    from backend.app.services.customer_seed_service import seed_customer_profiles

    seed_customer_profiles()


def _configure_sqlite_durability() -> None:
    with engine.begin() as connection:
        if connection.dialect.name != "sqlite":
            return

        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        connection.exec_driver_sql("PRAGMA synchronous=NORMAL")
        connection.exec_driver_sql("PRAGMA temp_store=MEMORY")
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")


def _ensure_documents_column() -> None:
    with engine.begin() as connection:
        if connection.dialect.name != "sqlite":
            return

        existing_columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(loan_applications)")}
        if "documents" not in existing_columns:
            connection.exec_driver_sql("ALTER TABLE loan_applications ADD COLUMN documents JSON NOT NULL DEFAULT '[]'")
