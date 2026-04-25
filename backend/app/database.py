from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "smartlend.db"
LEGACY_DB_PATH = APP_ROOT / "smartlend.db"

# Preserve historical records from old backend-local DB path when upgrading.
if not DB_PATH.exists() and LEGACY_DB_PATH.exists():
    LEGACY_DB_PATH.replace(DB_PATH)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
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
