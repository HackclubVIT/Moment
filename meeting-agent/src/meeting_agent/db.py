"""SQLAlchemy engine/session setup.

SQLite by default (``DATABASE_URL`` in ``.env``); pointing that at Postgres/MySQL
is the entire migration — nothing else in the API layer is SQLite-specific.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import Settings


class Base(DeclarativeBase):
    pass


def make_engine(settings: Settings):
    url = settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


engine = make_engine(Settings())
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def configure_engine(settings: Settings) -> None:
    """Repoint the module's engine/session factory. Used by tests to isolate storage
    so running the suite never touches the real DATABASE_URL."""
    global engine
    engine.dispose()
    engine = make_engine(settings)
    SessionLocal.configure(bind=engine)


def init_db() -> None:
    from . import dbmodels  # noqa: F401  (registers models on Base.metadata)

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
