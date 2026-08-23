"""Database engine and session plumbing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.infrastructure.models import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        kwargs: dict = {"pool_pre_ping": True, "future": True}
        # SQLite (used by the test suite) rejects pool sizing arguments.
        if not settings.database_url.startswith("sqlite"):
            kwargs.update(
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                pool_recycle=1800,
            )
        _engine = create_engine(settings.database_url, **kwargs)
    return _engine


def get_session_factory() -> "sessionmaker[Session]":
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False
        )
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on failure."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency — read-only request scope."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def create_all() -> None:
    Base.metadata.create_all(bind=get_engine())


def drop_all() -> None:
    Base.metadata.drop_all(bind=get_engine())


def ping() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def reset_engine() -> None:
    """Drop cached engine/factory. Used by tests after changing settings."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
