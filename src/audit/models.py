"""
src/audit/models.py

SQLAlchemy ORM models for the audit trail database.
Schema exactly as specified in architecture.md §7.
"""

from sqlalchemy import (
    Column, Integer, String, Text, Float, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session
from config import settings


class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    """
    Append-only audit log. Every node writes an entry here.
    state_snapshot_json stores the full ComplianceState at that point.
    """
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, nullable=False, index=True)
    node_name = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)
    input_summary = Column(Text)
    output_summary = Column(Text)
    confidence_score = Column(Float)
    state_snapshot_json = Column(Text, nullable=False)  # full ComplianceState JSON


class FlaggedIssue(Base):
    """
    Persistent record of each flagged issue.
    status: open → reviewed → confirmed / dismissed (analyst sets this).
    """
    __tablename__ = "flagged_issues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, nullable=False, index=True)
    clause_id = Column(String, nullable=False)
    issue_description = Column(Text, nullable=False)
    evidence_chunk_ids = Column(Text, nullable=False)   # JSON array string
    status = Column(String, default="open")             # open, reviewed, dismissed, confirmed
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(String, nullable=True)


def get_engine():
    """Return a SQLAlchemy engine connected to the configured SQLite DB."""
    db_url = f"sqlite:///{settings.audit_db_path}"
    return create_engine(db_url, connect_args={"check_same_thread": False})


def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Return a new SQLAlchemy session."""
    from sqlalchemy.orm import sessionmaker
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()
