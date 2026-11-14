from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from .models import Trace, TraceSummary

Base = declarative_base()


class TraceRecord(Base):
    """SQLAlchemy record for a complete trace.

    The full nested tree is stored as JSON; selected fields are
    denormalized for fast listing and ordering.
    """

    __tablename__ = "traces"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    start = Column(DateTime, nullable=False)
    duration = Column(Float, nullable=False)
    success = Column(Boolean, default=True)
    total_tokens = Column(Integer, default=0)
    trace_json = Column(Text, nullable=False)


class SQLiteStore:
    """Persistent SQLite storage for FlowTrace.

    Zero-config local database. Full traces are stored as JSON
    (easy round-tripping with Pydantic) while key fields are
    indexed for the summary list.
    """

    def __init__(self, db_path: str = "flowtrace.db") -> None:
        """Initialize the store.

        Args:
            db_path: Path to SQLite file (default: flowtrace.db in cwd).
                     Parent directories are created automatically.
        """
        self.db_path = db_path
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.create_tables()

    def create_tables(self) -> None:
        """Create schema if it does not exist (idempotent)."""
        Base.metadata.create_all(bind=self.engine)

    def save_trace(self, trace: Union[Trace, Any]) -> str:
        """Persist a trace.

        Accepts either a Pydantic Trace model or a tracer.TraceContext
        (which is automatically converted). Returns the trace id.
        """
        if not isinstance(trace, Trace):
            # Assume it is a TraceContext from tracer.py
            trace = Trace.from_tracer_context(trace)

        with self.SessionLocal() as session:
            data = trace.model_dump(mode="json")

            record = TraceRecord(
                id=trace.id,
                name=trace.name,
                start=trace.start,
                duration=float(trace.duration),
                success=bool(trace.success),
                total_tokens=int(trace.total_tokens),
                trace_json=json.dumps(data),
            )

            existing = session.get(TraceRecord, trace.id)
            if existing:
                for attr, value in vars(record).items():
                    if not attr.startswith("_"):
                        setattr(existing, attr, value)
            else:
                session.add(record)

            session.commit()
            return trace.id

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a full trace by id or return None."""
        with self.SessionLocal() as session:
            record = session.get(TraceRecord, trace_id)
            if not record:
                return None
            data = json.loads(record.trace_json)
            return Trace.model_validate(data)

    def list_traces(self, limit: int = 50) -> List[TraceSummary]:
        """Return the most recent traces as lightweight summaries."""
        with self.SessionLocal() as session:
            stmt = (
                select(TraceRecord)
                .order_by(TraceRecord.start.desc())
                .limit(limit)
            )
            records = session.execute(stmt).scalars().all()

            return [
                TraceSummary(
                    id=r.id,
                    name=r.name,
                    start=r.start,
                    duration_ms=round(r.duration * 1000, 2),
                    success=bool(r.success),
                    total_tokens=int(r.total_tokens or 0),
                )
                for r in records
            ]

    def clear_all(self) -> int:
        """Delete every stored trace. Returns number of deleted rows.

        Used by the CLI ``flowtrace clear``.
        """
        with self.SessionLocal() as session:
            stmt = select(TraceRecord)
            records = session.execute(stmt).scalars().all()
            count = len(records)
            for rec in records:
                session.delete(rec)
            session.commit()
            return count
