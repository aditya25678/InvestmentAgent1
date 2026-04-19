from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from investment_agent_system.config import get_settings


class Base(DeclarativeBase):
    pass


class ResearchRunORM(Base):
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    horizon: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    evidence_items: Mapped[list["EvidenceORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    claims: Mapped[list["ClaimORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    memos: Mapped[list["MemoORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    challenges: Mapped[list["ChallengeORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    rebuttals: Mapped[list["RebuttalORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    final_theses: Mapped[list["FinalThesisORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvidenceORM(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.id"), index=True)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    source_name: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(String(1000))
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    run: Mapped[ResearchRunORM] = relationship(back_populates="evidence_items")


class ClaimORM(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.id"), index=True)
    agent_role: Mapped[str] = mapped_column(String(64), index=True)
    stage: Mapped[str] = mapped_column(String(32), index=True)
    claim_type: Mapped[str] = mapped_column(String(32), index=True)
    statement: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    horizon: Mapped[str] = mapped_column(String(64))
    falsifiers_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="active")
    parent_claim_id: Mapped[Optional[int]] = mapped_column(ForeignKey("claims.id"), nullable=True)

    run: Mapped[ResearchRunORM] = relationship(back_populates="claims")


class MemoORM(Base):
    __tablename__ = "memos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.id"), index=True)
    role: Mapped[str] = mapped_column(String(64), index=True)
    stage: Mapped[str] = mapped_column(String(32), index=True)
    content_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    run: Mapped[ResearchRunORM] = relationship(back_populates="memos")


class ChallengeORM(Base):
    __tablename__ = "challenges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.id"), index=True)
    challenger_role: Mapped[str] = mapped_column(String(64), index=True)
    target_role: Mapped[str] = mapped_column(String(64), index=True)
    content_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    run: Mapped[ResearchRunORM] = relationship(back_populates="challenges")


class RebuttalORM(Base):
    __tablename__ = "rebuttals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.id"), index=True)
    responder_role: Mapped[str] = mapped_column(String(64), index=True)
    challenger_role: Mapped[str] = mapped_column(String(64), index=True)
    content_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    run: Mapped[ResearchRunORM] = relationship(back_populates="rebuttals")


class FinalThesisORM(Base):
    __tablename__ = "final_theses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    recommendation: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    content_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    run: Mapped[ResearchRunORM] = relationship(back_populates="final_theses")


settings = get_settings()
engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
