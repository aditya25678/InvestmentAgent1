from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from investment_agent_system.models.schemas import (
    AgentMemo,
    Challenge,
    Claim,
    FinalThesis,
    Rebuttal,
    RunMetadata,
    RunStatus,
    Stage,
)
from investment_agent_system.storage.db import (
    ChallengeORM,
    ClaimORM,
    EvidenceORM,
    FinalThesisORM,
    MemoORM,
    RebuttalORM,
    ResearchRunORM,
)


class ResearchRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_run(self, metadata: RunMetadata, metadata_json: Optional[dict] = None) -> None:
        orm = ResearchRunORM(
            id=metadata.run_id,
            ticker=metadata.ticker,
            horizon=metadata.horizon,
            status=metadata.status.value,
            started_at=metadata.started_at,
            metadata_json=metadata_json or {},
        )
        self.session.add(orm)

    def complete_run(self, run_id: str) -> None:
        run = self._get_run(run_id)
        run.status = RunStatus.COMPLETED.value
        run.completed_at = datetime.now(timezone.utc)

    def fail_run(self, run_id: str, error: str) -> None:
        run = self._get_run(run_id)
        run.status = RunStatus.FAILED.value
        run.error = error
        run.completed_at = datetime.now(timezone.utc)

    def save_evidence(
        self,
        run_id: str,
        source_type: str,
        source_name: str,
        title: str,
        url: str,
        content: str,
        published_at: Optional[datetime] = None,
        metadata_json: Optional[dict] = None,
    ) -> int:
        orm = EvidenceORM(
            run_id=run_id,
            source_type=source_type,
            source_name=source_name,
            title=title,
            url=url,
            content=content,
            published_at=published_at,
            metadata_json=metadata_json or {},
        )
        self.session.add(orm)
        self.session.flush()
        return orm.id

    def save_memo(self, run_id: str, stage: Stage, memo: AgentMemo) -> None:
        self.save_memo_payload(
            run_id=run_id,
            stage=stage,
            role=memo.role.value,
            content_json=memo.model_dump(mode="json"),
        )
        for claim in memo.strongest_supporting_points:
            self._save_claim(run_id=run_id, role=memo.role.value, stage=stage.value, claim=claim)
        for claim in memo.strongest_risks:
            self._save_claim(run_id=run_id, role=memo.role.value, stage=stage.value, claim=claim)

    def save_memo_payload(
        self,
        *,
        run_id: str,
        stage: Stage,
        role: str,
        content_json: dict,
    ) -> None:
        self.session.add(
            MemoORM(
                run_id=run_id,
                role=role,
                stage=stage.value,
                content_json=content_json,
            )
        )

    def save_challenge(self, run_id: str, challenge: Challenge) -> None:
        self.session.add(
            ChallengeORM(
                run_id=run_id,
                challenger_role=challenge.challenger_role.value,
                target_role=challenge.target_role.value,
                content_json=challenge.model_dump(mode="json"),
            )
        )

    def save_rebuttal(self, run_id: str, rebuttal: Rebuttal) -> None:
        self.session.add(
            RebuttalORM(
                run_id=run_id,
                responder_role=rebuttal.responder_role.value,
                challenger_role=rebuttal.challenger_role.value,
                content_json=rebuttal.model_dump(mode="json"),
            )
        )

    def save_final_thesis(self, run_id: str, thesis: FinalThesis) -> None:
        self.session.add(
            FinalThesisORM(
                run_id=run_id,
                ticker=thesis.ticker,
                recommendation=thesis.recommendation.value,
                confidence=thesis.confidence.overall_conviction,
                content_json=thesis.model_dump(mode="json"),
            )
        )

    def fetch_final_thesis(self, run_id: str) -> Optional[dict]:
        stmt = (
            select(FinalThesisORM)
            .where(FinalThesisORM.run_id == run_id)
            .order_by(FinalThesisORM.id.desc())
        )
        row = self.session.execute(stmt).scalars().first()
        return row.content_json if row else None

    def fetch_run(self, run_id: str) -> Optional[dict]:
        run = self._get_run_optional(run_id)
        if not run:
            return None
        return {
            "run_id": run.id,
            "ticker": run.ticker,
            "horizon": run.horizon,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "error": run.error,
            "metadata": run.metadata_json,
        }

    def list_recent_runs(self, limit: int = 25) -> list[dict]:
        stmt = select(ResearchRunORM).order_by(ResearchRunORM.started_at.desc()).limit(limit)
        rows = self.session.execute(stmt).scalars().all()
        return [
            {
                "run_id": row.id,
                "ticker": row.ticker,
                "status": row.status,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            }
            for row in rows
        ]

    def fetch_evidence_for_run(self, run_id: str) -> list[dict]:
        stmt = (
            select(EvidenceORM).where(EvidenceORM.run_id == run_id).order_by(EvidenceORM.id.asc())
        )
        rows = self.session.execute(stmt).scalars().all()
        return [
            {
                "id": row.id,
                "source_type": row.source_type,
                "source_name": row.source_name,
                "title": row.title,
                "url": row.url,
                "published_at": row.published_at.isoformat() if row.published_at else None,
                "content": row.content,
                "metadata": row.metadata_json,
            }
            for row in rows
        ]

    def fetch_memos(self, run_id: str) -> list[dict]:
        stmt = select(MemoORM).where(MemoORM.run_id == run_id).order_by(MemoORM.id.asc())
        rows = self.session.execute(stmt).scalars().all()
        return [row.content_json for row in rows]

    def fetch_memo_records(self, run_id: str, stage: Optional[Stage] = None) -> list[dict]:
        stmt = select(MemoORM).where(MemoORM.run_id == run_id).order_by(MemoORM.id.asc())
        if stage is not None:
            stmt = stmt.where(MemoORM.stage == stage.value)
        rows = self.session.execute(stmt).scalars().all()
        return [
            {
                "id": row.id,
                "run_id": row.run_id,
                "role": row.role,
                "stage": row.stage,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "content": row.content_json,
            }
            for row in rows
        ]

    def fetch_challenges(self, run_id: str) -> list[dict]:
        stmt = (
            select(ChallengeORM)
            .where(ChallengeORM.run_id == run_id)
            .order_by(ChallengeORM.id.asc())
        )
        rows = self.session.execute(stmt).scalars().all()
        return [row.content_json for row in rows]

    def fetch_rebuttals(self, run_id: str) -> list[dict]:
        stmt = (
            select(RebuttalORM).where(RebuttalORM.run_id == run_id).order_by(RebuttalORM.id.asc())
        )
        rows = self.session.execute(stmt).scalars().all()
        return [row.content_json for row in rows]

    def _save_claim(self, run_id: str, role: str, stage: str, claim: Claim) -> None:
        self.session.add(
            ClaimORM(
                run_id=run_id,
                agent_role=role,
                stage=stage,
                claim_type=claim.claim_type.value,
                statement=claim.statement,
                confidence=claim.confidence,
                horizon=claim.horizon,
                falsifiers_json=claim.falsifiers,
                citations_json=[c.model_dump(mode="json") for c in claim.citations],
            )
        )

    def _get_run(self, run_id: str) -> ResearchRunORM:
        run = self._get_run_optional(run_id)
        if not run:
            msg = f"Run not found: {run_id}"
            raise ValueError(msg)
        return run

    def _get_run_optional(self, run_id: str) -> Optional[ResearchRunORM]:
        stmt = select(ResearchRunORM).where(ResearchRunORM.id == run_id)
        return self.session.execute(stmt).scalars().first()


def merge_claims(memos: Iterable[AgentMemo]) -> list[Claim]:
    claims: list[Claim] = []
    for memo in memos:
        claims.extend(memo.strongest_supporting_points)
        claims.extend(memo.strongest_risks)
    return claims
