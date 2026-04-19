from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from investment_agent_system.config import get_settings
from investment_agent_system.models.schemas import RunRequest, Stage
from investment_agent_system.orchestration.workflow import InvestmentThesisWorkflow, WorkflowOutput
from investment_agent_system.storage.db import SessionLocal
from investment_agent_system.storage.repository import ResearchRepository


class ThesisService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def run(self, request: RunRequest) -> WorkflowOutput:
        workflow = InvestmentThesisWorkflow()
        output = await workflow.run(request)
        self._persist_reports(output)
        return output

    def get_run(self, run_id: str) -> Optional[dict]:
        session = SessionLocal()
        try:
            repo = ResearchRepository(session)
            return repo.fetch_run(run_id)
        finally:
            session.close()

    def list_runs(self, limit: int = 25) -> list[dict]:
        session = SessionLocal()
        try:
            repo = ResearchRepository(session)
            return repo.list_recent_runs(limit=limit)
        finally:
            session.close()

    def get_final_thesis(self, run_id: str) -> Optional[dict]:
        session = SessionLocal()
        try:
            repo = ResearchRepository(session)
            return repo.fetch_final_thesis(run_id)
        finally:
            session.close()

    def get_full_audit(self, run_id: str) -> Optional[dict]:
        session = SessionLocal()
        try:
            repo = ResearchRepository(session)
            run = repo.fetch_run(run_id)
            if not run:
                return None
            return {
                "run": run,
                "evidence": repo.fetch_evidence_for_run(run_id),
                "memos": repo.fetch_memos(run_id),
                "challenges": repo.fetch_challenges(run_id),
                "rebuttals": repo.fetch_rebuttals(run_id),
                "final_thesis": repo.fetch_final_thesis(run_id),
            }
        finally:
            session.close()

    def get_agent_views(self, run_id: str) -> Optional[dict]:
        session = SessionLocal()
        try:
            repo = ResearchRepository(session)
            run = repo.fetch_run(run_id)
            if not run:
                return None
            memo_rows = repo.fetch_memo_records(run_id, stage=Stage.INDEPENDENT)
            return {
                "run": run,
                "memos": {row["role"]: row["content"] for row in memo_rows},
            }
        finally:
            session.close()

    def get_discussion(self, run_id: str) -> Optional[dict]:
        session = SessionLocal()
        try:
            repo = ResearchRepository(session)
            run = repo.fetch_run(run_id)
            if not run:
                return None
            return {
                "run": run,
                "challenges": repo.fetch_challenges(run_id),
                "rebuttals": repo.fetch_rebuttals(run_id),
            }
        finally:
            session.close()

    def _persist_reports(self, output: WorkflowOutput) -> None:
        report_dir = Path(self.settings.reports_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{output.run_id}_{output.ticker}"
        md_path = report_dir / f"{stem}.md"
        json_path = report_dir / f"{stem}.json"
        agents_path = report_dir / f"{stem}_agents.json"
        discussion_path = report_dir / f"{stem}_discussion.json"

        md_path.write_text(output.report_markdown, encoding="utf-8")
        json_path.write_text(
            json.dumps(output.final_thesis.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        agents_path.write_text(
            json.dumps(
                {memo.role.value: memo.model_dump(mode="json") for memo in output.memos},
                indent=2,
            ),
            encoding="utf-8",
        )
        discussion_path.write_text(
            json.dumps(
                {
                    "challenges": [item.model_dump(mode="json") for item in output.challenges],
                    "rebuttals": [item.model_dump(mode="json") for item in output.rebuttals],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
