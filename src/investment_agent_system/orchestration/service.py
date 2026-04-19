from __future__ import annotations

import json
import re
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
                "memo_records": repo.fetch_memo_records(run_id),
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
            memo_records = repo.fetch_memo_records(run_id)
            team_members = [
                item for item in memo_records if item["stage"] == Stage.TEAM_DRAFT.value
            ]
            team_consensus = [
                item for item in memo_records if item["stage"] == Stage.TEAM_CONSENSUS.value
            ]

            grouped: dict[str, dict] = {}
            for row in team_consensus:
                role = str(row.get("role", "unknown"))
                grouped[role] = row.get("content", {})
            return {
                "run": run,
                "team_consensus_memos": grouped,
                "team_member_memos": team_members,
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
            timeline = repo.fetch_memo_records(run_id, stage=Stage.DISCUSSION)
            return {
                "run": run,
                "challenges": repo.fetch_challenges(run_id),
                "rebuttals": repo.fetch_rebuttals(run_id),
                "timeline": timeline,
            }
        finally:
            session.close()

    def _persist_reports(self, output: WorkflowOutput) -> None:
        base_report_dir = Path(self.settings.reports_dir)
        base_report_dir.mkdir(parents=True, exist_ok=True)
        run_folder = base_report_dir / f"{self._slugify(output.request_title)}__{output.run_id}"
        run_folder.mkdir(parents=True, exist_ok=True)

        md_path = run_folder / "final_thesis.md"
        json_path = run_folder / "final_thesis.json"
        agents_path = run_folder / "team_consensus_memos.json"
        team_members_path = run_folder / "team_member_memos.json"
        discussion_path = run_folder / "discussion.json"
        summary_path = run_folder / "run_summary.json"

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
        team_members_path.write_text(
            json.dumps(
                [
                    {
                        "role": item.role.value,
                        "member_label": item.member_label,
                        "memo": item.memo.model_dump(mode="json"),
                    }
                    for item in output.team_member_memos
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        discussion_path.write_text(
            json.dumps(
                {
                    "timeline": output.discussion_timeline,
                    "challenges": [item.model_dump(mode="json") for item in output.challenges],
                    "rebuttals": [item.model_dump(mode="json") for item in output.rebuttals],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(
                {
                    "run_id": output.run_id,
                    "ticker": output.ticker,
                    "horizon": output.horizon,
                    "request_title": output.request_title,
                    "folder": str(run_folder),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _slugify(self, value: str) -> str:
        normalized = value.strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        normalized = normalized.strip("_")
        return normalized[:80] or "thesis_request"
