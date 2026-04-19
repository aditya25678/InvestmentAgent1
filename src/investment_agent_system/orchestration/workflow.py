from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Sequence, TypeVar

from investment_agent_system.agents.engine import AgentEngine
from investment_agent_system.config import get_settings
from investment_agent_system.llm.client import OllamaCloudLLMClient
from investment_agent_system.models.schemas import (
    AgentMemo,
    AgentRole,
    Challenge,
    FinalThesis,
    Rebuttal,
    RunMetadata,
    RunRequest,
    Stage,
)
from investment_agent_system.reporting.renderer import render_final_markdown
from investment_agent_system.research.data_aggregator import ResearchDataAggregator
from investment_agent_system.storage.db import SessionLocal
from investment_agent_system.storage.repository import ResearchRepository

logger = logging.getLogger(__name__)
T = TypeVar("T")


DEFAULT_ROLES: list[AgentRole] = [
    AgentRole.FUNDAMENTAL,
    AgentRole.QUANT,
    AgentRole.MACRO,
    AgentRole.SENTIMENT,
    AgentRole.SKEPTIC,
    AgentRole.CATALYST,
    AgentRole.PORTFOLIO,
]

TEAM_MEMBER_LABELS = ["member_a", "member_b"]
DEBATE_ROUNDS = 2

CHALLENGE_PAIRS: list[tuple[AgentRole, AgentRole]] = [
    (AgentRole.SKEPTIC, AgentRole.FUNDAMENTAL),
    (AgentRole.QUANT, AgentRole.FUNDAMENTAL),
    (AgentRole.FUNDAMENTAL, AgentRole.QUANT),
    (AgentRole.MACRO, AgentRole.QUANT),
    (AgentRole.SENTIMENT, AgentRole.FUNDAMENTAL),
    (AgentRole.CATALYST, AgentRole.FUNDAMENTAL),
    (AgentRole.PORTFOLIO, AgentRole.FUNDAMENTAL),
    (AgentRole.PORTFOLIO, AgentRole.QUANT),
    (AgentRole.MACRO, AgentRole.FUNDAMENTAL),
]


@dataclass
class TeamMemberMemo:
    role: AgentRole
    member_label: str
    memo: AgentMemo


@dataclass
class WorkflowOutput:
    run_id: str
    ticker: str
    horizon: str
    request_title: str
    team_member_memos: list[TeamMemberMemo]
    memos: list[AgentMemo]
    challenges: list[Challenge]
    rebuttals: list[Rebuttal]
    discussion_timeline: list[dict[str, Any]]
    final_thesis: FinalThesis
    report_markdown: str


class InvestmentThesisWorkflow:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.engine = AgentEngine(llm=OllamaCloudLLMClient())

    async def run(self, request: RunRequest) -> WorkflowOutput:
        roles = request.include_roles or DEFAULT_ROLES
        request_title = request.request_title or f"{request.ticker} {request.horizon} thesis"
        metadata = RunMetadata(ticker=request.ticker, horizon=request.horizon)
        session = SessionLocal()
        repo = ResearchRepository(session)

        try:
            repo.create_run(
                metadata=metadata,
                metadata_json={
                    "roles": [r.value for r in roles],
                    "model_override": request.model,
                    "request_title": request_title,
                },
            )
            session.commit()

            aggregator = ResearchDataAggregator(repository=repo)
            evidence = await aggregator.collect_and_persist(
                run_id=metadata.run_id, ticker=request.ticker
            )
            session.commit()

            # Stage A1: two same-role team members produce independent drafts.
            team_member_tasks: list[asyncio.Task[AgentMemo]] = []
            member_index: list[tuple[AgentRole, str]] = []
            for role in roles:
                for member_label in TEAM_MEMBER_LABELS:
                    team_member_tasks.append(asyncio.create_task(
                        self.engine.write_team_member_memo(
                            run_id=metadata.run_id,
                            role=role,
                            member_label=member_label,
                            ticker=request.ticker,
                            horizon=request.horizon,
                            evidence=evidence,
                            model_override=request.model,
                        )
                    ))
                    member_index.append((role, member_label))
            team_member_raw = await _gather_limited(
                team_member_tasks,
                max(1, self.settings.ollama_max_concurrency),
            )
            team_member_memos: list[TeamMemberMemo] = []
            for (role, member_label), memo in zip(member_index, team_member_raw):
                team_member_memos.append(
                    TeamMemberMemo(role=role, member_label=member_label, memo=memo)
                )
                repo.save_memo_payload(
                    run_id=metadata.run_id,
                    stage=Stage.TEAM_DRAFT,
                    role=f"{role.value}:{member_label}",
                    content_json={
                        "team_role": role.value,
                        "member_label": member_label,
                        "memo": memo.model_dump(mode="json"),
                    },
                )
            session.commit()

            # Stage A2: per-role team consolidation.
            memo_tasks = [
                self.engine.consolidate_team_memo(
                    run_id=metadata.run_id,
                    role=role,
                    ticker=request.ticker,
                    horizon=request.horizon,
                    member_memos=[
                        item.memo
                        for item in team_member_memos
                        if item.role == role
                    ],
                    evidence=evidence,
                    model_override=request.model,
                )
                for role in roles
            ]
            memos = await _gather_limited(
                memo_tasks,
                max(1, self.settings.ollama_max_concurrency),
            )
            for memo in memos:
                repo.save_memo(metadata.run_id, Stage.TEAM_CONSENSUS, memo)
            session.commit()

            memo_map = {memo.role: memo for memo in memos}

            # Stage B/C: multi-round cross-team challenge/rebuttal discussion.
            discussion_timeline: list[dict[str, Any]] = []
            prior_rebuttals: dict[tuple[AgentRole, AgentRole], Rebuttal] = {}
            all_challenges: list[Challenge] = []
            all_rebuttals: list[Rebuttal] = []

            for round_index in range(1, DEBATE_ROUNDS + 1):
                challenge_tasks = []
                for challenger, target in CHALLENGE_PAIRS:
                    if challenger in memo_map and target in memo_map:
                        challenge_tasks.append(
                            self.engine.generate_challenge(
                                run_id=metadata.run_id,
                                round_index=round_index,
                                challenger_role=challenger,
                                ticker=request.ticker,
                                horizon=request.horizon,
                                target_memo=memo_map[target],
                                evidence=evidence,
                                prior_rebuttal=prior_rebuttals.get((challenger, target)),
                                model_override=request.model,
                            )
                        )

                round_challenges = (
                    await _gather_limited(
                        challenge_tasks,
                        max(1, self.settings.ollama_max_concurrency),
                    )
                    if challenge_tasks
                    else []
                )
                for challenge in round_challenges:
                    repo.save_challenge(metadata.run_id, challenge)
                    challenge_record = {
                        "round": round_index,
                        "type": "challenge",
                        "from_role": challenge.challenger_role.value,
                        "to_role": challenge.target_role.value,
                        "summary": " | ".join(challenge.critical_questions[:3]),
                        "severity": challenge.severity,
                        "critical_questions": challenge.critical_questions,
                    }
                    discussion_timeline.append(challenge_record)
                    repo.save_memo_payload(
                        run_id=metadata.run_id,
                        stage=Stage.DISCUSSION,
                        role=challenge.challenger_role.value,
                        content_json=challenge_record,
                    )
                session.commit()
                all_challenges.extend(round_challenges)

                rebuttal_tasks = []
                for challenge in round_challenges:
                    target_memo = memo_map.get(challenge.target_role)
                    if not target_memo:
                        continue
                    rebuttal_tasks.append(
                        self.engine.generate_rebuttal(
                            run_id=metadata.run_id,
                            round_index=round_index,
                            ticker=request.ticker,
                            horizon=request.horizon,
                            target_memo=target_memo,
                            challenge=challenge,
                            evidence=evidence,
                            model_override=request.model,
                        )
                    )
                round_rebuttals = (
                    await _gather_limited(
                        rebuttal_tasks,
                        max(1, self.settings.ollama_max_concurrency),
                    )
                    if rebuttal_tasks
                    else []
                )
                for rebuttal in round_rebuttals:
                    repo.save_rebuttal(metadata.run_id, rebuttal)
                    prior_rebuttals[(rebuttal.challenger_role, rebuttal.responder_role)] = rebuttal
                    rebuttal_record = {
                        "round": round_index,
                        "type": "rebuttal",
                        "from_role": rebuttal.responder_role.value,
                        "to_role": rebuttal.challenger_role.value,
                        "summary": rebuttal.revised_view,
                        "responses": [
                            point.model_dump(mode="json") for point in rebuttal.responses
                        ],
                    }
                    discussion_timeline.append(rebuttal_record)
                    repo.save_memo_payload(
                        run_id=metadata.run_id,
                        stage=Stage.DISCUSSION,
                        role=rebuttal.responder_role.value,
                        content_json=rebuttal_record,
                    )
                session.commit()
                all_rebuttals.extend(round_rebuttals)

            # Stage D: committee synthesis.
            team_notes: dict[str, list[dict[str, str]]] = {}
            for role in roles:
                notes: list[dict[str, str]] = []
                for item in [t for t in team_member_memos if t.role == role]:
                    notes.append(
                        {
                            "member_label": item.member_label,
                            "recommendation": item.memo.recommendation.value,
                            "confidence": f"{item.memo.confidence:.2f}",
                            "thesis": item.memo.thesis,
                        }
                    )
                team_notes[role.value] = notes

            final_thesis = await self.engine.synthesize_committee_view(
                run_id=metadata.run_id,
                ticker=request.ticker,
                horizon=request.horizon,
                memos=memos,
                challenges=all_challenges,
                rebuttals=all_rebuttals,
                evidence=evidence,
                team_notes=team_notes,
                model_override=request.model,
            )
            repo.save_final_thesis(metadata.run_id, final_thesis)
            repo.complete_run(metadata.run_id)
            session.commit()

            report_md = render_final_markdown(
                run_id=metadata.run_id,
                ticker=request.ticker,
                horizon=request.horizon,
                memos=memos,
                challenges=all_challenges,
                rebuttals=all_rebuttals,
                final_thesis=final_thesis,
                team_member_memos=[
                    {
                        "role": item.role.value,
                        "member_label": item.member_label,
                        "memo": item.memo.model_dump(mode="json"),
                    }
                    for item in team_member_memos
                ],
                discussion_timeline=discussion_timeline,
            )
            return WorkflowOutput(
                run_id=metadata.run_id,
                ticker=request.ticker,
                horizon=request.horizon,
                request_title=request_title,
                team_member_memos=team_member_memos,
                memos=memos,
                challenges=all_challenges,
                rebuttals=all_rebuttals,
                discussion_timeline=discussion_timeline,
                final_thesis=final_thesis,
                report_markdown=report_md,
            )
        except Exception as exc:
            logger.exception("Workflow failed for run %s", metadata.run_id)
            repo.fail_run(metadata.run_id, str(exc))
            session.commit()
            raise
        finally:
            session.close()


async def _gather_limited(tasks: Sequence[Awaitable[T]], limit: int) -> list[T]:
    semaphore = asyncio.Semaphore(limit)

    async def _run(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return list(await asyncio.gather(*[_run(task) for task in tasks]))
