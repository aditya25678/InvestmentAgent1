from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Sequence, TypeVar

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
class WorkflowOutput:
    run_id: str
    ticker: str
    horizon: str
    memos: list[AgentMemo]
    challenges: list[Challenge]
    rebuttals: list[Rebuttal]
    final_thesis: FinalThesis
    report_markdown: str


class InvestmentThesisWorkflow:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.engine = AgentEngine(llm=OllamaCloudLLMClient())

    async def run(self, request: RunRequest) -> WorkflowOutput:
        roles = request.include_roles or DEFAULT_ROLES
        metadata = RunMetadata(ticker=request.ticker, horizon=request.horizon)
        session = SessionLocal()
        repo = ResearchRepository(session)

        try:
            repo.create_run(
                metadata=metadata,
                metadata_json={
                    "roles": [r.value for r in roles],
                    "model_override": request.model,
                },
            )
            session.commit()

            aggregator = ResearchDataAggregator(repository=repo)
            evidence = await aggregator.collect_and_persist(
                run_id=metadata.run_id, ticker=request.ticker
            )
            session.commit()

            # Stage A: independent memos.
            memo_tasks: list[Awaitable[AgentMemo]] = []
            for role in roles:
                memo_tasks.append(
                    self.engine.write_independent_memo(
                        run_id=metadata.run_id,
                        role=role,
                        ticker=request.ticker,
                        horizon=request.horizon,
                        evidence=evidence,
                        model_override=request.model,
                    )
                )
            memo_results = (
                await _gather_limited_allow_errors(
                    memo_tasks,
                    limit=max(1, self.settings.ollama_max_concurrency),
                )
                if memo_tasks
                else []
            )
            memos: list[AgentMemo] = []
            for role, memo_result in zip(roles, memo_results):
                if isinstance(memo_result, BaseException):
                    logger.warning(
                        "Memo generation failed for role=%s: %s",
                        role.value,
                        memo_result,
                    )
                    continue
                memos.append(memo_result)
            for memo in memos:
                repo.save_memo(metadata.run_id, Stage.INDEPENDENT, memo)
            session.commit()
            if not memos:
                msg = "All independent memos failed; unable to synthesize final thesis."
                raise RuntimeError(msg)

            memo_map = {memo.role: memo for memo in memos}

            # Stage B: structured challenges.
            challenge_specs = [
                (challenger, target)
                for challenger, target in CHALLENGE_PAIRS
                if challenger in memo_map and target in memo_map
            ]
            challenge_tasks = [
                self.engine.generate_challenge(
                    run_id=metadata.run_id,
                    challenger_role=challenger,
                    ticker=request.ticker,
                    horizon=request.horizon,
                    target_memo=memo_map[target],
                    evidence=evidence,
                    model_override=request.model,
                )
                for challenger, target in challenge_specs
            ]
            challenge_results = (
                await _gather_limited_allow_errors(
                    challenge_tasks,
                    limit=max(1, self.settings.ollama_max_concurrency),
                )
                if challenge_tasks
                else []
            )
            challenges: list[Challenge] = []
            for (challenger, target), challenge_result in zip(challenge_specs, challenge_results):
                if isinstance(challenge_result, BaseException):
                    logger.warning(
                        "Challenge failed for %s -> %s: %s",
                        challenger.value,
                        target.value,
                        challenge_result,
                    )
                    continue
                challenges.append(challenge_result)
            for challenge in challenges:
                repo.save_challenge(metadata.run_id, challenge)
            session.commit()

            # Stage C: rebuttals.
            rebuttal_tasks: list[Awaitable[Rebuttal]] = []
            for challenge in challenges:
                target_memo = memo_map.get(challenge.target_role)
                if not target_memo:
                    continue
                rebuttal_tasks.append(
                    self.engine.generate_rebuttal(
                        run_id=metadata.run_id,
                        ticker=request.ticker,
                        horizon=request.horizon,
                        target_memo=target_memo,
                        challenge=challenge,
                        evidence=evidence,
                        model_override=request.model,
                    )
                )
            rebuttal_results = (
                await _gather_limited_allow_errors(
                    rebuttal_tasks,
                    limit=max(1, self.settings.ollama_max_concurrency),
                )
                if rebuttal_tasks
                else []
            )
            rebuttals: list[Rebuttal] = []
            for challenge, rebuttal_result in zip(challenges, rebuttal_results):
                if isinstance(rebuttal_result, BaseException):
                    logger.warning(
                        "Rebuttal failed for %s -> %s: %s",
                        challenge.target_role.value,
                        challenge.challenger_role.value,
                        rebuttal_result,
                    )
                    continue
                rebuttals.append(rebuttal_result)
            for rebuttal in rebuttals:
                repo.save_rebuttal(metadata.run_id, rebuttal)
            session.commit()

            # Stage D: committee synthesis.
            final_thesis = await self.engine.synthesize_committee_view(
                run_id=metadata.run_id,
                ticker=request.ticker,
                horizon=request.horizon,
                memos=memos,
                challenges=challenges,
                rebuttals=rebuttals,
                evidence=evidence,
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
                challenges=challenges,
                rebuttals=rebuttals,
                final_thesis=final_thesis,
            )
            return WorkflowOutput(
                run_id=metadata.run_id,
                ticker=request.ticker,
                horizon=request.horizon,
                memos=memos,
                challenges=challenges,
                rebuttals=rebuttals,
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


async def _gather_limited(
    tasks: Sequence[Awaitable[T]],
    *,
    limit: int,
) -> list[T]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _run(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return list(await asyncio.gather(*[_run(task) for task in tasks]))


async def _gather_limited_allow_errors(
    tasks: Sequence[Awaitable[T]],
    *,
    limit: int,
) -> list[T | BaseException]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _run(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return list(await asyncio.gather(*[_run(task) for task in tasks], return_exceptions=True))
