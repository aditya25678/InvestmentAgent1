from __future__ import annotations

import json
from typing import Iterable, Optional

from investment_agent_system.agents.prompts import (
    challenge_system_prompt,
    committee_system_prompt,
    independent_system_prompt,
    rebuttal_system_prompt,
)
from investment_agent_system.llm.client import LLMClient
from investment_agent_system.models.schemas import (
    AgentMemo,
    AgentRole,
    Challenge,
    EvidenceItem,
    FinalThesis,
    Rebuttal,
)
from investment_agent_system.research.context_builder import (
    render_global_packet,
    render_research_packet,
)


class AgentEngine:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def write_independent_memo(
        self,
        *,
        run_id: str,
        role: AgentRole,
        ticker: str,
        horizon: str,
        evidence: list[EvidenceItem],
        model_override: Optional[str] = None,
    ) -> AgentMemo:
        system = independent_system_prompt(role)
        packet = render_research_packet(
            ticker=ticker, horizon=horizon, role=role, evidence=evidence
        )
        user_prompt = (
            "Create your independent memo.\n"
            "Requirements:\n"
            "- 3 to 6 strongest supporting points\n"
            "- 3 to 6 strongest risks\n"
            "- confidence in [0,1]\n"
            "- score each category from 0 to 10\n"
            "- Use citation URLs from evidence packet and include evidence IDs where possible.\n\n"
            f"{packet}"
        )
        memo = await self.llm.generate_structured(
            schema=AgentMemo,
            system_prompt=system,
            user_prompt=user_prompt,
            model_override=model_override,
        )
        memo.run_id = run_id
        memo.role = role
        return memo

    async def generate_challenge(
        self,
        *,
        run_id: str,
        challenger_role: AgentRole,
        ticker: str,
        horizon: str,
        target_memo: AgentMemo,
        evidence: list[EvidenceItem],
        model_override: Optional[str] = None,
    ) -> Challenge:
        system = challenge_system_prompt(challenger_role)
        packet = render_research_packet(
            ticker=ticker,
            horizon=horizon,
            role=challenger_role,
            evidence=evidence,
        )
        target_json = json.dumps(target_memo.model_dump(mode="json"), indent=2)
        user_prompt = (
            f"Target role: {target_memo.role.value}\n"
            "Generate a challenge against this memo. Identify weak assumptions and evidence gaps.\n"
            "Return direct questions requiring falsifiable answers.\n\n"
            f"TARGET_MEMO_JSON:\n{target_json}\n\n"
            f"ROLE_EVIDENCE:\n{packet}"
        )
        challenge = await self.llm.generate_structured(
            schema=Challenge,
            system_prompt=system,
            user_prompt=user_prompt,
            model_override=model_override,
        )
        challenge.run_id = run_id
        challenge.challenger_role = challenger_role
        challenge.target_role = target_memo.role
        return challenge

    async def generate_rebuttal(
        self,
        *,
        run_id: str,
        ticker: str,
        horizon: str,
        target_memo: AgentMemo,
        challenge: Challenge,
        evidence: list[EvidenceItem],
        model_override: Optional[str] = None,
    ) -> Rebuttal:
        system = rebuttal_system_prompt(target_memo.role)
        packet = render_research_packet(
            ticker=ticker,
            horizon=horizon,
            role=target_memo.role,
            evidence=evidence,
        )
        user_prompt = (
            "Respond point-by-point to challenge questions.\n"
            "If a question cannot be resolved, mark unresolved and lower confidence.\n"
            "Keep responses evidence-based with citations.\n\n"
            f"TARGET_MEMO:\n{json.dumps(target_memo.model_dump(mode='json'), indent=2)}\n\n"
            f"CHALLENGE:\n{json.dumps(challenge.model_dump(mode='json'), indent=2)}\n\n"
            f"ROLE_EVIDENCE:\n{packet}"
        )
        rebuttal = await self.llm.generate_structured(
            schema=Rebuttal,
            system_prompt=system,
            user_prompt=user_prompt,
            model_override=model_override,
        )
        rebuttal.run_id = run_id
        rebuttal.responder_role = target_memo.role
        rebuttal.challenger_role = challenge.challenger_role
        return rebuttal

    async def synthesize_committee_view(
        self,
        *,
        run_id: str,
        ticker: str,
        horizon: str,
        memos: Iterable[AgentMemo],
        challenges: Iterable[Challenge],
        rebuttals: Iterable[Rebuttal],
        evidence: list[EvidenceItem],
        model_override: Optional[str] = None,
    ) -> FinalThesis:
        system = committee_system_prompt()
        payload = {
            "memos": [memo.model_dump(mode="json") for memo in memos],
            "challenges": [challenge.model_dump(mode="json") for challenge in challenges],
            "rebuttals": [rebuttal.model_dump(mode="json") for rebuttal in rebuttals],
        }
        packet = render_global_packet(ticker=ticker, horizon=horizon, evidence=evidence)
        user_prompt = (
            f"Ticker: {ticker}\nHorizon: {horizon}\n"
            "Synthesize a final investment thesis from internal committee materials only.\n"
            "You must provide long/short/watchlist/pass and implementation guidance.\n"
            "Include unresolved uncertainties and cannot-verify items.\n\n"
            f"INTERNAL_PAYLOAD:\n{json.dumps(payload, indent=2)}\n\n"
            f"GLOBAL_EVIDENCE:\n{packet}"
        )
        thesis = await self.llm.generate_structured(
            schema=FinalThesis,
            system_prompt=system,
            user_prompt=user_prompt,
            model_override=model_override,
        )
        thesis.run_id = run_id
        thesis.ticker = ticker
        thesis.horizon = horizon
        return thesis
