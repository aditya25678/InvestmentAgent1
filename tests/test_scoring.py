from investment_agent_system.models.schemas import (
    AgentMemo,
    AgentRole,
    Claim,
    ClaimType,
    Recommendation,
    ScoreCard,
)
from investment_agent_system.research.scoring import aggregate_scorecards


def _claim(text: str) -> Claim:
    return Claim(
        claim_type=ClaimType.FACT,
        statement=text,
        confidence=0.7,
        horizon="6-12 months",
        falsifiers=["falsifier"],
        citations=[],
    )


def _memo(role: AgentRole, score: float) -> AgentMemo:
    return AgentMemo(
        role=role,
        thesis="thesis",
        recommendation=Recommendation.WATCHLIST,
        horizon="6-12 months",
        confidence=0.6,
        strongest_supporting_points=[_claim("a"), _claim("b"), _claim("c")],
        strongest_risks=[_claim("d"), _claim("e"), _claim("f")],
        what_would_change_my_mind=["x", "y"],
        unknowns=["u1", "u2"],
        scorecard=ScoreCard(
            business_quality=score,
            valuation_attractiveness=score,
            earnings_durability=score,
            balance_sheet_strength=score,
            management_credibility=score,
            catalyst_strength=score,
            technical_setup=score,
            macro_sensitivity=score,
            downside_asymmetry=score,
        ),
    )


def test_aggregate_scorecards() -> None:
    memos = [_memo(AgentRole.FUNDAMENTAL, 6.0), _memo(AgentRole.QUANT, 8.0)]
    result = aggregate_scorecards(memos)
    assert result["business_quality"] == 7.0
    assert result["weighted_total_mean"] == 7.0
