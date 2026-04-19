from pathlib import Path

from investment_agent_system.models.schemas import (
    AgentMemo,
    AgentRole,
    Claim,
    ClaimType,
    ConfidenceDecomposition,
    FinalThesis,
    PositionPlan,
    Recommendation,
    ScoreCard,
)
from investment_agent_system.orchestration.service import ThesisService
from investment_agent_system.orchestration.workflow import TeamMemberMemo, WorkflowOutput


def _claim(text: str) -> Claim:
    return Claim(
        claim_type=ClaimType.FACT,
        statement=text,
        confidence=0.7,
        horizon="1-2 months",
        falsifiers=["falsifier"],
        citations=[],
    )


def _memo(role: AgentRole) -> AgentMemo:
    return AgentMemo(
        role=role,
        thesis=f"{role.value} thesis",
        recommendation=Recommendation.LONG,
        horizon="1-2 months",
        confidence=0.65,
        strongest_supporting_points=[_claim("a"), _claim("b"), _claim("c"), _claim("d")],
        strongest_risks=[_claim("e"), _claim("f"), _claim("g"), _claim("h")],
        what_would_change_my_mind=["x", "y"],
        unknowns=["u1", "u2"],
        scorecard=ScoreCard(
            business_quality=7.0,
            valuation_attractiveness=7.0,
            earnings_durability=7.0,
            balance_sheet_strength=7.0,
            management_credibility=7.0,
            catalyst_strength=7.0,
            technical_setup=7.0,
            macro_sensitivity=7.0,
            downside_asymmetry=7.0,
        ),
    )


def _final_thesis() -> FinalThesis:
    return FinalThesis(
        run_id="run-123",
        ticker="AAPL",
        recommendation=Recommendation.LONG,
        horizon="1-2 months",
        variant_perception="variant",
        why_market_is_wrong="reason",
        base_case="base",
        bull_case="bull",
        bear_case="bear",
        supporting_evidence=["one", "two", "three"],
        main_risks=["r1", "r2", "r3"],
        catalysts=["c1", "c2", "c3"],
        current_price=100.0,
        base_case_price_target=115.0,
        bull_case_price_target=130.0,
        bear_case_price_target=90.0,
        expected_return_base_pct=15.0,
        expected_downside_bear_pct=-10.0,
        valuation_method="method",
        valuation_and_expected_return="valuation text",
        position_plan=PositionPlan(
            suggested_size_pct_nav=3.0,
            position_expression="common stock",
            entry_price=100.0,
            add_price=97.0,
            take_profit_price=115.0,
            stop_loss_price=92.0,
            invalidation_price=89.0,
            risk_reward_ratio=2.0,
            entry_plan="entry",
            exit_plan="exit",
            hedging_plan="hedge",
            sizing_rationale="sizing",
            stop_conditions=["s1", "s2"],
        ),
        key_debates=["d1", "d2", "d3"],
        unresolved_uncertainties=["u1", "u2"],
        monitoring_checklist=["m1", "m2", "m3", "m4"],
        confidence=ConfidenceDecomposition(
            data_quality_confidence=0.7,
            model_confidence=0.7,
            market_timing_confidence=0.7,
            overall_conviction=0.7,
        ),
        cannot_verify=["cv1"],
    )


def test_reports_written_to_title_subfolder(tmp_path: Path) -> None:
    service = ThesisService()
    service.settings.reports_dir = str(tmp_path)

    team_memo = _memo(AgentRole.FUNDAMENTAL)
    output = WorkflowOutput(
        run_id="run-123",
        ticker="AAPL",
        horizon="1-2 months",
        request_title="AAPL Folder Smoke",
        team_member_memos=[
            TeamMemberMemo(role=AgentRole.FUNDAMENTAL, member_label="member_a", memo=team_memo),
            TeamMemberMemo(role=AgentRole.FUNDAMENTAL, member_label="member_b", memo=team_memo),
        ],
        memos=[team_memo],
        challenges=[],
        rebuttals=[],
        discussion_timeline=[],
        final_thesis=_final_thesis(),
        report_markdown="# test",
    )

    service._persist_reports(output)

    run_folder = tmp_path / "aapl_folder_smoke__run-123"
    assert run_folder.exists()
    assert (run_folder / "final_thesis.md").exists()
    assert (run_folder / "final_thesis.json").exists()
    assert (run_folder / "team_consensus_memos.json").exists()
    assert (run_folder / "team_member_memos.json").exists()
    assert (run_folder / "discussion.json").exists()
    assert (run_folder / "run_summary.json").exists()
