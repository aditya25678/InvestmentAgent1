from __future__ import annotations

from investment_agent_system.models.schemas import AgentMemo, Challenge, FinalThesis, Rebuttal
from investment_agent_system.research.scoring import aggregate_scorecards


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_final_markdown(
    *,
    run_id: str,
    ticker: str,
    horizon: str,
    memos: list[AgentMemo],
    challenges: list[Challenge],
    rebuttals: list[Rebuttal],
    final_thesis: FinalThesis,
) -> str:
    lines: list[str] = [
        f"# Investment Thesis - {ticker}",
        "",
        f"Run ID: `{run_id}`",
        f"Horizon: {horizon}",
        f"Recommendation: **{final_thesis.recommendation.value}**",
        "",
        "## Thesis",
        f"- Variant perception: {final_thesis.variant_perception}",
        f"- Why market may be wrong: {final_thesis.why_market_is_wrong}",
        f"- Base case: {final_thesis.base_case}",
        f"- Bull case: {final_thesis.bull_case}",
        f"- Bear case: {final_thesis.bear_case}",
        "",
        "## Evidence and Risks",
        "- Supporting evidence:",
    ]
    lines.extend([f"  - {item}" for item in final_thesis.supporting_evidence])
    lines.append("- Main risks:")
    lines.extend([f"  - {item}" for item in final_thesis.main_risks])
    lines.append("- Catalysts:")
    lines.extend([f"  - {item}" for item in final_thesis.catalysts])
    lines.extend(
        [
            "",
            "## Valuation and Position Plan",
            f"- Valuation / expected return: {final_thesis.valuation_and_expected_return}",
            f"- Suggested size (% NAV): {final_thesis.position_plan.suggested_size_pct_nav:.2f}",
            f"- Entry plan: {final_thesis.position_plan.entry_plan}",
            f"- Exit plan: {final_thesis.position_plan.exit_plan}",
            f"- Hedging plan: {final_thesis.position_plan.hedging_plan}",
            "- Stop conditions:",
        ]
    )
    lines.extend([f"  - {item}" for item in final_thesis.position_plan.stop_conditions])
    lines.extend(
        [
            "",
            "## Confidence",
            f"- Data quality: {_fmt_pct(final_thesis.confidence.data_quality_confidence)}",
            f"- Model confidence: {_fmt_pct(final_thesis.confidence.model_confidence)}",
            "- Market timing confidence: "
            f"{_fmt_pct(final_thesis.confidence.market_timing_confidence)}",
            f"- Overall conviction: {_fmt_pct(final_thesis.confidence.overall_conviction)}",
            "",
            "## Debate and Monitoring",
            "- Key debates:",
        ]
    )
    lines.extend([f"  - {item}" for item in final_thesis.key_debates])
    lines.append("- Unresolved uncertainties:")
    lines.extend([f"  - {item}" for item in final_thesis.unresolved_uncertainties])
    lines.append("- Could not verify:")
    lines.extend([f"  - {item}" for item in final_thesis.cannot_verify])
    lines.append("- Monitoring checklist:")
    lines.extend([f"  - {item}" for item in final_thesis.monitoring_checklist])

    lines.extend(["", "## Agent Snapshots"])
    for memo in memos:
        lines.extend(
            [
                f"### {memo.role.value}",
                f"- Recommendation: {memo.recommendation.value}",
                f"- Confidence: {_fmt_pct(memo.confidence)}",
                f"- Thesis: {memo.thesis}",
                f"- Scorecard total (0-10): {memo.scorecard.weighted_total:.2f}",
                "- Strongest supporting points:",
            ]
        )
        for claim_item in memo.strongest_supporting_points:
            lines.append(
                "  - "
                f"[{claim_item.claim_type.value} | {_fmt_pct(claim_item.confidence)}] "
                f"{claim_item.statement}"
            )
        lines.append("- Strongest risks:")
        for claim_item in memo.strongest_risks:
            lines.append(
                "  - "
                f"[{claim_item.claim_type.value} | {_fmt_pct(claim_item.confidence)}] "
                f"{claim_item.statement}"
            )
        lines.append("- What would change my mind:")
        for item in memo.what_would_change_my_mind:
            lines.append(f"  - {item}")
        lines.append("- Unknowns:")
        for item in memo.unknowns:
            lines.append(f"  - {item}")
        lines.append("")

    aggregate = aggregate_scorecards(memos)
    if aggregate:
        lines.extend(["## Aggregate Scorecard", ""])
        for key, value in aggregate.items():
            lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "## Process Stats",
            f"- Independent memos: {len(memos)}",
            f"- Challenge records: {len(challenges)}",
            f"- Rebuttal records: {len(rebuttals)}",
        ]
    )

    return "\n".join(lines)
