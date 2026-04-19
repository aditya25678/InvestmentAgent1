from datetime import datetime, timezone

from investment_agent_system.models.schemas import AgentRole, EvidenceItem
from investment_agent_system.research.context_builder import (
    render_research_packet,
    select_evidence_for_role,
)


def test_role_evidence_filtering() -> None:
    evidence = [
        EvidenceItem(
            id=1,
            source_type="market_data",
            source_name="source",
            title="market",
            url="https://example.com/1",
            content="market content",
            published_at=datetime.now(timezone.utc),
        ),
        EvidenceItem(
            id=2,
            source_type="filings",
            source_name="source",
            title="filing",
            url="https://example.com/2",
            content="filing content",
            published_at=datetime.now(timezone.utc),
        ),
    ]
    quant_evidence = select_evidence_for_role(AgentRole.QUANT, evidence)
    assert len(quant_evidence) == 1
    assert quant_evidence[0].source_type == "market_data"

    packet = render_research_packet("AAPL", "6-12 months", AgentRole.QUANT, evidence)
    assert "EVIDENCE_ID=1" in packet
    assert "EVIDENCE_ID=2" not in packet
