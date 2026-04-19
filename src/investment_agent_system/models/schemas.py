from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentRole(str, Enum):
    FUNDAMENTAL = "fundamental"
    QUANT = "quant"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    SKEPTIC = "skeptic"
    CATALYST = "catalyst"
    PORTFOLIO = "portfolio"
    COMMITTEE = "committee_chair"

    @classmethod
    def _missing_(cls, value: object) -> Optional["AgentRole"]:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases: dict[str, AgentRole] = {
            "committee": cls.COMMITTEE,
            "committee_chair": cls.COMMITTEE,
            "chair": cls.COMMITTEE,
        }
        return aliases.get(normalized)


class Stage(str, Enum):
    INDEPENDENT = "independent"
    CHALLENGE = "challenge"
    REBUTTAL = "rebuttal"
    SYNTHESIS = "synthesis"


class ClaimType(str, Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"
    ESTIMATE = "ESTIMATE"
    OPINION = "OPINION"

    @classmethod
    def _missing_(cls, value: object) -> Optional["ClaimType"]:
        if not isinstance(value, str):
            return None
        normalized = value.strip().upper()
        aliases = {
            "FACTUAL": cls.FACT,
            "INFERRED": cls.INFERENCE,
            "PROJECTION": cls.ESTIMATE,
            "VIEW": cls.OPINION,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return None


class Recommendation(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WATCHLIST = "WATCHLIST"
    PASS = "PASS"

    @classmethod
    def _missing_(cls, value: object) -> Optional["Recommendation"]:
        if not isinstance(value, str):
            return None
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "BUY": cls.LONG,
            "OUTPERFORM": cls.LONG,
            "OVERWEIGHT": cls.LONG,
            "SELL": cls.SHORT,
            "UNDERPERFORM": cls.SHORT,
            "UNDERWEIGHT": cls.SHORT,
            "HOLD": cls.WATCHLIST,
            "NEUTRAL": cls.WATCHLIST,
            "WAIT": cls.WATCHLIST,
            "NO_POSITION": cls.PASS,
            "AVOID": cls.PASS,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return None


class RunStatus(str, Enum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(description="Human readable source name")
    url: str = Field(description="Canonical URL for the source")
    published_at: Optional[datetime] = None
    evidence_id: Optional[int] = None


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[int] = None
    source_type: str
    source_name: str
    title: str
    url: str
    published_at: Optional[datetime] = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_type: ClaimType
    statement: str
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: str
    falsifiers: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ScoreCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_quality: float = Field(ge=0.0, le=10.0)
    valuation_attractiveness: float = Field(ge=0.0, le=10.0)
    earnings_durability: float = Field(ge=0.0, le=10.0)
    balance_sheet_strength: float = Field(ge=0.0, le=10.0)
    management_credibility: float = Field(ge=0.0, le=10.0)
    catalyst_strength: float = Field(ge=0.0, le=10.0)
    technical_setup: float = Field(ge=0.0, le=10.0)
    macro_sensitivity: float = Field(ge=0.0, le=10.0)
    downside_asymmetry: float = Field(ge=0.0, le=10.0)

    @property
    def weighted_total(self) -> float:
        # Slightly prioritize downside and durability.
        return round(
            (
                self.business_quality * 0.12
                + self.valuation_attractiveness * 0.12
                + self.earnings_durability * 0.14
                + self.balance_sheet_strength * 0.1
                + self.management_credibility * 0.1
                + self.catalyst_strength * 0.12
                + self.technical_setup * 0.1
                + self.macro_sensitivity * 0.08
                + self.downside_asymmetry * 0.12
            ),
            2,
        )


class AgentMemo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = None
    role: AgentRole
    thesis: str
    recommendation: Recommendation
    horizon: str
    confidence: float = Field(ge=0.0, le=1.0)
    strongest_supporting_points: list[Claim] = Field(min_length=3, max_length=6)
    strongest_risks: list[Claim] = Field(min_length=3, max_length=6)
    what_would_change_my_mind: list[str] = Field(min_length=2, max_length=8)
    unknowns: list[str] = Field(min_length=2, max_length=8)
    scorecard: ScoreCard


class Challenge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = None
    challenger_role: AgentRole
    target_role: AgentRole
    target_claims_under_attack: list[str] = Field(min_length=1, max_length=5)
    critical_questions: list[str] = Field(min_length=3, max_length=8)
    severity: float = Field(ge=0.0, le=1.0)


class RebuttalPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    response: str
    status: str = Field(description="One of: resolved, partially_resolved, unresolved.")
    updated_confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)


class Rebuttal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = None
    responder_role: AgentRole
    challenger_role: AgentRole
    responses: list[RebuttalPoint] = Field(min_length=1, max_length=10)
    revised_view: str


class ConfidenceDecomposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_quality_confidence: float = Field(ge=0.0, le=1.0)
    model_confidence: float = Field(ge=0.0, le=1.0)
    market_timing_confidence: float = Field(ge=0.0, le=1.0)
    overall_conviction: float = Field(ge=0.0, le=1.0)


class PositionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggested_size_pct_nav: float = Field(ge=0.0, le=20.0)
    entry_plan: str
    exit_plan: str
    hedging_plan: str
    stop_conditions: list[str] = Field(min_length=2, max_length=8)


class FinalThesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = None
    ticker: str
    recommendation: Recommendation
    horizon: str
    variant_perception: str
    why_market_is_wrong: str
    base_case: str
    bull_case: str
    bear_case: str
    supporting_evidence: list[str] = Field(min_length=3, max_length=12)
    main_risks: list[str] = Field(min_length=3, max_length=12)
    catalysts: list[str] = Field(min_length=3, max_length=12)
    valuation_and_expected_return: str
    position_plan: PositionPlan
    key_debates: list[str] = Field(min_length=3, max_length=12)
    unresolved_uncertainties: list[str] = Field(min_length=2, max_length=10)
    monitoring_checklist: list[str] = Field(min_length=4, max_length=16)
    confidence: ConfidenceDecomposition
    cannot_verify: list[str] = Field(min_length=1, max_length=10)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    horizon: str = "6-12 months"
    include_roles: Optional[list[AgentRole]] = None
    model: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            msg = "Ticker cannot be empty"
            raise ValueError(msg)
        return normalized


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    ticker: str
    horizon: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: RunStatus = RunStatus.STARTED
    error: Optional[str] = None
