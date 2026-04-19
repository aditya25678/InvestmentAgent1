from investment_agent_system.config import Settings
from investment_agent_system.models.schemas import AgentRole, ClaimType, Recommendation


def test_recommendation_aliases() -> None:
    assert Recommendation("BUY") == Recommendation.LONG
    assert Recommendation("sell") == Recommendation.SHORT
    assert Recommendation("hold") == Recommendation.WATCHLIST
    assert Recommendation("avoid") == Recommendation.PASS


def test_claim_type_aliases() -> None:
    assert ClaimType("fact") == ClaimType.FACT
    assert ClaimType("inferred") == ClaimType.INFERENCE
    assert ClaimType("projection") == ClaimType.ESTIMATE
    assert ClaimType("view") == ClaimType.OPINION


def test_agent_role_aliases() -> None:
    assert AgentRole("committee") == AgentRole.COMMITTEE
    assert AgentRole("committee-chair") == AgentRole.COMMITTEE
    assert AgentRole("committee_chair") == AgentRole.COMMITTEE


def test_ollama_fallback_models_parser_dedupes() -> None:
    settings = Settings(
        _env_file=None,
        OLLAMA_FALLBACK_MODELS="gpt-oss:20b-cloud, qwen3.5:cloud, gpt-oss:20b-cloud",
    )
    assert settings.ollama_fallback_models() == ["gpt-oss:20b-cloud", "qwen3.5:cloud"]
