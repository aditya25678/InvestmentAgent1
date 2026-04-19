import pytest

from investment_agent_system.models.schemas import RunRequest


def test_ticker_normalization() -> None:
    request = RunRequest(ticker=" aapl ")
    assert request.ticker == "AAPL"


def test_invalid_ticker_rejected() -> None:
    with pytest.raises(ValueError):
        RunRequest(ticker="")
