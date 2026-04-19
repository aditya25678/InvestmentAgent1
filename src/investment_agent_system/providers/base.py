from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProviderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: str
    source_name: str
    title: str
    url: str
    content: str
    published_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchProvider(ABC):
    source_type: str
    source_name: str

    @abstractmethod
    async def fetch(self, ticker: str) -> list[ProviderResult]:
        raise NotImplementedError


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
