from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="investment-agent-system", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    ollama_api_key: str = Field(default="", alias="OLLAMA_API_KEY")
    ollama_model: str = Field(default="glm-5.1:cloud", alias="OLLAMA_MODEL")
    ollama_fallback_models_csv: str = Field(
        default="gpt-oss:20b-cloud,qwen3.5:cloud", alias="OLLAMA_FALLBACK_MODELS"
    )
    ollama_host: str = Field(default="https://ollama.com", alias="OLLAMA_HOST")
    ollama_base_url: str = Field(default="", alias="OLLAMA_BASE_URL")
    ollama_temperature: float = Field(default=0.2, alias="OLLAMA_TEMPERATURE")
    ollama_schema_repair_attempts: int = Field(default=1, alias="OLLAMA_SCHEMA_REPAIR_ATTEMPTS")
    ollama_request_timeout_seconds: int = Field(
        default=90, alias="OLLAMA_REQUEST_TIMEOUT_SECONDS"
    )

    newsapi_key: str = Field(default="", alias="NEWSAPI_KEY")
    fmp_api_key: str = Field(default="", alias="FMP_API_KEY")

    database_url: str = Field(default="sqlite:///./data/investment_agent.db", alias="DATABASE_URL")
    reports_dir: str = Field(default="./reports", alias="REPORTS_DIR")
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    max_news_articles: int = Field(default=20, alias="MAX_NEWS_ARTICLES")
    max_web_results: int = Field(default=15, alias="MAX_WEB_RESULTS")
    required_conda_env: str = Field(default="myenv", alias="REQUIRED_CONDA_ENV")
    enforce_conda_env: bool = Field(default=True, alias="ENFORCE_CONDA_ENV")
    sec_user_agent: str = Field(
        default="InvestmentAgentSystem/0.1 (research@yourdomain.com)", alias="SEC_USER_AGENT"
    )

    def ensure_directories(self) -> None:
        Path("data").mkdir(parents=True, exist_ok=True)
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)

    def active_environment_name(self) -> str:
        conda_name = os.getenv("CONDA_DEFAULT_ENV", "").strip()
        if conda_name:
            return conda_name
        venv_path = os.getenv("VIRTUAL_ENV", "").strip()
        if venv_path:
            return Path(venv_path).name
        return ""

    def validate_runtime_environment(self) -> None:
        if not self.enforce_conda_env:
            return
        active_env = self.active_environment_name()
        if active_env == self.required_conda_env:
            return
        msg = (
            f"Environment check failed. Activate conda env '{self.required_conda_env}' "
            f"before running. Current env: '{active_env or 'none'}'."
        )
        raise RuntimeError(msg)

    def resolved_ollama_host(self) -> str:
        host = self.ollama_host.strip() or self.ollama_base_url.strip()
        if not host:
            host = "https://ollama.com"
        if host.endswith("/api"):
            host = host[: -len("/api")]
        if host.endswith("/"):
            host = host[:-1]
        return host

    def ollama_fallback_models(self) -> list[str]:
        values: list[str] = []
        for raw in self.ollama_fallback_models_csv.split(","):
            value = raw.strip()
            if not value or value in values:
                continue
            values.append(value)
        return values


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
