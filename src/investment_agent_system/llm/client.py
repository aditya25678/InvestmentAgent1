from __future__ import annotations

import asyncio
import json
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar

from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from investment_agent_system.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _status_code(exc: ResponseError) -> int:
    value = getattr(exc, "status_code", 0)
    return int(value or 0)


def _is_model_unavailable_error(exc: ResponseError) -> bool:
    status_code = _status_code(exc)
    message = str(exc).lower()
    return status_code == 404 or (
        status_code == 403
        and (
            "high volume" in message
            or "subscription is required" in message
            or "capacity" in message
            or "access denied for model" in message
        )
    )


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, ResponseError):
        return _status_code(exc) in TRANSIENT_STATUS_CODES
    return isinstance(exc, TimeoutError)


class LLMClient(ABC):
    @abstractmethod
    async def generate_structured(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
        model_override: Optional[str] = None,
    ) -> T:
        raise NotImplementedError


class OllamaCloudLLMClient(LLMClient):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.ollama_api_key:
            msg = (
                "OLLAMA_API_KEY is missing. Set it in environment or .env file. "
                "The system requires a live Ollama cloud model to run agent debates."
            )
            raise ValueError(msg)
        self.settings = settings
        self.client = Client(
            host=settings.resolved_ollama_host(),
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
        )
        self._model_state_lock = threading.Lock()
        self._blocked_models: set[str] = set()
        self._preferred_model_by_request: dict[str, str] = {}

    @retry(
        retry=retry_if_exception(_is_retryable_exception),
        wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5), reraise=True
    )
    async def generate_structured(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
        model_override: Optional[str] = None,
    ) -> T:
        requested_model = model_override or self.settings.ollama_model
        candidates = self._build_model_candidates(requested_model)
        errors: list[str] = []

        for model in candidates:
            try:
                payload = await self._chat_once(model, system_prompt, user_prompt, schema)
            except ResponseError as exc:
                status_code = _status_code(exc)
                errors.append(f"{model} -> HTTP {status_code}: {exc}")
                if status_code == 401:
                    msg = (
                        "Ollama API authentication failed (HTTP 401). "
                        "Check OLLAMA_API_KEY in your .env."
                    )
                    raise ValueError(msg) from exc
                if _is_model_unavailable_error(exc):
                    self._mark_model_blocked(model)
                    continue
                raise

            content = _extract_ollama_content(payload)
            if not content:
                logger.error("Empty content from Ollama response: %s", payload)
                errors.append(f"{model} -> empty response")
                self._mark_model_blocked(model)
                continue

            try:
                output = await self._coerce_response_to_schema(
                    schema=schema,
                    model=model,
                    content=content,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except (ValidationError, ValueError) as exc:
                errors.append(f"{model} -> schema/parse failure: {exc}")
                continue

            self._remember_working_model(requested_model=requested_model, working_model=model)
            if model != requested_model:
                logger.warning(
                    "Primary Ollama model '%s' unavailable. Using fallback model '%s'.",
                    requested_model,
                    model,
                )
            return output

        detail = "; ".join(errors[:3]).strip() or "no successful model responses"
        msg = (
            f"No accessible Ollama model for request '{requested_model}'. "
            f"Attempted: {', '.join(candidates)}. "
            "If glm-5.1:cloud is capacity-gated, either upgrade your Ollama account "
            "or set OLLAMA_FALLBACK_MODELS to accessible cloud models. "
            f"Recent errors: {detail}"
        )
        raise ValueError(msg)

    async def _coerce_response_to_schema(
        self,
        *,
        schema: type[T],
        model: str,
        content: str,
        system_prompt: str,
        user_prompt: str,
    ) -> T:
        current_content = content
        attempts = max(0, self.settings.ollama_schema_repair_attempts)

        for _ in range(attempts + 1):
            try:
                parsed_json = _load_json_payload(current_content)
            except ValueError as exc:
                if attempts <= 0:
                    raise
                current_content = await self._request_schema_repair(
                    model=model,
                    schema=schema,
                    invalid_payload={"raw_text": current_content[:5000]},
                    validation_errors=f"JSON parse error: {exc}",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                attempts -= 1
                continue

            try:
                return schema.model_validate(parsed_json)
            except ValidationError as exc:
                if attempts <= 0:
                    raise
                current_content = await self._request_schema_repair(
                    model=model,
                    schema=schema,
                    invalid_payload=parsed_json,
                    validation_errors=_summarize_validation_error(exc),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                attempts -= 1

        msg = "Unable to validate model output against schema"
        raise ValueError(msg)

    async def _chat_once(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> Any:
        return await asyncio.wait_for(
            asyncio.to_thread(
                self.client.chat,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                format=schema.model_json_schema(),
                options={"temperature": self.settings.ollama_temperature},
            ),
            timeout=float(self.settings.ollama_request_timeout_seconds),
        )

    async def _request_schema_repair(
        self,
        *,
        model: str,
        schema: type[T],
        invalid_payload: Any,
        validation_errors: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        schema_json = _safe_json_dumps(schema.model_json_schema())
        input_json = _safe_json_dumps(invalid_payload)
        repair_user_prompt = (
            "Convert INPUT_JSON into a JSON object that strictly satisfies TARGET_SCHEMA.\n"
            "Return only a single JSON object and no markdown.\n"
            "Preserve the original research intent and do not invent new sources.\n"
            "For required fields that are missing, infer conservatively and keep "
            "uncertainty explicit.\n\n"
            f"VALIDATION_ERRORS:\n{validation_errors}\n\n"
            f"TARGET_SCHEMA:\n{schema_json}\n\n"
            f"ORIGINAL_SYSTEM_PROMPT:\n{system_prompt[:1200]}\n\n"
            f"ORIGINAL_USER_PROMPT:\n{user_prompt[:2200]}\n\n"
            f"INPUT_JSON:\n{input_json}"
        )
        payload = await asyncio.wait_for(
            asyncio.to_thread(
                self.client.chat,
                model=model,
                messages=[
                    {"role": "system", "content": "You are a strict JSON schema repair engine."},
                    {"role": "user", "content": repair_user_prompt},
                ],
                stream=False,
                format=schema.model_json_schema(),
                options={"temperature": 0},
            ),
            timeout=float(self.settings.ollama_request_timeout_seconds),
        )
        content = _extract_ollama_content(payload)
        if not content:
            raise ValueError("Schema repair produced empty content")
        return content

    def _build_model_candidates(self, requested_model: str) -> list[str]:
        candidates: list[str] = []

        with self._model_state_lock:
            preferred = self._preferred_model_by_request.get(requested_model)
            blocked = set(self._blocked_models)

        if preferred and preferred not in blocked:
            candidates.append(preferred)

        if requested_model not in blocked:
            candidates.append(requested_model)

        if requested_model.endswith(":cloud"):
            unsuffixed = requested_model[: -len(":cloud")]
            if unsuffixed and unsuffixed not in blocked:
                candidates.append(unsuffixed)

        for fallback in self.settings.ollama_fallback_models():
            if fallback not in blocked:
                candidates.append(fallback)

        deduped: list[str] = []
        for item in candidates:
            if item and item not in deduped:
                deduped.append(item)
        return deduped or [requested_model]

    def _remember_working_model(self, requested_model: str, working_model: str) -> None:
        with self._model_state_lock:
            self._preferred_model_by_request[requested_model] = working_model
            if working_model in self._blocked_models:
                self._blocked_models.remove(working_model)

    def _mark_model_blocked(self, model: str) -> None:
        with self._model_state_lock:
            self._blocked_models.add(model)


def _extract_ollama_content(payload: Any) -> str:
    if hasattr(payload, "message") and hasattr(payload.message, "content"):
        return str(payload.message.content).strip()
    if isinstance(payload, dict):
        return str(payload.get("message", {}).get("content", "")).strip()
    return ""


def _load_json_payload(content: str) -> dict[str, Any]:
    try:
        loaded = json.loads(content)
        if isinstance(loaded, dict):
            return loaded
        msg = "Structured output was not a JSON object"
        raise ValueError(msg)
    except json.JSONDecodeError:
        # Fallback for occasional wrapped responses with prose around JSON.
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.error("Failed to parse model output as JSON: %s", content)
            raise ValueError("Model returned invalid JSON") from None
        snippet = content[start : end + 1]
        try:
            loaded = json.loads(snippet)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse model output as JSON snippet: %s", content)
            raise ValueError("Model returned invalid JSON") from exc
        if not isinstance(loaded, dict):
            msg = "Structured output was not a JSON object"
            raise ValueError(msg) from None
        return loaded


def _summarize_validation_error(exc: ValidationError) -> str:
    details: list[str] = []
    for err in exc.errors()[:20]:
        loc = ".".join(str(part) for part in err.get("loc", []))
        msg = str(err.get("msg", "validation error"))
        details.append(f"{loc}: {msg}")
    return "; ".join(details)


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)
