from __future__ import annotations

import json
from typing import Any, Protocol

from openai import OpenAI, OpenAIError

from .config import RuntimeConfig
from .types import Completion


class LanguageModel(Protocol):
    model: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Completion: ...


class OpenAICompatibleModel:
    """Chat-completions client for any OpenAI-compatible model endpoint."""

    def __init__(self, config: RuntimeConfig, *, client: Any | None = None) -> None:
        self.config = config
        self.model = config.model
        self.client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=float(config.timeout_seconds),
            max_retries=config.max_attempts - 1,
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Completion:
        response: Any | None = None
        choice: Any | None = None
        content: str | None = None
        response_attempt = 0
        for response_attempt in range(1, self.config.max_attempts + 1):
            request_failed = False
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except OpenAIError:
                request_failed = True
            if request_failed:
                raise RuntimeError(
                    "OpenAI-compatible model request failed for "
                    f"{self.config.endpoint_origin}"
                )
            if response is None:
                raise RuntimeError("model endpoint returned no chat completion")
            try:
                choice = response.choices[0]
                content = choice.message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("completion content is empty")
            except (AttributeError, IndexError, TypeError, ValueError):
                choice = None
                content = None
                if response_attempt < self.config.max_attempts:
                    continue
                raise RuntimeError(
                    "model endpoint returned an invalid chat completion"
                )
            break
        if response is None or choice is None or content is None:
            raise RuntimeError("model endpoint returned no validated chat completion")
        usage = response.usage.model_dump() if response.usage is not None else {}
        extra = response.model_extra or {}
        return Completion(
            text=content,
            audit={
                "purpose": purpose,
                "requested_model": self.model,
                "resolved_model": response.model,
                "base_url": self.config.base_url,
                "provider": extra.get("provider"),
                "request_id": response.id,
                "finish_reason": choice.finish_reason,
                "usage": usage,
                "sdk_max_attempts": self.config.max_attempts,
                "response_validation_attempts": response_attempt,
            },
        )


def json_object(text: str) -> dict[str, object]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model did not return a JSON object") from exc
        payload = json.loads(candidate[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    return payload
