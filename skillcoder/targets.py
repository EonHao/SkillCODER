from __future__ import annotations

from typing import Any, Protocol

from .config import RuntimeConfig
from .llm import LanguageModel, OpenAICompatibleModel


PROBE_SYSTEM_TEMPLATE = (
    "Follow the supplied agent skill. Return only the final answer.\n\n"
    "SKILL_MARKDOWN:\n{skill}"
)
SUPPORTED_PROBE_RUNTIMES = ("direct", "langchain", "camel")


class ProbeTarget(Protocol):
    model: str
    runtime: str

    def invoke(self, query: str, *, purpose: str) -> tuple[str, dict[str, object]]: ...


class DirectSkillTarget:
    """Execute a delivered Skill through the core OpenAI-compatible client."""

    runtime = "direct"

    def __init__(self, skill: str, model: LanguageModel) -> None:
        self.skill = skill
        self.client = model
        self.model = model.model

    def invoke(self, query: str, *, purpose: str) -> tuple[str, dict[str, object]]:
        completion = self.client.complete(
            PROBE_SYSTEM_TEMPLATE.format(skill=self.skill),
            query,
            purpose=purpose,
            max_tokens=2048,
        )
        return completion.text, completion.audit


def _load_langchain() -> tuple[Any, Any]:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LangChain probing requires the optional dependency: "
            "pip install 'skillcoder-core[langchain]'"
        ) from exc
    return ChatPromptTemplate, ChatOpenAI


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for block in content:
            if isinstance(block, str):
                fragments.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                fragments.append(block["text"])
        joined = "".join(fragments)
        if joined.strip():
            return joined
    text = getattr(message, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    if callable(text):
        rendered_text = text()
        if isinstance(rendered_text, str) and rendered_text.strip():
            return rendered_text
    raise RuntimeError("model target returned an empty or unsupported message")


class LangChainSkillTarget:
    """Execute a delivered Skill as a LangChain prompt-to-model runnable."""

    runtime = "langchain"

    def __init__(
        self,
        skill: str,
        config: RuntimeConfig,
        *,
        chat_model: Any | None = None,
    ) -> None:
        ChatPromptTemplate, ChatOpenAI = _load_langchain()
        self.skill = skill
        self.config = config
        self.model = config.model
        if chat_model is None:
            chat_model = ChatOpenAI(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                temperature=0,
                max_tokens=2048,
                timeout=float(config.timeout_seconds),
                max_retries=config.max_attempts - 1,
            )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROBE_SYSTEM_TEMPLATE),
                ("human", "{query}"),
            ]
        )
        self.chain = prompt | chat_model

    def invoke(self, query: str, *, purpose: str) -> tuple[str, dict[str, object]]:
        message: Any | None = None
        request_failed = False
        try:
            message = self.chain.invoke({"skill": self.skill, "query": query})
        except Exception:
            request_failed = True
        if request_failed:
            raise RuntimeError(
                f"LangChain model request failed for {self.config.endpoint_origin}"
            )
        response_metadata = getattr(message, "response_metadata", None)
        if not isinstance(response_metadata, dict):
            response_metadata = {}
        usage_metadata = getattr(message, "usage_metadata", None)
        if not isinstance(usage_metadata, dict):
            usage_metadata = {}
        return _message_text(message), {
            "purpose": purpose,
            "runtime": self.runtime,
            "requested_model": self.model,
            "resolved_model": response_metadata.get("model_name")
            or response_metadata.get("model")
            or self.model,
            "base_url": self.config.base_url,
            "request_id": getattr(message, "id", None),
            "finish_reason": response_metadata.get("finish_reason"),
            "usage": usage_metadata,
        }


def _load_camel() -> tuple[Any, Any, Any]:
    try:
        from camel.agents import ChatAgent
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType
    except ImportError as exc:
        raise RuntimeError(
            "CAMEL probing requires the optional dependency: "
            "pip install 'skillcoder-core[camel]'"
        ) from exc
    return ChatAgent, ModelFactory, ModelPlatformType


def _camel_response_text(response: Any) -> str:
    if bool(getattr(response, "terminated", False)):
        raise RuntimeError("CAMEL target terminated without a usable response")
    messages = getattr(response, "msgs", None)
    if not isinstance(messages, (list, tuple)) or len(messages) != 1:
        raise RuntimeError("CAMEL target returned an unsupported message count")
    return _message_text(messages[0])


def _camel_response_metadata(response: Any) -> tuple[object, object, dict[str, object]]:
    info = getattr(response, "info", None)
    if not isinstance(info, dict):
        return None, None, {}
    request_id = info.get("id")
    if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
        request_id = None
    finish_reason: object = None
    finish_reasons = info.get("finish_reasons")
    if finish_reasons is None:
        finish_reasons = info.get("termination_reasons")
    if isinstance(finish_reasons, (list, tuple)) and finish_reasons:
        candidate = finish_reasons[0]
        if isinstance(candidate, (str, int)) and not isinstance(candidate, bool):
            finish_reason = candidate
    usage: dict[str, object] = {}
    raw_usage = info.get("usage")
    if isinstance(raw_usage, dict):
        usage = {
            key: value
            for key, value in raw_usage.items()
            if isinstance(key, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    return request_id, finish_reason, usage


class CamelSkillTarget:
    """Execute a delivered Skill with a stateless CAMEL ChatAgent step."""

    runtime = "camel"

    def __init__(
        self,
        skill: str,
        config: RuntimeConfig,
        *,
        model_backend: Any | None = None,
        agent_factory: Any | None = None,
    ) -> None:
        ChatAgent, ModelFactory, ModelPlatformType = _load_camel()
        self.skill = skill
        self.config = config
        self.model = config.model
        self.system_message = PROBE_SYSTEM_TEMPLATE.format(skill=skill)
        self.agent_factory = agent_factory or ChatAgent
        if model_backend is None:
            initialization_failed = False
            try:
                model_backend = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                    model_type=config.model,
                    model_config_dict={
                        "temperature": 0,
                        "max_tokens": 2048,
                        "stream": False,
                    },
                    api_key=config.api_key,
                    url=config.base_url,
                    timeout=float(config.timeout_seconds),
                    max_retries=config.max_attempts - 1,
                )
            except Exception:
                initialization_failed = True
            if initialization_failed:
                raise RuntimeError(
                    f"CAMEL model initialization failed for {config.endpoint_origin}"
                )
        self.model_backend = model_backend

    def invoke(self, query: str, *, purpose: str) -> tuple[str, dict[str, object]]:
        response: Any | None = None
        request_failed = False
        try:
            agent = self.agent_factory(
                system_message=self.system_message,
                model=self.model_backend,
                summarize_threshold=None,
                max_iteration=1,
                retry_attempts=1,
                step_timeout=float(self.config.timeout_seconds),
            )
            response = agent.step(query)
        except Exception:
            request_failed = True
        if request_failed:
            raise RuntimeError(
                f"CAMEL model request failed for {self.config.endpoint_origin}"
            )
        if response is None:
            raise RuntimeError("CAMEL target returned no response")
        request_id, finish_reason, usage = _camel_response_metadata(response)
        return _camel_response_text(response), {
            "purpose": purpose,
            "runtime": self.runtime,
            "requested_model": self.model,
            "resolved_model": self.model,
            "base_url": self.config.base_url,
            "request_id": request_id,
            "finish_reason": finish_reason,
            "usage": usage,
        }


def create_probe_target(
    runtime: str,
    *,
    skill: str,
    config: RuntimeConfig,
    model: LanguageModel | None = None,
) -> ProbeTarget:
    if runtime == "direct":
        return DirectSkillTarget(skill, model or OpenAICompatibleModel(config))
    if runtime == "langchain":
        if model is not None:
            raise ValueError("model injection is supported only by the direct probe runtime")
        return LangChainSkillTarget(skill, config)
    if runtime == "camel":
        if model is not None:
            raise ValueError("model injection is supported only by the direct probe runtime")
        return CamelSkillTarget(skill, config)
    choices = ", ".join(SUPPORTED_PROBE_RUNTIMES)
    raise ValueError(f"unsupported probe runtime {runtime!r}; choose one of: {choices}")
