from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
import os
from urllib.parse import urlparse

from .crypto import validate_owner_key


PROTOCOL = "skillcoder-core/2"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MIN_ACTIVE_RATE = 0.60
MAX_DECOY_ACTIVATION_RATE = 0.20
MAX_NORMAL_ACTIVATION_RATE = 0.10
MIN_NORMAL_QUERY_COUNT = 10
MAX_NORMAL_QUERY_COUNT = 100
MAX_QUERY_CHARACTERS = 4_000
MAX_TOTAL_QUERY_CHARACTERS = 100_000
MIN_PROBE_PAIRS = 5
MAX_PROBE_PAIRS = 100
MAX_PROBE_JOBS = 300


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str
    owner_key: str
    model: str
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: int = 180
    max_attempts: int = 3
    allow_insecure_local_http: bool = False

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key is required")
        if not self.model.strip():
            raise ValueError("model is required")
        validate_owner_key(self.owner_key)
        parsed = urlparse(self.base_url)
        if not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an absolute URL without query or fragment")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain username or password")
        if parsed.scheme == "http":
            host = parsed.hostname or ""
            try:
                loopback = host == "localhost" or ip_address(host).is_loopback
            except ValueError:
                loopback = host == "localhost"
            if not self.allow_insecure_local_http or not loopback:
                raise ValueError(
                    "HTTP base_url is allowed only for loopback with "
                    "SKILLCODER_ALLOW_INSECURE_LOCAL_HTTP=1"
                )
        elif parsed.scheme != "https":
            raise ValueError("base_url must use HTTPS")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> "RuntimeConfig":
        api_key = os.getenv("SKILLCODER_MODEL_API_KEY", "").strip()
        configured_base_url = os.getenv("SKILLCODER_MODEL_BASE_URL", "").strip()
        owner_key = os.getenv("SKILLCODER_OWNER_KEY", "").strip()
        try:
            validate_owner_key(owner_key)
        except ValueError as exc:
            raise RuntimeError(
                "SKILLCODER_OWNER_KEY must contain at least 32 UTF-8 bytes"
            ) from exc
        selected_model = (
            model if model is not None else (os.getenv("SKILLCODER_MODEL") or "")
        ).strip()
        if not selected_model:
            raise RuntimeError("--model or SKILLCODER_MODEL is required")
        if not api_key:
            raise RuntimeError("SKILLCODER_MODEL_API_KEY is required")
        selected_base_url = base_url or configured_base_url or DEFAULT_BASE_URL
        return cls(
            api_key=api_key,
            owner_key=owner_key,
            model=selected_model,
            base_url=selected_base_url,
            allow_insecure_local_http=os.getenv(
                "SKILLCODER_ALLOW_INSECURE_LOCAL_HTTP", ""
            ).strip() == "1",
        )

    @property
    def endpoint_origin(self) -> str:
        parsed = urlparse(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}"
