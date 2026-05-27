"""Ask Sage client adapter for Genesis challenge classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import ssl
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from .llm_classifier_common import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_REASONING_EFFORT,
    SYSTEM_PROMPT,
    VALID_REASONING_EFFORTS,
    api_payload,
    decision_schema,
    get_value,
    load_audit_cache,
    request_payload_hash,
    response_id,
    validate_decision,
    write_audit_entry,
)


DEFAULT_ASKSAGE_MODEL = "gpt-5.1"
DEFAULT_ASKSAGE_TIMEOUT_SECONDS = DEFAULT_LLM_TIMEOUT_SECONDS
DEFAULT_ASKSAGE_OPENAI_USE_TRUSTSTORE = False
DEFAULT_ASKSAGE_USER_BASE_URL = "https://api.nnsa.asksage.ai/user/"
DEFAULT_ASKSAGE_SERVER_BASE_URL = "https://api.nnsa.asksage.ai/server/"
ASKSAGE_OPENAI_PATH = "openai/v1"
DEFAULT_CREDENTIALS_FILENAME = "credentials.json"
OPENAI_COMPATIBLE_ENDPOINT_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/responses",
)


@dataclass(frozen=True)
class AskSageCredentials:
    """Credential values needed to initialize the Ask Sage client."""

    api_key: str
    email: str
    user_base_url: str = DEFAULT_ASKSAGE_USER_BASE_URL
    server_base_url: str = DEFAULT_ASKSAGE_SERVER_BASE_URL
    openai_url: str | None = None
    model: str | None = None


def default_credentials_path() -> Path:
    """Return the default project-root credentials path."""

    return Path(__file__).resolve().parents[2] / DEFAULT_CREDENTIALS_FILENAME


def load_credentials(filename: str | Path | None = None) -> dict[str, Any]:
    """Load API credentials from a JSON file."""

    path = Path(filename) if filename is not None else default_credentials_path()
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Credentials file '{path}' not found.") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in credentials file '{path}'.") from error

    if not isinstance(loaded, dict):
        raise ValueError(f"Credentials file '{path}' must contain a JSON object.")
    return loaded


def parse_asksage_credentials(
    credentials_data: Mapping[str, Any],
    *,
    source: str = "credentials.json",
) -> AskSageCredentials:
    """Parse Ask Sage credentials from the project credentials JSON shape."""

    credentials = credentials_data.get("credentials")
    if not isinstance(credentials, Mapping):
        raise ValueError(f"{source} must contain a 'credentials' object.")

    api_key = str(credentials.get("api_key", "")).strip()
    user_info = credentials.get("Ask_sage_user_info", {})
    email = (
        str(user_info.get("username", "")).strip()
        if isinstance(user_info, Mapping)
        else ""
    )

    if not api_key:
        raise ValueError(f"{source} is missing credentials.api_key.")
    if not email:
        raise ValueError(
            f"{source} is missing credentials.Ask_sage_user_info.username."
        )

    user_base_url = str(
        credentials.get("user_base_url") or DEFAULT_ASKSAGE_USER_BASE_URL
    ).strip()
    server_base_url = str(
        credentials.get("server_base_url") or DEFAULT_ASKSAGE_SERVER_BASE_URL
    ).strip()
    model = str(credentials.get("model", "")).strip() or None
    openai_url = (
        str(credentials.get("openai_url") or credentials.get("openai_base_url") or "").strip()
        or None
    )

    return AskSageCredentials(
        api_key=api_key,
        email=email,
        user_base_url=user_base_url.rstrip("/"),
        server_base_url=server_base_url.rstrip("/"),
        openai_url=openai_url.rstrip("/") if openai_url else None,
        model=model,
    )


def build_asksage_client(
    *,
    api_key: str | None = None,
    email: str | None = None,
    base_url: str | None = None,
    user_base_url: str | None = None,
    server_base_url: str | None = None,
    openai_url: str | None = None,
    credentials_path: str | Path | None = None,
    path_to_ca_bundle: str | Path | None = None,
    client_class: Any | None = None,
    use_truststore: bool = True,
    timeout_seconds: float = DEFAULT_ASKSAGE_TIMEOUT_SECONDS,
) -> Any:
    """Build an ``asksageclient.AskSageClient`` from explicit or file credentials."""

    del use_truststore, timeout_seconds  # Kept for compatibility with prior callers.
    del openai_url  # Classic AskSageClient does not accept the OpenAI-compatible URL.
    credentials = _load_optional_asksage_credentials(credentials_path)

    resolved_api_key = _first_present(
        api_key,
        credentials.api_key if credentials else None,
        os.environ.get("ASKSAGE_API_KEY"),
    )
    resolved_email = _first_present(
        email,
        credentials.email if credentials else None,
        os.environ.get("ASKSAGE_EMAIL"),
        os.environ.get("ASKSAGE_USERNAME"),
    )
    resolved_user_base_url = _first_present(
        user_base_url,
        credentials.user_base_url if credentials else None,
        os.environ.get("ASKSAGE_USER_BASE_URL"),
        DEFAULT_ASKSAGE_USER_BASE_URL,
    )
    resolved_server_base_url = _first_present(
        server_base_url,
        base_url,
        credentials.server_base_url if credentials else None,
        os.environ.get("ASKSAGE_SERVER_BASE_URL"),
        os.environ.get("ASKSAGE_BASE_URL"),
        DEFAULT_ASKSAGE_SERVER_BASE_URL,
    )
    if not resolved_api_key:
        raise RuntimeError(
            "Ask Sage API key is required. Add credentials.api_key to "
            "credentials.json or pass api_key explicitly."
        )
    if not resolved_email:
        raise RuntimeError(
            "Ask Sage email is required. Add "
            "credentials.Ask_sage_user_info.username to credentials.json or "
            "pass email explicitly."
        )

    if client_class is None:
        try:
            from asksageclient import AskSageClient
        except ImportError as error:  # pragma: no cover - exercised via dependency tests.
            raise RuntimeError(
                "The asksageclient and pip-system-certs packages are required "
                "for the Ask Sage classifier."
            ) from error
        client_class = AskSageClient

    return client_class(
        email=resolved_email,
        api_key=resolved_api_key,
        user_base_url=str(resolved_user_base_url).rstrip("/"),
        server_base_url=str(resolved_server_base_url).rstrip("/"),
        path_to_CA_Bundle=str(path_to_ca_bundle) if path_to_ca_bundle else None,
    )


def build_asksage_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    server_base_url: str | None = None,
    openai_url: str | None = None,
    credentials_path: str | Path | None = None,
    use_truststore: bool = DEFAULT_ASKSAGE_OPENAI_USE_TRUSTSTORE,
    timeout_seconds: float = DEFAULT_ASKSAGE_TIMEOUT_SECONDS,
) -> Any:
    """Build an OpenAI SDK client configured for Ask Sage structured outputs."""

    credentials = _load_optional_asksage_credentials(credentials_path)
    resolved_api_key = _first_present(
        api_key,
        credentials.api_key if credentials else None,
        os.environ.get("ASKSAGE_API_KEY"),
    )

    if not resolved_api_key:
        raise RuntimeError(
            "Ask Sage API key is required. Add credentials.api_key to "
            "credentials.json or pass api_key explicitly."
        )

    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover - dependency is in project metadata.
        raise RuntimeError(
            "The openai package is required for Ask Sage structured outputs."
        ) from error

    client_kwargs = {
        "api_key": resolved_api_key,
        "base_url": _resolve_asksage_openai_base_url(
            credentials=credentials,
            base_url=base_url,
            server_base_url=server_base_url,
            openai_url=openai_url,
        ),
        "default_headers": {"x-access-tokens": resolved_api_key},
    }
    if not use_truststore:
        return OpenAI(timeout=timeout_seconds, **client_kwargs)

    try:
        import httpx
        import truststore
    except ImportError as error:  # pragma: no cover - dependencies are in metadata.
        raise RuntimeError(
            "httpx and truststore are required when use_truststore=True."
        ) from error

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    http_client = httpx.Client(
        verify=ssl_context,
        timeout=timeout_seconds,
        trust_env=True,
    )
    return OpenAI(http_client=http_client, **client_kwargs)


def resolve_asksage_openai_base_url(
    *,
    base_url: str | None = None,
    server_base_url: str | None = None,
    openai_url: str | None = None,
    credentials_path: str | Path | None = None,
) -> str:
    """Resolve the OpenAI-compatible Ask Sage base URL without API secrets."""

    return _resolve_asksage_openai_base_url(
        credentials=_load_optional_asksage_credentials(credentials_path),
        base_url=base_url,
        server_base_url=server_base_url,
        openai_url=openai_url,
    )


def normalize_asksage_openai_base_url(
    base_url: str,
    *,
    append_openai_path: bool = False,
    explicit_openai_url: bool = False,
) -> str:
    """Return a base URL suitable for the OpenAI SDK."""

    normalized = _strip_url_query_fragment(str(base_url).strip()).rstrip("/")
    if not normalized:
        raise ValueError("Ask Sage OpenAI-compatible base URL cannot be empty.")

    lowered = normalized.casefold()
    for suffix in OPENAI_COMPATIBLE_ENDPOINT_SUFFIXES:
        if lowered.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            lowered = normalized.casefold()
            break

    if lowered.endswith("/v1/models"):
        normalized = normalized[: -len("/models")]
        lowered = normalized.casefold()

    if lowered.endswith(f"/{ASKSAGE_OPENAI_PATH}"):
        return normalized
    if lowered.endswith("/v1"):
        return normalized
    if lowered.endswith("/openai"):
        return f"{normalized}/v1"
    if append_openai_path or lowered.endswith("/server"):
        return f"{normalized}/{ASKSAGE_OPENAI_PATH}"
    if explicit_openai_url:
        return f"{normalized}/v1"
    return f"{normalized}/{ASKSAGE_OPENAI_PATH}"


def resolve_asksage_model(
    model: str | None = None,
    *,
    credentials_path: str | Path | None = None,
) -> str:
    """Resolve the Ask Sage model from explicit input, env, credentials, or default."""

    if model and str(model).strip():
        return str(model).strip()

    env_model = os.environ.get("ASKSAGE_MODEL", "").strip()
    if env_model:
        return env_model

    credentials = _load_optional_asksage_credentials(credentials_path)
    if credentials and credentials.model:
        return credentials.model

    return DEFAULT_ASKSAGE_MODEL


class AskSageChallengeClassifier:
    """Callable classifier compatible with ``classify_programming_records``."""

    def __init__(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        base_url: str | None = None,
        openai_url: str | None = None, 
        api_key: str | None = None,
        email: str | None = None,
        credentials_path: str | Path | None = None,
        user_base_url: str | None = None,
        server_base_url: str | None = None,
        client: Any | None = None,
        audit_cache_path: str | Path | None = None,
        resume_cache: bool = False,
        use_truststore: bool = DEFAULT_ASKSAGE_OPENAI_USE_TRUSTSTORE,
        timeout_seconds: float = DEFAULT_ASKSAGE_TIMEOUT_SECONDS,
    ) -> None:
        if reasoning_effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                "reasoning_effort must be one of: "
                + ", ".join(sorted(VALID_REASONING_EFFORTS))
            )

        self.model = resolve_asksage_model(model, credentials_path=credentials_path)
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.api_key = api_key
        self.email = email
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self.user_base_url = user_base_url
        self.server_base_url = server_base_url
        self.openai_url = openai_url
        self.audit_cache_path = Path(audit_cache_path) if audit_cache_path else None
        self.resume_cache = resume_cache
        self.use_truststore = use_truststore
        self.timeout_seconds = timeout_seconds
        self.client = client if client is not None else self._build_client()
        self._cache = self._load_cache() if resume_cache and self.audit_cache_path else {}

    def __call__(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        """Return a structured classification decision for one payload."""

        request_hash = request_payload_hash(payload, model=self.model)
        cached = self._cache.get(request_hash)
        if cached is not None:
            return cached["decision"]

        candidate_names = [
            str(candidate["challenge_name"])
            for candidate in payload.get("candidate_challenges", [])
        ]
        if not candidate_names:
            raise ValueError("Ask Sage classifier received no candidate challenges.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _query_system_prompt()},
                {"role": "user", "content": _query_message(payload)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "genesis_challenge_classification",
                    "strict": True,
                    "schema": decision_schema(candidate_names),
                },
            },
            temperature=0,
        )
        try:
            decision = parse_asksage_chat_completion_decision(response)
            validate_decision(decision, candidate_names, provider_name="Ask Sage")
        except Exception as error:
            self._write_failure_audit_entry(
                payload,
                response,
                request_hash,
                error,
                candidate_names,
            )
            raise
        self._write_audit_entry(payload, response, request_hash, decision, candidate_names)
        return decision

    def _build_client(self) -> Any:
        return build_asksage_openai_client(
            api_key=self.api_key,
            base_url=self.base_url,
            openai_url=self.openai_url,
            credentials_path=self.credentials_path,
            server_base_url=self.server_base_url,
            use_truststore=self.use_truststore,
            timeout_seconds=self.timeout_seconds,
        )

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self.audit_cache_path is None or not self.audit_cache_path.exists():
            return {}
        return load_audit_cache(self.audit_cache_path)

    def _write_audit_entry(
        self,
        payload: dict[str, Any],
        response: Any,
        request_hash: str,
        decision: Mapping[str, Any],
        candidate_names: list[str],
    ) -> None:
        write_audit_entry(
            audit_cache_path=self.audit_cache_path,
            cache=self._cache,
            payload=payload,
            response=response,
            request_hash=request_hash,
            decision=decision,
            candidate_names=candidate_names,
            model=self.model,
        )

    def _write_failure_audit_entry(
        self,
        payload: dict[str, Any],
        response: Any,
        request_hash: str,
        error: Exception,
        candidate_names: list[str],
    ) -> None:
        if self.audit_cache_path is None:
            return

        evidence = payload.get("programming_evidence", {})
        entry = {
            "request_hash": request_hash,
            "model": self.model,
            "response_id": response_id(response),
            "programming_record_id": evidence.get("programming_record_id", ""),
            "source_row_number": evidence.get("source_row_number", ""),
            "wbs_name": evidence.get("wbs_name", ""),
            "candidate_count": len(candidate_names),
            "candidate_challenges": candidate_names,
            "error_type": type(error).__name__,
            "error": str(error),
            "response_preview": response_preview(response),
        }
        self.audit_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def parse_asksage_chat_completion_decision(response: Any) -> dict[str, Any]:
    """Parse an Ask Sage OpenAI-compatible Chat Completions response."""

    content = _extract_chat_completion_content(response)
    try:
        parsed = _loads_jsonish(content)
    except ValueError as error:
        raise ValueError(
            "Ask Sage response did not contain valid JSON. "
            f"Response preview: {response_preview(response)}"
        ) from error

    direct = _find_decision_mapping(parsed)
    if direct is not None:
        return direct
    raise ValueError(
        "Ask Sage response JSON must include predicted_challenge, confidence, "
        f"and rationale. Response preview: {response_preview(response)}"
    )


def parse_asksage_query_decision(response: Any) -> dict[str, Any]:
    """Parse the Ask Sage query response into a classifier decision object."""

    direct = _find_decision_mapping(response)
    if direct is not None:
        return direct

    text = _extract_query_text(response)
    parsed = _loads_jsonish(text)
    nested = _find_decision_mapping(parsed)
    if nested is not None:
        return nested
    if isinstance(parsed, Mapping):
        nested_text = _extract_query_text(parsed, required=False)
        if nested_text:
            nested = _find_decision_mapping(_loads_jsonish(nested_text))
            if nested is not None:
                return nested

    raise ValueError("Ask Sage response JSON must include a decision object.")


def _load_optional_asksage_credentials(
    credentials_path: str | Path | None = None,
) -> AskSageCredentials | None:
    path = Path(credentials_path) if credentials_path else default_credentials_path()
    if not path.exists():
        return None
    return parse_asksage_credentials(load_credentials(path), source=str(path))


def _resolve_asksage_openai_base_url(
    *,
    credentials: AskSageCredentials | None = None,
    base_url: str | None = None,
    server_base_url: str | None = None,
    openai_url: str | None = None,
) -> str:
    resolved_openai_url = _first_present(
        openai_url,
        os.environ.get("ASKSAGE_OPENAI_URL"),
        os.environ.get("ASKSAGE_OPENAI_BASE_URL"),
        credentials.openai_url if credentials else None,
    )
    if resolved_openai_url:
        return normalize_asksage_openai_base_url(
            resolved_openai_url,
            explicit_openai_url=True,
        )

    resolved_server_base_url = _first_present(
        base_url,
        server_base_url,
        os.environ.get("ASKSAGE_SERVER_BASE_URL"),
        os.environ.get("ASKSAGE_BASE_URL"),
        credentials.server_base_url if credentials else None,
        DEFAULT_ASKSAGE_SERVER_BASE_URL,
    )
    return normalize_asksage_openai_base_url(
        resolved_server_base_url,
        append_openai_path=True,
    )


def _strip_url_query_fragment(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _query_system_prompt() -> str:
    return (
        SYSTEM_PROMPT
        + "\nReturn only the JSON object required by the supplied schema."
    )


def _query_message(payload: Mapping[str, Any]) -> str:
    return json.dumps(api_payload(payload), ensure_ascii=False, sort_keys=True)


def _query_reasoning_effort(reasoning_effort: str) -> str:
    if reasoning_effort == "xhigh":
        return "high"
    return reasoning_effort


def _find_decision_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        if {"predicted_challenge", "confidence", "rationale"}.issubset(value):
            return dict(value)
        for key in ("response", "message", "answer", "content", "text", "output", "data"):
            nested = value.get(key)
            found = _find_decision_mapping(nested)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_decision_mapping(item)
            if found is not None:
                return found
    return None


def _extract_chat_completion_content(response: Any) -> str:
    choices = get_value(response, "choices") or []
    if not choices:
        raise ValueError(
            "Ask Sage response did not include choices. "
            f"Response preview: {response_preview(response)}"
        )

    choice = choices[0]
    message = get_value(choice, "message")
    content = get_value(message, "content") if message is not None else None
    if isinstance(content, list):
        text_parts = []
        for item in content:
            text = get_value(item, "text")
            if text:
                text_parts.append(str(text))
        content = "\n".join(text_parts)

    if not content:
        raise ValueError(
            "Ask Sage response did not include message content. "
            f"Response preview: {response_preview(response)}"
        )
    return str(content)


def _extract_query_text(value: Any, *, required: bool = True) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("response", "message", "answer", "content", "text", "output"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested
            if isinstance(nested, Mapping) or isinstance(nested, list):
                try:
                    return _extract_query_text(nested, required=True)
                except ValueError:
                    continue
        if required:
            return json.dumps(value, ensure_ascii=False)
        return ""
    if isinstance(value, list):
        for item in value:
            try:
                return _extract_query_text(item, required=True)
            except ValueError:
                continue
    if required:
        raise ValueError("Ask Sage response did not include output text.")
    return ""


def _loads_jsonish(text: str) -> Any:
    stripped = _strip_markdown_json(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Ask Sage response did not contain valid JSON.")


def _strip_markdown_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def response_preview(response: Any, *, max_length: int = 500) -> str:
    """Return a short, non-secret preview of a provider response."""

    try:
        preview = _extract_chat_completion_content(response)
    except ValueError:
        preview = _safe_json_preview(response)

    preview = re.sub(r"\s+", " ", str(preview)).strip()
    if len(preview) > max_length:
        return preview[: max_length - 3] + "..."
    return preview


def _safe_json_preview(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _first_present(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""
