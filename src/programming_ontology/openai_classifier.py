"""OpenAI Responses API adapter for Genesis challenge classification."""

from __future__ import annotations

from pathlib import Path
import json
import os
import ssl
from typing import Any, Mapping

from .llm_classifier_common import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_REASONING_EFFORT,
    SYSTEM_PROMPT,
    VALID_REASONING_EFFORTS,
    api_payload,
    decision_schema,
    get_value,
    load_audit_cache,
    parse_response_decision,
    request_payload_hash,
    response_id,
    validate_decision,
    write_audit_entry,
)


DEFAULT_OPENAI_MODEL = "gpt-5.5-pro"
DEFAULT_OPENAI_TIMEOUT_SECONDS = DEFAULT_LLM_TIMEOUT_SECONDS


def build_openai_client(
    *,
    use_truststore: bool = True,
    timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
) -> Any:
    """Build an OpenAI client suitable for locked-down Windows environments."""

    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover - dependency is in project metadata.
        raise RuntimeError(
            "The openai package is required for the OpenAI classifier."
        ) from error

    if not use_truststore:
        return OpenAI(timeout=timeout_seconds)

    try:
        import httpx
        import truststore
    except ImportError as error:  # pragma: no cover - dependencies are in project metadata.
        raise RuntimeError(
            "httpx and truststore are required when use_truststore=True."
        ) from error

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    http_client = httpx.Client(
        verify=ssl_context,
        timeout=timeout_seconds,
        trust_env=True,
    )
    return OpenAI(http_client=http_client)


class OpenAIChallengeClassifier:
    """Callable classifier compatible with ``classify_programming_records``."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        client: Any | None = None,
        audit_cache_path: str | Path | None = None,
        resume_cache: bool = False,
        use_truststore: bool = True,
        timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
    ) -> None:
        if reasoning_effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                "reasoning_effort must be one of: "
                + ", ".join(sorted(VALID_REASONING_EFFORTS))
            )

        self.model = model
        self.reasoning_effort = reasoning_effort
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
            raise ValueError("OpenAI classifier received no candidate challenges.")

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(api_payload(payload), ensure_ascii=False),
                },
            ],
            reasoning={"effort": self.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "genesis_challenge_classification",
                    "strict": True,
                    "schema": decision_schema(candidate_names),
                }
            },
            store=False,
        )
        decision = parse_response_decision(response, provider_name="OpenAI")
        validate_decision(decision, candidate_names, provider_name="OpenAI")
        self._write_audit_entry(payload, response, request_hash, decision, candidate_names)
        return decision

    def _build_client(self) -> Any:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required when using the OpenAI classifier."
            )

        return build_openai_client(
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


_api_payload = api_payload
_decision_schema = decision_schema
_response_id = response_id
_get_value = get_value


def _parse_response_decision(response: Any) -> dict[str, Any]:
    return parse_response_decision(response, provider_name="OpenAI")


def _extract_response_text(response: Any) -> str:
    from .llm_classifier_common import extract_response_text

    return extract_response_text(response, provider_name="OpenAI")


def _validate_decision(decision: Mapping[str, Any], candidate_names: list[str]) -> None:
    validate_decision(decision, candidate_names, provider_name="OpenAI")
