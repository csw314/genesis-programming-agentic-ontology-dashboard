"""Shared helpers for LLM-backed Genesis challenge classifiers."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
from typing import Any, Mapping


DEFAULT_REASONING_EFFORT = "medium"
VALID_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0

SYSTEM_PROMPT = """Classify one programming funding row against Genesis AI Mission Challenges.

Use the Scope Description as the primary evidence. Use WBS, STAT hierarchy,
program request, acquisition context, funding office, and dollars only as
supporting evidence. Do not assign, infer, or change the funding office. Choose
exactly one predicted_challenge from the supplied candidate_challenges. Return
only the structured fields requested by the response schema.
"""


def request_payload_hash(payload: Mapping[str, Any], *, model: str) -> str:
    """Return a stable hash for the model input without storing the prompt."""

    normalized = json.dumps(
        {"model": model, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(normalized.encode("utf-8")).hexdigest()


def api_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Trim the pipeline payload down to the fields sent to an LLM provider."""

    evidence = dict(payload.get("programming_evidence", {}))
    candidates = []
    for candidate in payload.get("candidate_challenges", []):
        candidate_data = dict(candidate)
        context = dict(candidate_data.get("context", {}))
        candidates.append(
            {
                "challenge_name": candidate_data.get("challenge_name", ""),
                "score": candidate_data.get("score", 0.0),
                "matched_terms": candidate_data.get("matched_terms", []),
                "has_mstc_context": candidate_data.get("has_mstc_context", False),
                "context": {
                    "challenge_name": context.get("challenge_name", ""),
                    "problem_statements": context.get("problem_statements", []),
                    "proposed_solutions": context.get("proposed_solutions", []),
                    "nnsa_advantages": context.get("nnsa_advantages", []),
                    "expected_outcomes": context.get("expected_outcomes", []),
                },
            }
        )

    return {
        "task": payload.get("instructions", ""),
        "programming_evidence": evidence,
        "candidate_challenges": candidates,
    }


def decision_schema(candidate_names: list[str]) -> dict[str, Any]:
    """Return the JSON schema all LLM adapters must satisfy."""

    return {
        "type": "object",
        "properties": {
            "predicted_challenge": {
                "type": "string",
                "enum": candidate_names,
                "description": "Exactly one challenge name from candidate_challenges.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence in the mapping based on supplied evidence.",
            },
            "rationale": {
                "type": "string",
                "description": "Brief evidence-based rationale for the selected challenge.",
            },
        },
        "required": ["predicted_challenge", "confidence", "rationale"],
        "additionalProperties": False,
    }


def parse_response_decision(response: Any, *, provider_name: str) -> dict[str, Any]:
    """Parse a Responses API object into a decision mapping."""

    text = extract_response_text(response, provider_name=provider_name)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{provider_name} response did not contain valid JSON.") from error

    if not isinstance(parsed, dict):
        raise ValueError(f"{provider_name} response JSON must be an object.")
    return parsed


def extract_response_text(response: Any, *, provider_name: str) -> str:
    """Extract output text from dict-like or SDK object Responses API results."""

    output_text = get_value(response, "output_text")
    if output_text:
        return str(output_text)

    output = get_value(response, "output") or []
    for item in output:
        content = get_value(item, "content") or []
        for content_item in content:
            text = get_value(content_item, "text")
            if text:
                return str(text)

    raise ValueError(f"{provider_name} response did not include output text.")


def validate_decision(
    decision: Mapping[str, Any],
    candidate_names: list[str],
    *,
    provider_name: str,
) -> None:
    """Validate a provider decision before it enters the classification pipeline."""

    missing = {"predicted_challenge", "confidence", "rationale"} - set(decision)
    if missing:
        raise ValueError(
            f"{provider_name} decision is missing fields: "
            + ", ".join(sorted(missing))
        )
    if decision["predicted_challenge"] not in candidate_names:
        raise ValueError(
            f"{provider_name} decision selected a challenge outside the candidate list."
        )
    if str(decision["confidence"]).casefold() not in {"high", "medium", "low"}:
        raise ValueError(
            f"{provider_name} decision confidence must be high, medium, or low."
        )


def load_audit_cache(audit_cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load cached LLM decisions from the provider audit JSONL file."""

    cache: dict[str, dict[str, Any]] = {}
    if not audit_cache_path.exists():
        return cache

    with audit_cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            decision = {
                "predicted_challenge": entry.get("predicted_challenge", ""),
                "confidence": entry.get("confidence", "low"),
                "rationale": entry.get("rationale", ""),
            }
            cache[str(entry["request_hash"])] = {"decision": decision}
    return cache


def write_audit_entry(
    *,
    audit_cache_path: Path | None,
    cache: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    response: Any,
    request_hash: str,
    decision: Mapping[str, Any],
    candidate_names: list[str],
    model: str,
) -> None:
    """Append a provider-neutral audit/cache entry for one LLM decision."""

    if audit_cache_path is None:
        return

    evidence = payload.get("programming_evidence", {})
    entry = {
        "request_hash": request_hash,
        "model": model,
        "response_id": response_id(response),
        "programming_record_id": evidence.get("programming_record_id", ""),
        "source_row_number": evidence.get("source_row_number", ""),
        "wbs_name": evidence.get("wbs_name", ""),
        "candidate_count": len(candidate_names),
        "candidate_challenges": candidate_names,
        "predicted_challenge": decision.get("predicted_challenge", ""),
        "confidence": decision.get("confidence", ""),
        "rationale": decision.get("rationale", ""),
    }
    audit_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    cache[request_hash] = {
        "decision": {
            "predicted_challenge": entry["predicted_challenge"],
            "confidence": entry["confidence"],
            "rationale": entry["rationale"],
        }
    }


def response_id(response: Any) -> str:
    """Return the provider response ID when available."""

    value = get_value(response, "id")
    return str(value) if value is not None else ""


def get_value(value: Any, key: str) -> Any:
    """Read a key from either a mapping or SDK object."""

    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)
