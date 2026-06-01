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

AI_DIRECTNESS_SYSTEM_PROMPT = """Assess how directly one programming funding row supports an AI/ML workload.

Use the Scope Description as the primary evidence. Use WBS, STAT hierarchy,
program request, acquisition context, funding office, dollars, and the already
selected Genesis Mission Challenge only as supporting context. Be thoughtful and
strict: do not let the words Genesis, modernization, platform, data, or compute
inflate the score unless the row actually describes an AI/ML workload or an
asset primarily meant to enable one.

Use the full 0-100 range and differentiate rows whenever the evidence differs.
Avoid defaulting many rows to the same midpoint or band boundary. Prefer scores
with meaningful separation for named tools, concrete products, infrastructure,
readiness work, and generic program labels.

Score rubric:
- 90-100: Direct AI/ML model, AI software, AI productization,
  autonomous/decision workflow, or AI-enabled operational product. Reserve the
  top of this range for named AI tools/models or concrete deployment work.
- 70-89: AI-enabled workflow, applied AI integration, or tooling that clearly
  embeds AI/ML in the work product. Generic "AI program work" usually belongs
  here or lower unless the scope explains the work product.
- 50-69: Compute, GPU, cloud, storage, model-hosting, or infrastructure
  investment that primarily supports AI but could be reused for non-AI work.
- 30-49: Data collection, curation, data platform, data engineering, or
  analytics foundation that could support AI but is not itself AI/ML.
- 10-29: General digital modernization, cyber, infrastructure, or
  instrumentation with only a weak AI/ML link.
- 0-9: No clear AI/ML workload.

Examples:
- Named AI tools, models, digital twins, prototype productization, or AI-driven
  operational workflows: high 80s to high 90s depending on specificity.
- Generic "Genesis AI" or "AI program work" with little scope detail: mid 50s
  to low 70s, not automatically direct AI/ML.
- AI readiness, workforce training, governance, collaboration, or portfolio
  enablement: 50s to 70s depending on operational linkage.
- GPUs, cloud, secure storage, model hosting, or compute facilities: 45-70
  depending on whether AI is the primary purpose or only a possible use.
- Data conditioning, curation, labeling, data management, or archiving:
  30-55 unless tied directly to model development/deployment.
- General cyber, modernization, networks, facilities, or instrumentation:
  10-35 unless the scope clearly says the work delivers or enables AI/ML.

Return only the structured fields requested by the response schema.
"""

AI_DIRECTNESS_LABELS = [
    "direct_ai_ml",
    "ai_enabled_workflow",
    "ai_infrastructure",
    "data_foundation",
    "weak_ai_link",
    "not_ai_ml",
]


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


def ai_directness_api_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Trim an AI-directness payload down to the fields sent to an LLM."""

    evidence = dict(payload.get("programming_evidence", {}))
    return {
        "task": payload.get("instructions", ""),
        "predicted_challenge": payload.get("predicted_challenge", ""),
        "programming_evidence": {
            "programming_record_id": evidence.get("programming_record_id", ""),
            "source_row_number": evidence.get("source_row_number", ""),
            "wbs_name": evidence.get("wbs_name", ""),
            "wbs_code": evidence.get("wbs_code", ""),
            "source_sub_office": evidence.get("source_sub_office", ""),
            "normalized_funding_office": evidence.get("normalized_funding_office", ""),
            "stat_l3": evidence.get("stat_l3", ""),
            "stat_l4": evidence.get("stat_l4", ""),
            "stat_l5": evidence.get("stat_l5", ""),
            "funding_level": evidence.get("funding_level", ""),
            "scope_description": evidence.get("scope_description", ""),
            "program_interest_area": evidence.get("program_interest_area", ""),
            "process_improvement_area": evidence.get("process_improvement_area", ""),
            "program_request": evidence.get("program_request", ""),
            "acquisition_context": evidence.get("acquisition_context", []),
            "five_year_total": evidence.get("five_year_total", 0.0),
            "additional_context": evidence.get("additional_context", []),
        },
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


def ai_directness_schema() -> dict[str, Any]:
    """Return the JSON schema for AI-directness assessments."""

    return {
        "type": "object",
        "properties": {
            "ai_directness_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "How directly the row supports an AI/ML workload.",
            },
            "ai_directness_label": {
                "type": "string",
                "enum": AI_DIRECTNESS_LABELS,
                "description": "Rubric label that best explains the score band.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence in the AI-directness assessment.",
            },
            "rationale": {
                "type": "string",
                "description": "Brief evidence-based rationale for the score.",
            },
        },
        "required": [
            "ai_directness_score",
            "ai_directness_label",
            "confidence",
            "rationale",
        ],
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


def validate_ai_directness_decision(
    decision: Mapping[str, Any],
    *,
    provider_name: str,
) -> None:
    """Validate a provider AI-directness decision before pipeline use."""

    missing = {
        "ai_directness_score",
        "ai_directness_label",
        "confidence",
        "rationale",
    } - set(decision)
    if missing:
        raise ValueError(
            f"{provider_name} AI-directness decision is missing fields: "
            + ", ".join(sorted(missing))
        )

    try:
        score = int(decision["ai_directness_score"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{provider_name} AI-directness score must be an integer."
        ) from error
    if score < 0 or score > 100:
        raise ValueError(
            f"{provider_name} AI-directness score must be between 0 and 100."
        )
    if str(decision["ai_directness_label"]) not in AI_DIRECTNESS_LABELS:
        raise ValueError(
            f"{provider_name} AI-directness label must be one of: "
            + ", ".join(AI_DIRECTNESS_LABELS)
        )
    if str(decision["confidence"]).casefold() not in {"high", "medium", "low"}:
        raise ValueError(
            f"{provider_name} AI-directness confidence must be high, medium, or low."
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
            if "error_type" in entry:
                continue
            decision = {
                "predicted_challenge": entry.get("predicted_challenge", ""),
                "confidence": entry.get("confidence", "low"),
                "rationale": entry.get("rationale", ""),
            }
            cache[str(entry["request_hash"])] = {"decision": decision}
    return cache


def load_ai_directness_audit_cache(audit_cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load cached AI-directness decisions from a provider audit JSONL file."""

    cache: dict[str, dict[str, Any]] = {}
    if not audit_cache_path.exists():
        return cache

    with audit_cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if "error_type" in entry:
                continue
            decision = {
                "ai_directness_score": entry.get("ai_directness_score", 0),
                "ai_directness_label": entry.get("ai_directness_label", "not_ai_ml"),
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


def write_ai_directness_audit_entry(
    *,
    audit_cache_path: Path | None,
    cache: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    response: Any,
    request_hash: str,
    decision: Mapping[str, Any],
    model: str,
) -> None:
    """Append a provider-neutral audit/cache entry for one AI-directness decision."""

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
        "predicted_challenge": payload.get("predicted_challenge", ""),
        "ai_directness_score": int(decision.get("ai_directness_score", 0)),
        "ai_directness_label": decision.get("ai_directness_label", ""),
        "confidence": decision.get("confidence", ""),
        "rationale": decision.get("rationale", ""),
    }
    audit_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    cache[request_hash] = {
        "decision": {
            "ai_directness_score": entry["ai_directness_score"],
            "ai_directness_label": entry["ai_directness_label"],
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
