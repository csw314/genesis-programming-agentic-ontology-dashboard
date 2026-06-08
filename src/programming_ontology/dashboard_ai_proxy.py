"""Local AskSage proxy for dashboard insight requests."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import httpx

from . import paths
from ._dashboard_report import ASKSAGE_MULTIMODAL_MODELS
from .asksage_classifier import (
    DEFAULT_ASKSAGE_SERVER_BASE_URL,
    default_credentials_path,
    normalize_asksage_openai_base_url,
    resolve_asksage_openai_base_url,
)


DEFAULT_PROXY_PORT = 8765
MAX_REQUEST_BYTES = 5_000_000
MAX_CONTEXT_CHARS = 120_000
DEFAULT_MAX_TOKENS = 1600


@dataclass(frozen=True)
class AskSageDashboardCredentials:
    """Credentials submitted by the dashboard for one in-memory request."""

    email: str
    api_key: str


class AskSageProxyError(RuntimeError):
    """Sanitized API error suitable for showing in the dashboard UI."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def dashboard_model_registry() -> dict[str, dict[str, Any]]:
    """Return configured dashboard models keyed by model id."""

    registry = {
        str(model["id"]): {
            "id": str(model["id"]),
            "name": str(model.get("name") or model["id"]),
            "cost": str(model.get("cost") or "Cost not configured"),
            "supports_image": bool(model.get("supports_image", True)),
        }
        for model in ASKSAGE_MULTIMODAL_MODELS
    }
    registry["gpt-o3-mini"] = {
        "id": "gpt-o3-mini",
        "name": "GPT o3 Mini",
        "cost": "Medium cost",
        "supports_image": False,
    }
    return registry


def resolve_proxy_openai_base_url(base_url: str | None = None) -> str:
    """Resolve the AskSage OpenAI-compatible base URL for proxy calls."""

    explicit_openai_url = _first_present(
        base_url,
        os.environ.get("ASKSAGE_OPENAI_URL"),
        os.environ.get("ASKSAGE_OPENAI_BASE_URL"),
    )
    if explicit_openai_url:
        return normalize_asksage_openai_base_url(
            explicit_openai_url,
            explicit_openai_url=True,
        )

    credentials_path = default_credentials_path()
    if credentials_path.exists():
        try:
            return resolve_asksage_openai_base_url(credentials_path=credentials_path)
        except Exception:
            pass

    server_base_url = _first_present(
        os.environ.get("ASKSAGE_SERVER_BASE_URL"),
        os.environ.get("ASKSAGE_BASE_URL"),
        DEFAULT_ASKSAGE_SERVER_BASE_URL,
    )
    return normalize_asksage_openai_base_url(
        server_base_url,
        append_openai_path=True,
    )


def list_asksage_dashboard_models(
    payload: Mapping[str, Any],
    *,
    http_client: httpx.Client | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Validate credentials and return image-capable AskSage models."""

    credentials = _parse_credentials(payload)
    registry = dashboard_model_registry()
    close_client = http_client is None
    client = http_client or httpx.Client(timeout=30.0, trust_env=True)
    try:
        response = client.get(
            f"{resolve_proxy_openai_base_url(base_url).rstrip('/')}/models",
            headers=_asksage_headers(credentials),
        )
    except httpx.RequestError as error:
        raise AskSageProxyError(
            502,
            "network_failure",
            "Unable to reach AskSage. Check the network connection or proxy configuration.",
        ) from error
    finally:
        if close_client:
            client.close()

    _raise_for_asksage_status(response)
    available_models = _models_from_response(response.json(), registry)
    return {"models": available_models or list(registry.values())}


def request_asksage_dashboard_insight(
    payload: Mapping[str, Any],
    *,
    http_client: httpx.Client | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Send one dashboard insight request to AskSage."""

    credentials = _parse_credentials(payload)
    model = _clean_text(payload.get("model"))
    if not model:
        raise AskSageProxyError(400, "missing_model", "Select an AskSage model.")

    registry = dashboard_model_registry()
    configured_model = registry.get(model)
    image = _clean_text(payload.get("image"))
    if image and configured_model and configured_model.get("supports_image") is False:
        raise AskSageProxyError(
            400,
            "unsupported_image_model",
            "The selected AskSage model does not support image input.",
        )

    prompt = _clean_text(payload.get("prompt"))
    if not prompt:
        raise AskSageProxyError(400, "missing_prompt", "Enter a question or choose an insight action.")

    context_text = prepare_dashboard_context_text(payload.get("context", {}))
    user_content: str | list[dict[str, Any]]
    text_part = _insight_user_text(prompt=prompt, context_text=context_text)
    if image:
        user_content = [
            {"type": "text", "text": text_part},
            {"type": "image_url", "image_url": {"url": image, "detail": "low"}},
        ]
    else:
        user_content = text_part

    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _insight_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }

    close_client = http_client is None
    client = http_client or httpx.Client(timeout=90.0, trust_env=True)
    try:
        response = client.post(
            f"{resolve_proxy_openai_base_url(base_url).rstrip('/')}/chat/completions",
            headers=_asksage_headers(credentials),
            json=request_body,
        )
    except httpx.RequestError as error:
        raise AskSageProxyError(
            502,
            "network_failure",
            "Unable to reach AskSage. Check the network connection or proxy configuration.",
        ) from error
    finally:
        if close_client:
            client.close()

    _raise_for_asksage_status(response)
    return {"answer": _extract_chat_completion_text(response.json())}


def prepare_dashboard_context_text(
    context: Any,
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Serialize dashboard context and cap large raw data sections."""

    compacted = _compact_context(context)
    text = json.dumps(compacted, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    compacted = _compact_context(context, raw_row_limit=40, graph_limit=60)
    compacted["_context_truncated"] = True
    text = json.dumps(compacted, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 120] + "\n...[dashboard context truncated to fit AskSage request limits]"


def run_proxy(
    *,
    dashboard_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PROXY_PORT,
    base_url: str | None = None,
) -> ThreadingHTTPServer:
    """Start the local dashboard proxy server."""

    resolved_dashboard = Path(dashboard_path) if dashboard_path else paths.PUBLISH_OUTPUT_DIR / "genesis_dashboard_latest.html"

    class Handler(DashboardAIProxyHandler):
        dashboard_path = resolved_dashboard
        asksage_base_url = base_url

    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()
    return server


class DashboardAIProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler for the static dashboard and AskSage API routes."""

    dashboard_path: Path = paths.PUBLISH_OUTPUT_DIR / "genesis_dashboard_latest.html"
    asksage_base_url: str | None = None

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path in {"/", "/dashboard", "/dashboard.html"}:
            self._send_dashboard()
            return
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        self._send_json({"error": "Not found."}, status_code=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        try:
            payload = self._read_json()
            if self.path == "/api/asksage/models":
                result = list_asksage_dashboard_models(
                    payload,
                    base_url=self.asksage_base_url,
                )
            elif self.path == "/api/asksage/insights":
                result = request_asksage_dashboard_insight(
                    payload,
                    base_url=self.asksage_base_url,
                )
            else:
                self._send_json({"error": "Not found."}, status_code=404)
                return
            self._send_json(result)
        except AskSageProxyError as error:
            self._send_json(
                {"error": error.message, "code": error.code},
                status_code=error.status_code,
            )
        except json.JSONDecodeError:
            self._send_json(
                {"error": "The dashboard sent invalid JSON.", "code": "invalid_json"},
                status_code=400,
            )
        except Exception:
            self._send_json(
                {"error": "AskSage insight proxy failed.", "code": "proxy_failure"},
                status_code=500,
            )

    def log_message(self, format: str, *args: Any) -> None:
        """Disable default request logging to avoid accidental sensitive output."""

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > MAX_REQUEST_BYTES:
            raise AskSageProxyError(
                413,
                "context_limit",
                "The AskSage request is too large. Narrow the dashboard selection and try again.",
            )
        raw = self.rfile.read(length)
        loaded = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(loaded, dict):
            raise AskSageProxyError(400, "invalid_json", "The dashboard request must be a JSON object.")
        return loaded

    def _send_dashboard(self) -> None:
        if not self.dashboard_path.exists():
            self._send_json(
                {"error": "Dashboard HTML was not found.", "code": "missing_dashboard"},
                status_code=404,
            )
            return
        html = self.dashboard_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _send_json(self, payload: Mapping[str, Any], *, status_code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the local AskSage dashboard proxy."""

    parser = argparse.ArgumentParser(
        description="Serve the Genesis dashboard with a local AskSage insight proxy."
    )
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=paths.PUBLISH_OUTPUT_DIR / "genesis_dashboard_latest.html",
        help="Dashboard HTML file to serve. Defaults to outputs/publish/genesis_dashboard_latest.html.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PROXY_PORT)
    parser.add_argument(
        "--asksage-openai-url",
        help="Optional AskSage OpenAI-compatible base URL override.",
    )
    args = parser.parse_args(argv)
    print(f"Serving dashboard: {args.dashboard}")
    print(f"Open http://{args.host}:{args.port}/")
    run_proxy(
        dashboard_path=args.dashboard,
        host=args.host,
        port=args.port,
        base_url=args.asksage_openai_url,
    )
    return 0


def _parse_credentials(payload: Mapping[str, Any]) -> AskSageDashboardCredentials:
    credentials = payload.get("credentials")
    if not isinstance(credentials, Mapping):
        raise AskSageProxyError(400, "missing_credentials", "Enter AskSage credentials.")

    email = _clean_text(credentials.get("email") or credentials.get("username"))
    api_key = _clean_text(credentials.get("apiKey") or credentials.get("api_key"))
    if not email:
        raise AskSageProxyError(400, "missing_credentials", "Enter your AskSage email or username.")
    if not api_key:
        raise AskSageProxyError(400, "missing_credentials", "Enter your AskSage API key.")
    return AskSageDashboardCredentials(email=email, api_key=api_key)


def _asksage_headers(credentials: AskSageDashboardCredentials) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {credentials.api_key}",
        "x-access-tokens": credentials.api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _raise_for_asksage_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    message = _safe_response_message(response)
    lowered = message.casefold()
    if response.status_code in {401, 403}:
        raise AskSageProxyError(401, "invalid_credentials", "AskSage rejected the credentials.")
    if response.status_code in {400, 413} or any(
        marker in lowered for marker in ["context", "token", "too large", "maximum"]
    ):
        raise AskSageProxyError(
            413,
            "context_limit",
            "The AskSage request exceeds the model context limit. Narrow the dashboard selection and try again.",
        )
    raise AskSageProxyError(
        502,
        "api_failure",
        "AskSage returned an API error while generating insights.",
    )


def _models_from_response(
    response_payload: Mapping[str, Any],
    registry: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_models = response_payload.get("data") or response_payload.get("models") or []
    if not isinstance(raw_models, list):
        return []

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_model in raw_models:
        if isinstance(raw_model, str):
            model_id = raw_model
            raw_model = {"id": raw_model}
        elif isinstance(raw_model, Mapping):
            model_id = _clean_text(raw_model.get("id") or raw_model.get("model"))
        else:
            continue
        if not model_id or model_id in seen:
            continue
        registered = registry.get(model_id)
        supports_image = _model_supports_image(raw_model, registered)
        if not supports_image:
            continue
        output.append(
            {
                "id": model_id,
                "name": _clean_text(raw_model.get("name") if isinstance(raw_model, Mapping) else "")
                or (registered or {}).get("name")
                or model_id,
                "cost": _model_cost(raw_model, registered),
                "supports_image": True,
            }
        )
        seen.add(model_id)

    return sorted(output, key=lambda model: (model.get("cost") == "High cost", model["name"]))


def _model_supports_image(
    raw_model: Mapping[str, Any],
    registered: Mapping[str, Any] | None,
) -> bool:
    if registered is not None:
        return bool(registered.get("supports_image", True))
    capability_text = json.dumps(raw_model, ensure_ascii=False, default=str).casefold()
    return any(marker in capability_text for marker in ["image_input", '"image"', "vision", "multimodal"])


def _model_cost(
    raw_model: Mapping[str, Any],
    registered: Mapping[str, Any] | None,
) -> str:
    for key in ["cost", "cost_indicator", "pricing_tier", "tier"]:
        value = _clean_text(raw_model.get(key))
        if value:
            lowered = value.casefold()
            if "low" in lowered:
                return "Low cost"
            if "medium" in lowered or "standard" in lowered:
                return "Medium cost"
            if "high" in lowered or "premium" in lowered:
                return "High cost"
            return value
    if registered is not None:
        return str(registered.get("cost") or "Cost not configured")
    return "Cost not configured"


def _compact_context(
    context: Any,
    *,
    raw_row_limit: int = 120,
    graph_limit: int = 140,
) -> Any:
    compacted = deepcopy(context)
    if not isinstance(compacted, dict):
        return compacted

    raw_data = compacted.get("raw_data")
    compacted["raw_data"] = _limit_rows(raw_data, raw_row_limit)
    ontology = compacted.get("ontology")
    if isinstance(ontology, dict):
        if isinstance(ontology.get("nodes"), list):
            ontology["nodes"] = ontology["nodes"][:graph_limit]
        if isinstance(ontology.get("edges"), list):
            ontology["edges"] = ontology["edges"][:graph_limit]
    return compacted


def _limit_rows(value: Any, row_limit: int) -> Any:
    if isinstance(value, list):
        return value[:row_limit]
    if isinstance(value, dict):
        return {key: _limit_rows(item, row_limit) for key, item in value.items()}
    return value


def _insight_system_prompt() -> str:
    return (
        "You are an executive analytics assistant for the Genesis Mission Funding "
        "and Collaboration Dashboard. Use only the supplied dashboard context, "
        "ontology snippet, raw data, metadata, and visualization image. Be concise, "
        "cite concrete offices, challenges, rows, funding levels, and caveats when relevant."
    )


def _insight_user_text(*, prompt: str, context_text: str) -> str:
    return (
        "User request:\n"
        f"{prompt}\n\n"
        "Dashboard context JSON:\n"
        f"{context_text}\n\n"
        "Return readable Markdown. Use bullets or a compact table when that makes the answer easier to scan."
    )


def _extract_chat_completion_text(response_payload: Mapping[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AskSageProxyError(502, "api_failure", "AskSage returned no response text.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise AskSageProxyError(502, "api_failure", "AskSage returned no response text.")
    message = first.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                text = _clean_text(item.get("text") or item.get("content"))
                if text:
                    parts.append(text)
        content = "\n".join(parts)
    text = _clean_text(content)
    if not text:
        raise AskSageProxyError(502, "api_failure", "AskSage returned no response text.")
    return text


def _safe_response_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = response.text
    if isinstance(data, Mapping):
        for key in ["error", "message", "detail"]:
            value = data.get(key)
            if isinstance(value, str):
                return _redact_secret(value)
            if isinstance(value, Mapping):
                nested = value.get("message") or value.get("detail")
                if nested:
                    return _redact_secret(str(nested))
    return _redact_secret(str(data))


def _redact_secret(text: str) -> str:
    text = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", text)
    text = re.sub(r"x-access-tokens['\"]?\s*:\s*['\"]?[A-Za-z0-9._~+/=-]+", "x-access-tokens: [redacted]", text)
    return text


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_present(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
