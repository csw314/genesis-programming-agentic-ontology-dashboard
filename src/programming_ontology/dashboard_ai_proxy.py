"""Local AskSage proxy for dashboard insight requests."""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
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
MAX_REQUEST_BYTES = 25_000_000
MAX_CONTEXT_CHARS = 120_000
DEFAULT_MAX_TOKENS = 1600
DEFAULT_REPORT_MAX_TOKENS = 3200
REPORT_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MAX_REPORT_IMAGES_FOR_MODEL = 4


@dataclass(frozen=True)
class AskSageDashboardCredentials:
    """Credentials submitted by the dashboard for one in-memory request."""

    email: str
    api_key: str


@dataclass(frozen=True)
class GeneratedReport:
    """Generated dashboard report response."""

    filename: str
    content_type: str
    body: bytes


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


def request_asksage_dashboard_report(
    payload: Mapping[str, Any],
    *,
    http_client: httpx.Client | None = None,
    base_url: str | None = None,
) -> GeneratedReport:
    """Generate a Word executive report from dashboard context."""

    credentials = _parse_credentials(payload)
    model = _clean_text(payload.get("model"))
    if not model:
        raise AskSageProxyError(400, "missing_model", "Select an AskSage model.")

    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise AskSageProxyError(
            400,
            "missing_dashboard_context",
            "Dashboard data is missing or unavailable.",
        )

    registry = dashboard_model_registry()
    configured_model = registry.get(model)
    target_images = _target_images_from_payload(targets)
    if target_images and configured_model and configured_model.get("supports_image") is False:
        raise AskSageProxyError(
            400,
            "unsupported_image_model",
            "The selected AskSage model does not support image input.",
        )

    report_context = prepare_dashboard_report_context(payload)
    context_text = prepare_dashboard_report_context_text(report_context)
    user_content: str | list[dict[str, Any]]
    text_part = _report_user_text(context_text)
    model_images = list(target_images.values())[:MAX_REPORT_IMAGES_FOR_MODEL]
    if model_images:
        user_content = [{"type": "text", "text": text_part}]
        user_content.extend(
            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}
            for image_url in model_images
        )
    else:
        user_content = text_part

    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _report_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": DEFAULT_REPORT_MAX_TOKENS,
    }

    close_client = http_client is None
    client = http_client or httpx.Client(timeout=120.0, trust_env=True)
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
    narrative = _extract_chat_completion_text(response.json())
    try:
        body = build_dashboard_report_docx(
            report_context,
            narrative,
            target_images=target_images,
        )
    except Exception as error:
        raise AskSageProxyError(
            500,
            "document_generation_failure",
            "Word document generation failed.",
        ) from error

    generated_at = _clean_text(report_context.get("generated_at")) or datetime.now().isoformat(timespec="seconds")
    date_part = generated_at[:10] if len(generated_at) >= 10 else datetime.now().strftime("%Y-%m-%d")
    return GeneratedReport(
        filename=f"dashboard_executive_summary_{date_part}.docx",
        content_type=REPORT_DOCX_MIME,
        body=body,
    )


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


def prepare_dashboard_report_context(
    payload: Mapping[str, Any],
    *,
    raw_sample_limit: int = 8,
    graph_limit: int = 60,
) -> dict[str, Any]:
    """Build compact report context without credentials or image data."""

    dashboard = payload.get("dashboard") if isinstance(payload.get("dashboard"), Mapping) else {}
    generated_at = _clean_text(dashboard.get("generated_at_client")) or datetime.now().isoformat(timespec="seconds")
    targets = []
    for target in payload.get("targets", []):
        if not isinstance(target, Mapping):
            continue
        context = target.get("context") if isinstance(target.get("context"), Mapping) else {}
        target_meta = context.get("target") if isinstance(context.get("target"), Mapping) else {}
        metadata = context.get("metadata") if isinstance(context.get("metadata"), Mapping) else {}
        ontology = context.get("ontology") if isinstance(context.get("ontology"), Mapping) else {}
        raw_data = context.get("raw_data")
        target_id = _clean_text(target_meta.get("id")) or f"target-{len(targets) + 1}"
        targets.append(
            {
                "id": target_id,
                "title": _clean_text(target_meta.get("title")) or "Dashboard element",
                "type": _clean_text(target_meta.get("type")),
                "fynsp_view": _clean_text(target_meta.get("fynsp_view")),
                "metadata": metadata,
                "current_dashboard_selections": context.get("current_dashboard_selections", {}),
                "raw_data_summary": _summarize_raw_data(raw_data, sample_limit=raw_sample_limit),
                "ontology": {
                    "summary": ontology.get("summary", {}),
                    "selection": ontology.get("selection", {}),
                    "nodes": (ontology.get("nodes") or [])[:graph_limit]
                    if isinstance(ontology.get("nodes"), list)
                    else [],
                    "edges": (ontology.get("edges") or [])[:graph_limit]
                    if isinstance(ontology.get("edges"), list)
                    else [],
                },
                "has_image": bool(_clean_text(target.get("image"))),
                "capture_error": _clean_text(target.get("capture_error")),
                "session_insight": _clean_text(target.get("session_insight")),
                "text_snapshot": _clean_text(target.get("text_snapshot"))[:4000],
            }
        )

    return {
        "report_type": "dashboard_executive_summary",
        "generated_at": generated_at,
        "dashboard": {
            "title": _clean_text(dashboard.get("title")) or "Genesis Mission Funding and Collaboration Dashboard",
            "metadata": _clean_text(dashboard.get("metadata")),
            "current_view": _clean_text(dashboard.get("current_view")),
            "assumptions": dashboard.get("assumptions", []),
        },
        "target_count": len(targets),
        "targets": targets,
    }


def prepare_dashboard_report_context_text(
    context: Mapping[str, Any],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Serialize report context and cap target-level evidence if needed."""

    text = json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    compacted = deepcopy(dict(context))
    for target in compacted.get("targets", []):
        if isinstance(target, dict):
            summary = target.get("raw_data_summary")
            if isinstance(summary, dict):
                summary["sample_rows"] = summary.get("sample_rows", [])[:3]
            ontology = target.get("ontology")
            if isinstance(ontology, dict):
                ontology["nodes"] = ontology.get("nodes", [])[:20]
                ontology["edges"] = ontology.get("edges", [])[:20]
            target["text_snapshot"] = _clean_text(target.get("text_snapshot"))[:1000]
    compacted["_context_truncated"] = True
    text = json.dumps(compacted, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 120] + "\n...[dashboard report context truncated to fit AskSage request limits]"


def build_dashboard_report_docx(
    report_context: Mapping[str, Any],
    narrative: str,
    *,
    target_images: Mapping[str, str] | None = None,
) -> bytes:
    """Build a formatted Word report as bytes."""

    target_images = target_images or {}
    document = Document()
    _configure_document_styles(document)

    dashboard = report_context.get("dashboard") if isinstance(report_context.get("dashboard"), Mapping) else {}
    title = _clean_text(dashboard.get("title")) or "Genesis Mission Funding and Collaboration Dashboard"
    generated_at = _clean_text(report_context.get("generated_at")) or datetime.now().isoformat(timespec="seconds")

    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = heading.add_run(f"{title}\nExecutive Summary")
    title_run.bold = True
    title_run.font.size = Pt(20)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run(f"Generated {generated_at}").italic = True

    metadata = _clean_text(dashboard.get("metadata"))
    if metadata:
        document.add_paragraph(metadata)

    document.add_heading("Current Dashboard Selections and Assumptions", level=1)
    if _clean_text(dashboard.get("current_view")):
        document.add_paragraph(f"Dashboard view: {_clean_text(dashboard.get('current_view'))}")
    assumptions = dashboard.get("assumptions") if isinstance(dashboard.get("assumptions"), list) else []
    for assumption in assumptions:
        document.add_paragraph(_clean_text(assumption), style="List Bullet")

    document.add_page_break()
    document.add_heading("Executive Narrative", level=1)
    _add_markdownish_text(document, narrative)

    document.add_page_break()
    document.add_heading("Supporting Dashboard Evidence", level=1)
    targets = report_context.get("targets") if isinstance(report_context.get("targets"), list) else []
    for index, target in enumerate(targets, start=1):
        if not isinstance(target, Mapping):
            continue
        title = _clean_text(target.get("title")) or f"Dashboard Element {index}"
        target_id = _clean_text(target.get("id"))
        document.add_heading(title, level=2)
        raw_summary = target.get("raw_data_summary") if isinstance(target.get("raw_data_summary"), Mapping) else {}
        if raw_summary:
            row_count = raw_summary.get("row_count", 0)
            columns = ", ".join(raw_summary.get("columns", [])[:8])
            document.add_paragraph(
                f"Source data: {row_count} row{'s' if row_count != 1 else ''}"
                + (f"; columns include {columns}." if columns else ".")
            )

        image_url = target_images.get(target_id)
        if image_url:
            _add_image(document, image_url, caption=f"Figure {index}. {title}")
        elif _clean_text(target.get("capture_error")):
            note = document.add_paragraph()
            note.add_run(f"Figure capture note: {_clean_text(target.get('capture_error'))}").italic = True

        sample_rows = raw_summary.get("sample_rows") if isinstance(raw_summary, Mapping) else None
        if isinstance(sample_rows, list) and sample_rows:
            _add_sample_table(document, sample_rows, caption=f"Table {index}. Representative data for {title}")

        session_insight = _clean_text(target.get("session_insight"))
        if session_insight:
            note = document.add_paragraph()
            note.add_run("AI insight generated during this dashboard session:").italic = True
            _add_markdownish_text(document, session_insight)

    document.add_heading("Conclusion and Recommended Next Steps", level=1)
    document.add_paragraph(
        "Use this report as an executive briefing companion to the dashboard. "
        "Validate resourcing or coordination decisions against the source programming rows, "
        "SME review notes, and current AskSage-generated narrative above."
    )

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


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
            elif self.path == "/api/asksage/report":
                report = request_asksage_dashboard_report(
                    payload,
                    base_url=self.asksage_base_url,
                )
                self._send_bytes(
                    report.body,
                    content_type=report.content_type,
                    filename=report.filename,
                )
                return
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
                {"error": "AskSage dashboard proxy failed.", "code": "proxy_failure"},
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

    def _send_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        filename: str,
        status_code: int = 200,
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
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


def _summarize_raw_data(value: Any, *, sample_limit: int) -> dict[str, Any]:
    rows = _raw_rows(value)
    columns = sorted({str(key) for row in rows for key in row.keys()})
    numeric_columns: dict[str, dict[str, float]] = {}
    for column in columns:
        numeric_values = []
        for row in rows:
            try:
                numeric_values.append(float(row.get(column)))
            except (TypeError, ValueError):
                continue
        if numeric_values:
            numeric_columns[column] = {
                "min": min(numeric_values),
                "max": max(numeric_values),
                "sum": sum(numeric_values),
            }
    return {
        "row_count": len(rows),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "sample_rows": rows[:sample_limit],
    }


def _raw_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = []
        for item in value:
            if isinstance(item, Mapping):
                rows.append({str(key): item.get(key) for key in item.keys()})
        return rows
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            for row in _raw_rows(item):
                rows.append({"dataset": key, **row})
        return rows
    return []


def _target_images_from_payload(targets: Sequence[Any]) -> dict[str, str]:
    images: dict[str, str] = {}
    for index, target in enumerate(targets, start=1):
        if not isinstance(target, Mapping):
            continue
        context = target.get("context") if isinstance(target.get("context"), Mapping) else {}
        target_meta = context.get("target") if isinstance(context.get("target"), Mapping) else {}
        target_id = _clean_text(target_meta.get("id")) or f"target-{index}"
        image = _clean_text(target.get("image"))
        if image.startswith("data:image/"):
            images[target_id] = image
    return images


def _report_system_prompt() -> str:
    return (
        "You are a senior executive briefing writer. Draft polished, evidence-grounded "
        "narrative for the Genesis Mission Funding and Collaboration Dashboard. "
        "Prioritize decision-relevant findings over technical detail, explain what "
        "charts and tables mean, identify trends, anomalies, risks, opportunities, "
        "and recommended next steps, and avoid unsupported claims."
    )


def _report_user_text(context_text: str) -> str:
    return (
        "Draft a report-ready executive narrative using the dashboard context below. "
        "Write for a senior executive audience. Use these sections: Executive Summary, "
        "Key Findings, Narrative Discussion, Risks and Opportunities, and Conclusion "
        "and Recommended Next Steps. Reference dashboard evidence when making claims.\n\n"
        "Dashboard report context JSON:\n"
        f"{context_text}\n\n"
        "Return readable Markdown. Use concise bullets where helpful and avoid raw data dumps."
    )


def _configure_document_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.5)
    for style_name in ["Heading 1", "Heading 2", "Title"]:
        style = document.styles[style_name]
        style.font.name = "Aptos"


def _add_markdownish_text(document: Document, text: str) -> None:
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            _add_markdown_table(document, table_lines)
            table_lines = []

    for raw_line in _clean_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            flush_table()
            continue
        if line.startswith("|") and line.endswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if line.startswith("### "):
            document.add_heading(line[4:].strip(), level=3)
        elif line.startswith("## "):
            document.add_heading(line[3:].strip(), level=2)
        elif line.startswith("# "):
            document.add_heading(line[2:].strip(), level=1)
        elif re.match(r"^[-*]\s+", line):
            document.add_paragraph(re.sub(r"^[-*]\s+", "", line), style="List Bullet")
        else:
            document.add_paragraph(line)
    flush_table()


def _add_markdown_table(document: Document, lines: list[str]) -> None:
    useful = [
        line
        for line in lines
        if not re.fullmatch(r"\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|", line)
    ]
    rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in useful
        if line.split("|")[1:-1]
    ]
    if not rows:
        return
    _add_table_rows(document, rows)


def _add_sample_table(document: Document, rows: list[Any], *, caption: str) -> None:
    dict_rows = [row for row in rows if isinstance(row, Mapping)]
    if not dict_rows:
        return
    columns = sorted({str(key) for row in dict_rows for key in row.keys()})[:6]
    table_rows = [columns]
    for row in dict_rows[:6]:
        table_rows.append([_clean_text(row.get(column))[:120] for column in columns])
    _add_table_rows(document, table_rows)
    caption_paragraph = document.add_paragraph(caption)
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.runs[0].italic = True


def _add_table_rows(document: Document, rows: list[list[str]]) -> None:
    column_count = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    for row_index, row_values in enumerate(rows):
        for column_index in range(column_count):
            value = row_values[column_index] if column_index < len(row_values) else ""
            cell = table.cell(row_index, column_index)
            cell.text = _clean_text(value)
            if row_index == 0:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True


def _add_image(document: Document, image_url: str, *, caption: str) -> None:
    image_bytes = _decode_data_url(image_url)
    if image_bytes is None:
        return
    try:
        document.add_picture(BytesIO(image_bytes), width=Inches(6.3))
    except Exception:
        return
    last_paragraph = document.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph = document.add_paragraph(caption)
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.runs[0].italic = True


def _decode_data_url(value: str) -> bytes | None:
    match = re.match(r"^data:image/(?:png|jpeg|jpg);base64,(.+)$", value, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        return base64.b64decode(match.group(1), validate=True)
    except ValueError:
        return None


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
