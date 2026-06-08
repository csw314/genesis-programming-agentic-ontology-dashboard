"""Static HTML report rendering for the classification dashboard."""

from __future__ import annotations

from html import escape
from itertools import combinations
import json
from math import sqrt
from typing import Any

import networkx as nx
import pandas as pd


FUNDING_LEVELS = ("BASELINE", "UFR", "ROT")
FY_COLUMNS = ("fy2027", "fy2028", "fy2029", "fy2030", "fy2031", "fy2032")
ASKSAGE_MULTIMODAL_MODELS = (
    {
        "id": "gpt-4.1-mini",
        "name": "GPT 4.1 Mini",
        "cost": "Low cost",
        "supports_image": True,
    },
    {
        "id": "gpt-4.1",
        "name": "GPT 4.1",
        "cost": "Medium cost",
        "supports_image": True,
    },
    {
        "id": "gpt-5.1",
        "name": "GPT 5.1",
        "cost": "High cost",
        "supports_image": True,
    },
    {
        "id": "google-gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "cost": "Low cost",
        "supports_image": True,
    },
    {
        "id": "google-gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "cost": "High cost",
        "supports_image": True,
    },
    {
        "id": "google-claude-46-sonnet",
        "name": "Claude 4.6 Sonnet",
        "cost": "Medium cost",
        "supports_image": True,
    },
    {
        "id": "google-claude-47-opus",
        "name": "Claude 4.7 Opus",
        "cost": "High cost",
        "supports_image": True,
    },
)
_INSIGHT_TERM_KEYS = {
    "actual_funding_offices",
    "challenge",
    "challenge_name",
    "claimed_office",
    "claimed_offices",
    "lead_offices",
    "lead_program_office",
    "normalized_funding_office",
    "office",
    "partner_offices",
    "predicted_challenge",
    "program_office_players",
    "wbs_name",
    "wbs_names",
}
_INSIGHT_NODE_ID_KEYS = {
    "challenge_node_id",
    "evidence_node_ids",
    "programming_record_id",
    "wbs_node_id",
}


def write_classification_dashboard_html(
    paths_module: Any,
    *,
    reviewed_classifications: pd.DataFrame,
    office_claim_validation: pd.DataFrame,
    challenge_coverage: pd.DataFrame,
    office_challenge_evidence: pd.DataFrame,
    metadata_text: str | None = None,
) -> str:
    """Build and write the executive classification dashboard HTML."""

    html = build_classification_dashboard_html(
        paths_module=paths_module,
        reviewed_classifications=reviewed_classifications,
        office_claim_validation=office_claim_validation,
        challenge_coverage=challenge_coverage,
        office_challenge_evidence=office_challenge_evidence,
        metadata_text=metadata_text,
    )
    paths_module.CLASSIFICATION_DASHBOARD_HTML_PATH.write_text(html, encoding="utf-8")
    return html


def build_classification_dashboard_html(
    paths_module: Any,
    *,
    reviewed_classifications: pd.DataFrame,
    office_claim_validation: pd.DataFrame,
    challenge_coverage: pd.DataFrame,
    office_challenge_evidence: pd.DataFrame,
    metadata_text: str | None = None,
) -> str:
    """Return static HTML focused on mission funding and collaboration."""

    metadata_html = _metadata_html(paths_module, metadata_text)
    claims = _claims_frame(office_claim_validation)
    views = _fynsp_views(reviewed_classifications)
    ontology_graph = _load_dashboard_ontology_graph(paths_module)
    view_html = "".join(
        _dashboard_view_html(
            view_id=view_id,
            view_label=view_label,
            reviewed_classifications=view_frame,
            office_claim_validation=office_claim_validation,
            challenge_coverage=challenge_coverage,
            office_challenge_evidence=office_challenge_evidence,
            claims=claims,
            ontology_graph=ontology_graph,
            default_visible=index == 0,
        )
        for index, (view_id, view_label, view_frame) in enumerate(views)
    )
    view_buttons = "".join(
        f"<button type=\"button\" class=\"{'active' if index == 0 else ''}\" "
        f"data-fynsp-view-button=\"{_ha(view_id)}\">{_h(view_label)}</button>"
        for index, (view_id, view_label, _) in enumerate(views)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Genesis Mission Funding and Collaboration Dashboard</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9e2ec;
      --soft: #f8fafc;
      --panel: #f0f4f8;
      --baseline: #3f6f9f;
      --ufr: #d97706;
      --rot: #7c3aed;
      --lead: #2f7d59;
      --partner: #6f55a2;
      --contributor: #52606d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: Segoe UI, Arial, sans-serif;
      margin: 0;
      color: var(--ink);
      background: #ffffff;
    }}
    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 32px 44px;
    }}
    h1, h2, h3 {{ color: #102a43; letter-spacing: 0; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    h2 {{ margin: 34px 0 10px; font-size: 22px; }}
    h3 {{ margin: 18px 0 8px; font-size: 17px; }}
    p {{ line-height: 1.42; }}
    .meta {{ color: var(--muted); margin: 0 0 22px; }}
    .section-note {{ color: var(--muted); max-width: 1080px; }}
    .asksage-credentials {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 18px 0 20px;
      background: var(--soft);
    }}
    .asksage-credentials h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .asksage-credentials-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      align-items: end;
    }}
    .asksage-field {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .asksage-field label {{
      color: #102a43;
      font-weight: 650;
    }}
    .asksage-field input,
    .asksage-field select,
    .ai-custom-query textarea {{
      width: 100%;
      border: 1px solid #bcccdc;
      border-radius: 6px;
      padding: 7px 8px;
      background: #fff;
      color: #102a43;
      font: inherit;
    }}
    .asksage-status {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .insight-target {{
      margin: 12px 0 24px;
    }}
    .insight-header {{
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 8px;
      margin: 4px 0 8px;
    }}
    .insight-button {{
      font-weight: 650;
    }}
    .insight-panel {{
      border: 1px solid var(--line);
      border-left: 4px solid #3f6f9f;
      border-radius: 8px;
      margin: 10px 0 18px;
      padding: 12px;
      background: #fff;
    }}
    .insight-panel[hidden] {{ display: none; }}
    .ai-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .ai-response {{
      min-height: 46px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: var(--soft);
      line-height: 1.42;
    }}
    .ai-response.loading {{
      color: var(--muted);
      font-style: italic;
    }}
    .ai-response.error {{
      border-color: #d97706;
      background: #fff7ed;
      color: #7c2d12;
    }}
    .ai-response table {{
      border-collapse: collapse;
      width: 100%;
      margin: 8px 0;
    }}
    .ai-response th,
    .ai-response td {{
      border: 1px solid var(--line);
      padding: 5px 7px;
      vertical-align: top;
    }}
    .ai-response pre {{
      white-space: pre-wrap;
      overflow-x: auto;
    }}
    .ai-custom-query {{
      display: grid;
      gap: 8px;
      margin: 10px 0;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0 26px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: #fff;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .metric .value {{
      display: block;
      margin-top: 5px;
      font-size: 23px;
      font-weight: 650;
      color: #102a43;
    }}
    .viz-block {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 12px 0 24px;
      overflow-x: auto;
    }}
    .tool-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: end;
      margin-bottom: 12px;
    }}
    .tool-control {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .tool-control label {{
      font-weight: 650;
      color: #102a43;
    }}
    .tool-control select {{
      min-width: 210px;
      border: 1px solid #bcccdc;
      border-radius: 6px;
      padding: 6px 8px;
      background: #fff;
      color: #102a43;
    }}
    .segmented {{
      display: inline-flex;
      border: 1px solid #bcccdc;
      border-radius: 7px;
      overflow: hidden;
      background: #fff;
    }}
    .segmented button {{
      border: 0;
      border-right: 1px solid #bcccdc;
      border-radius: 0;
      padding: 7px 10px;
      font-weight: 650;
    }}
    .segmented button:last-child {{ border-right: 0; }}
    .segmented button.active {{
      background: #102a43;
      color: #fff;
    }}
    .fynsp-toggle {{
      margin: 12px 0 20px;
    }}
    .fynsp-view[hidden] {{
      display: none;
    }}
    .disruptiveness-layout {{
      display: grid;
      grid-template-columns: minmax(760px, 1fr) minmax(250px, 320px);
      gap: 14px;
      align-items: stretch;
    }}
    .disruptiveness-svg {{
      width: 100%;
      min-width: 760px;
      height: auto;
      display: block;
      background: #fff;
    }}
    .detail-panel {{
      border-left: 3px solid #627d98;
      background: var(--soft);
      padding: 12px;
      font-size: 13px;
      line-height: 1.38;
    }}
    .detail-panel h3 {{ margin: 0 0 8px; }}
    .detail-grid {{
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 5px 8px;
    }}
    .detail-grid .label {{
      color: var(--muted);
      font-weight: 650;
    }}
    .tile-baseline {{ fill: var(--baseline); }}
    .tile-ufr {{ fill: var(--ufr); }}
    .tile-rot {{ fill: var(--rot); }}
    .tile-selected {{
      stroke: #102a43;
      stroke-width: 2.5;
    }}
    .data-table {{
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0 24px;
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      border: 1px solid var(--line);
      padding: 7px 9px;
      vertical-align: top;
    }}
    .data-table th {{
      background: var(--panel);
      text-align: left;
      font-weight: 650;
    }}
    .data-table tr:nth-child(even) td {{ background: var(--soft); }}
    .num {{ text-align: right; white-space: nowrap; }}
    .muted {{ color: var(--muted); }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      margin: 8px 0 12px;
    }}
    .legend-swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      margin-right: 5px;
      vertical-align: -1px;
    }}
    .bar-wrap {{
      width: 100%;
      min-width: 190px;
      height: 18px;
      background: #e5e7eb;
      border-radius: 4px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 18px;
      display: flex;
      min-width: 3px;
    }}
    .baseline {{ background: var(--baseline); }}
    .ufr {{ background: var(--ufr); }}
    .rot {{ background: var(--rot); }}
    .mini-stack {{
      display: flex;
      height: 7px;
      min-width: 72px;
      background: #e5e7eb;
      border-radius: 3px;
      overflow: hidden;
      margin-top: 4px;
    }}
    .role-pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 650;
      color: #fff;
      white-space: nowrap;
    }}
    .role-lead {{ background: var(--lead); }}
    .role-partner {{ background: var(--partner); }}
    .role-contributor {{ background: var(--contributor); }}
    .matrix-table {{
      min-width: 1040px;
      table-layout: fixed;
    }}
    .matrix-table th:first-child,
    .matrix-table td:first-child {{
      min-width: 250px;
      width: 250px;
    }}
    .matrix-cell {{
      min-height: 56px;
      font-size: 12px;
    }}
    .cell-total {{
      font-weight: 650;
      margin-bottom: 3px;
      white-space: nowrap;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      margin: 12px 0 24px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .card h3 {{ margin-top: 0; }}
    .chip-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 7px 0 10px;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      background: var(--soft);
    }}
    .office-card {{
      border-top: 3px solid #627d98;
    }}
    .office-summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(115px, 1fr));
      gap: 8px;
      margin: 8px 0 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .portfolio-row {{
      display: grid;
      grid-template-columns: minmax(150px, 1fr) minmax(140px, 220px) 82px;
      gap: 8px;
      align-items: center;
      margin: 8px 0;
      font-size: 12px;
    }}
    .citation-list {{
      margin: 8px 0 0 18px;
      padding: 0;
      font-size: 12px;
      color: var(--muted);
    }}
    .citation-list li {{ margin: 3px 0; }}
    .svg-chart {{
      width: 100%;
      min-width: 900px;
      height: auto;
      display: block;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      margin: 10px 0 14px;
      background: #fff;
    }}
    summary {{
      cursor: pointer;
      font-weight: 650;
      color: #102a43;
    }}
    .appendix-actions {{
      display: flex;
      gap: 8px;
      margin: 10px 0;
    }}
    button {{
      border: 1px solid #bcccdc;
      background: #fff;
      color: #102a43;
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    @media (max-width: 760px) {{
      .page {{ padding: 20px 14px 34px; }}
      .portfolio-row {{ grid-template-columns: 1fr; }}
      .disruptiveness-layout {{ grid-template-columns: 1fr; }}
      .data-table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
<div class="page">
  <h1>Genesis Mission Funding and Collaboration Dashboard</h1>
  <p class="meta">{metadata_html}</p>
  {_asksage_credentials_html()}
  <div class="tool-control fynsp-toggle">
    <label>FYNSP View</label>
    <div class="segmented" role="group" aria-label="FYNSP view">
      {view_buttons}
    </div>
  </div>
  {view_html}
</div>
<script>
  function setAppendix(open) {{
    document.querySelectorAll('.fynsp-view:not([hidden]) .appendix details').forEach(function(detail) {{
      detail.open = open;
    }});
  }}
  (function () {{
    const buttons = Array.from(document.querySelectorAll('[data-fynsp-view-button]'));
    const views = Array.from(document.querySelectorAll('[data-fynsp-view]'));
    function showView(viewId) {{
      views.forEach(function (view) {{
        view.hidden = view.dataset.fynspView !== viewId;
      }});
      buttons.forEach(function (button) {{
        button.classList.toggle('active', button.dataset.fynspViewButton === viewId);
      }});
    }}
    buttons.forEach(function (button) {{
      button.addEventListener('click', function () {{
        showView(button.dataset.fynspViewButton);
      }});
    }});
    if (buttons.length) showView(buttons[0].dataset.fynspViewButton);
  }})();
</script>
{_dashboard_insights_script()}
</body>
</html>
"""


def _asksage_credentials_html() -> str:
    options = "".join(
        "<option "
        f"value=\"{_ha(model['id'])}\" "
        f"data-cost=\"{_ha(model['cost'])}\" "
        f"data-supports-image=\"{'true' if model['supports_image'] else 'false'}\">"
        f"{_h(model['name'])} - {_h(model['cost'])}</option>"
        for model in ASKSAGE_MULTIMODAL_MODELS
    )
    return f"""
  <section class="asksage-credentials" aria-labelledby="asksage-credentials-heading">
    <h2 id="asksage-credentials-heading">AskSage AI Credentials</h2>
    <div class="asksage-credentials-grid">
      <div class="asksage-field">
        <label for="asksage-email">AskSage Email or Username</label>
        <input id="asksage-email" type="text" autocomplete="username" data-asksage-email>
      </div>
      <div class="asksage-field">
        <label for="asksage-api-key">AskSage API Key</label>
        <input id="asksage-api-key" type="password" autocomplete="off" data-asksage-api-key>
      </div>
      <div class="asksage-field">
        <label for="asksage-model">Preferred AskSage Model</label>
        <select id="asksage-model" data-asksage-model>
          {options}
        </select>
      </div>
      <div>
        <button type="button" data-asksage-refresh-models>Check Models</button>
      </div>
    </div>
    <div class="asksage-status" aria-live="polite" data-asksage-status>Models are loaded from dashboard configuration.</div>
  </section>
"""


def _insight_target_html(
    *,
    view_id: str,
    view_label: str,
    target_key: str,
    title: str,
    target_type: str,
    visual_html: str,
    raw_data: Any,
    metadata: dict[str, Any],
    ontology_graph: nx.MultiDiGraph | None,
) -> str:
    target_id = f"{view_id}-{target_key}"
    context = {
        "schema_version": 1,
        "target": {
            "id": target_id,
            "key": target_key,
            "title": title,
            "type": target_type,
            "fynsp_view": view_id,
            "fynsp_label": view_label,
        },
        "metadata": _jsonable_data(metadata),
        "raw_data": _jsonable_data(raw_data),
        "ontology": _ontology_context_for_target(
            ontology_graph,
            raw_data,
            metadata,
        ),
    }
    context_json = _json_script_text(context)
    return f"""
      <div class="insight-target" data-insight-target data-insight-target-id="{_ha(target_id)}" data-insight-title="{_ha(title)}">
        <div class="insight-header">
          <button type="button" class="insight-button" data-insight-open>Get Insights</button>
        </div>
        <div class="insight-visual" data-insight-visual>
          {visual_html}
        </div>
        <div class="insight-panel" data-insight-panel hidden></div>
        <script type="application/json" data-insight-context>{context_json}</script>
      </div>
"""


def _load_dashboard_ontology_graph(paths_module: Any) -> nx.MultiDiGraph | None:
    graph_path = getattr(paths_module, "CLASSIFIED_ONTOLOGY_GRAPH_PATH", None)
    if graph_path is None:
        return None
    try:
        if not graph_path.exists():
            return None
        return nx.read_graphml(graph_path)
    except Exception:
        return None


def _ontology_context_for_target(
    graph: nx.MultiDiGraph | None,
    raw_data: Any,
    metadata: dict[str, Any],
    *,
    max_seed_nodes: int = 48,
    max_nodes: int = 160,
    max_edges: int = 180,
) -> dict[str, Any]:
    if graph is None:
        return {"available": False, "reason": "classified ontology graph was not available"}

    json_data = _jsonable_data(raw_data)
    terms: set[str] = set()
    node_ids: set[str] = set()
    _collect_insight_terms(json_data, terms, node_ids)
    _collect_insight_terms(_jsonable_data(metadata), terms, node_ids)

    label_lookup: dict[str, list[str]] = {}
    for node_id, data in graph.nodes(data=True):
        label = _clean_text(data.get("label"))
        if label:
            label_lookup.setdefault(label.casefold(), []).append(str(node_id))

    seeds: list[str] = []
    for node_id in sorted(node_ids):
        if node_id in graph and node_id not in seeds:
            seeds.append(node_id)
    for term in sorted(terms):
        for node_id in label_lookup.get(term.casefold(), []):
            if node_id not in seeds:
                seeds.append(node_id)
            if len(seeds) >= max_seed_nodes:
                break
        if len(seeds) >= max_seed_nodes:
            break

    selected_nodes: dict[str, dict[str, Any]] = {}
    selected_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node_id: str) -> None:
        if node_id not in graph or node_id in selected_nodes or len(selected_nodes) >= max_nodes:
            return
        data = graph.nodes[node_id]
        selected_nodes[node_id] = {
            "id": node_id,
            "label": _clean_text(data.get("label")) or node_id,
            "node_type": _clean_text(data.get("node_type")),
            "source_files": _clean_text(data.get("source_files")),
        }

    def add_edge(source: str, target: str, key: Any, data: dict[str, Any]) -> None:
        edge_key = (str(source), str(target), str(key))
        if edge_key in seen_edges or len(selected_edges) >= max_edges:
            return
        seen_edges.add(edge_key)
        add_node(str(source))
        add_node(str(target))
        selected_edges.append(
            {
                "source": str(source),
                "target": str(target),
                "relation": _clean_text(data.get("relation")) or "related_to",
                "label": _clean_text(data.get("label")),
            }
        )

    for seed in seeds[:max_seed_nodes]:
        add_node(seed)
        if len(selected_edges) >= max_edges:
            break
        for source, target, key, data in graph.out_edges(seed, keys=True, data=True):
            add_edge(source, target, key, data)
            if len(selected_edges) >= max_edges:
                break
        if len(selected_edges) >= max_edges:
            break
        for source, target, key, data in graph.in_edges(seed, keys=True, data=True):
            add_edge(source, target, key, data)
            if len(selected_edges) >= max_edges:
                break

    return {
        "available": True,
        "summary": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "source_files": _clean_text(graph.graph.get("source_files")),
        },
        "selection": {
            "seed_node_count": len(seeds[:max_seed_nodes]),
            "term_count": len(terms),
            "node_count": len(selected_nodes),
            "edge_count": len(selected_edges),
            "capped": len(seeds) > max_seed_nodes
            or len(selected_nodes) >= max_nodes
            or len(selected_edges) >= max_edges,
        },
        "nodes": list(selected_nodes.values()),
        "edges": selected_edges,
    }


def _collect_insight_terms(
    value: Any,
    terms: set[str],
    node_ids: set[str],
    *,
    key: str = "",
) -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _collect_insight_terms(child_value, terms, node_ids, key=str(child_key))
        return
    if isinstance(value, list):
        for item in value:
            _collect_insight_terms(item, terms, node_ids, key=key)
        return

    text = _clean_text(value)
    if not text:
        return
    if key in _INSIGHT_NODE_ID_KEYS:
        for part in _split_semicolon_list(text):
            node_ids.add(part)
    if key in _INSIGHT_TERM_KEYS:
        for part in _split_semicolon_list(text):
            terms.add(part)


def _jsonable_data(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        return [
            {str(key): _jsonable_data(item) for key, item in row.items()}
            for row in frame.where(pd.notna(frame), None).to_dict(orient="records")
        ]
    if isinstance(value, pd.Series):
        return {str(key): _jsonable_data(item) for key, item in value.to_dict().items()}
    if isinstance(value, dict):
        return {str(key): _jsonable_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_data(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json_script_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")


def _dashboard_insights_script() -> str:
    return r"""
<script>
  (function () {
    const emailInput = document.querySelector('[data-asksage-email]');
    const apiKeyInput = document.querySelector('[data-asksage-api-key]');
    const modelSelect = document.querySelector('[data-asksage-model]');
    const status = document.querySelector('[data-asksage-status]');
    const refreshButton = document.querySelector('[data-asksage-refresh-models]');

    function escapeHtml(value) {
      return String(value || '').replace(/[&<>"']/g, function (character) {
        return ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]);
      });
    }
    function setStatus(message) {
      if (status) status.textContent = message;
    }
    function credentials() {
      return {
        email: emailInput ? emailInput.value.trim() : '',
        apiKey: apiKeyInput ? apiKeyInput.value.trim() : ''
      };
    }
    function selectedModel() {
      const option = modelSelect ? modelSelect.options[modelSelect.selectedIndex] : null;
      return {
        id: modelSelect ? modelSelect.value : '',
        cost: option ? option.dataset.cost || '' : '',
        supportsImage: option ? option.dataset.supportsImage !== 'false' : true
      };
    }
    function credentialError() {
      const creds = credentials();
      if (!creds.email) return 'Enter your AskSage email or username.';
      if (!creds.apiKey) return 'Enter your AskSage API key.';
      if (!selectedModel().id) return 'Select an AskSage model.';
      return '';
    }
    function populateModels(models) {
      if (!modelSelect || !Array.isArray(models) || !models.length) return;
      const current = modelSelect.value;
      modelSelect.replaceChildren();
      models.forEach(function (model) {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = (model.name || model.id) + ' - ' + (model.cost || 'Cost not configured');
        option.dataset.cost = model.cost || '';
        option.dataset.supportsImage = model.supports_image === false ? 'false' : 'true';
        modelSelect.appendChild(option);
      });
      if (models.some(function (model) { return model.id === current; })) {
        modelSelect.value = current;
      }
    }
    async function refreshModels() {
      const creds = credentials();
      if (!creds.email || !creds.apiKey) {
        setStatus('Enter AskSage credentials before checking models.');
        return;
      }
      setStatus('Checking AskSage models...');
      try {
        const response = await fetch('/api/asksage/models', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({credentials: creds})
        });
        const payload = await response.json().catch(function () { return {}; });
        if (!response.ok) throw new Error(payload.error || 'Unable to check AskSage models.');
        populateModels(payload.models || []);
        setStatus('AskSage models are ready.');
      } catch (error) {
        setStatus(error.message || 'AskSage model check failed.');
      }
    }
    if (refreshButton) refreshButton.addEventListener('click', refreshModels);

    function targetContext(target) {
      const dataElement = target.querySelector('[data-insight-context]');
      const context = JSON.parse(dataElement ? dataElement.textContent || '{}' : '{}');
      context.current_dashboard_selections = currentSelections(target);
      return context;
    }
    function currentSelections(target) {
      const activeViewButton = document.querySelector('[data-fynsp-view-button].active');
      const selections = {
        fynsp_view: activeViewButton ? activeViewButton.dataset.fynspViewButton || '' : ''
      };
      const disruptiveness = target.querySelector('.disruptiveness-tool');
      if (disruptiveness) {
        const activeMode = disruptiveness.querySelector('[data-disruptiveness-mode].active');
        const office = disruptiveness.querySelector('[data-disruptiveness-office-filter]');
        const challenge = disruptiveness.querySelector('[data-disruptiveness-challenge-filter]');
        selections.disruptiveness = {
          funding_mode: activeMode ? activeMode.dataset.disruptivenessMode || '' : '',
          office: office ? office.value : '',
          challenge: challenge ? challenge.value : ''
        };
      }
      return selections;
    }
    function ensurePanel(target) {
      const panel = target.querySelector('[data-insight-panel]');
      if (!panel || panel.dataset.initialized === 'true') return panel;
      panel.innerHTML = [
        '<div class="ai-actions">',
        '<button type="button" data-insight-action="summarize">Summarize Insights</button>',
        '<button type="button" data-insight-action="raw">Get Raw Data</button>',
        '<button type="button" data-insight-action="custom">Write Your Own Query</button>',
        '</div>',
        '<form class="ai-custom-query" data-insight-custom-form>',
        '<textarea rows="3" data-insight-custom-input aria-label="Custom AskSage query"></textarea>',
        '<div><button type="submit">Send Query</button></div>',
        '</form>',
        '<div class="ai-response" data-insight-response>Select an option to begin.</div>'
      ].join('');
      panel.querySelectorAll('[data-insight-action]').forEach(function (button) {
        button.addEventListener('click', function () {
          if (button.dataset.insightAction === 'raw') {
            showRawData(target, panel);
          } else if (button.dataset.insightAction === 'custom') {
            const input = panel.querySelector('[data-insight-custom-input]');
            if (input) input.focus();
          } else {
            runInsight(target, panel, 'summarize', 'Summarize the most important insights from this dashboard visualization. Focus on funding magnitude, collaboration posture, notable outliers, risks, and recommended next questions.');
          }
        });
      });
      const form = panel.querySelector('[data-insight-custom-form]');
      form.addEventListener('submit', function (event) {
        event.preventDefault();
        const input = panel.querySelector('[data-insight-custom-input]');
        const prompt = input ? input.value.trim() : '';
        if (!prompt) {
          setResponse(panel, 'Enter a question about this chart or table.', 'error');
          return;
        }
        runInsight(target, panel, 'custom', prompt);
      });
      panel.dataset.initialized = 'true';
      return panel;
    }
    function setResponse(panel, html, state) {
      const response = panel.querySelector('[data-insight-response]');
      if (!response) return;
      response.classList.toggle('loading', state === 'loading');
      response.classList.toggle('error', state === 'error');
      response.innerHTML = html;
    }
    function showRawData(target, panel) {
      const context = targetContext(target);
      setResponse(panel, '<pre>' + escapeHtml(JSON.stringify(context.raw_data || {}, null, 2)) + '</pre>', '');
    }
    async function runInsight(target, panel, promptKind, prompt) {
      const missing = credentialError();
      if (missing) {
        setResponse(panel, escapeHtml(missing), 'error');
        return;
      }
      setResponse(panel, 'Generating AskSage response...', 'loading');
      try {
        const context = targetContext(target);
        const image = await captureTargetImage(target);
        const response = await fetch('/api/asksage/insights', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            credentials: credentials(),
            model: selectedModel().id,
            prompt_kind: promptKind,
            prompt: prompt,
            context: context,
            image: image
          })
        });
        const payload = await response.json().catch(function () { return {}; });
        if (!response.ok) throw new Error(payload.error || 'AskSage request failed.');
        setResponse(panel, formatText(payload.answer || ''), '');
      } catch (error) {
        setResponse(panel, escapeHtml(error.message || 'AskSage request failed.'), 'error');
      }
    }
    async function captureTargetImage(target) {
      const visual = target.querySelector('[data-insight-visual]');
      if (!visual) return null;
      const svg = visual.querySelector('svg');
      if (svg) {
        try {
          return await svgToPng(svg);
        } catch (error) {
          return textSnapshotImage(visual);
        }
      }
      return textSnapshotImage(visual);
    }
    function svgToPng(svg) {
      return new Promise(function (resolve, reject) {
        const source = new XMLSerializer().serializeToString(svg);
        const blob = new Blob([source], {type: 'image/svg+xml;charset=utf-8'});
        const url = URL.createObjectURL(blob);
        const image = new Image();
        image.onload = function () {
          const viewBox = svg.viewBox && svg.viewBox.baseVal;
          const width = Math.max(320, Math.ceil(viewBox && viewBox.width ? viewBox.width : svg.getBoundingClientRect().width || 900));
          const height = Math.max(180, Math.ceil(viewBox && viewBox.height ? viewBox.height : svg.getBoundingClientRect().height || 500));
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const context = canvas.getContext('2d');
          context.fillStyle = '#ffffff';
          context.fillRect(0, 0, width, height);
          context.drawImage(image, 0, 0, width, height);
          URL.revokeObjectURL(url);
          resolve(canvas.toDataURL('image/png'));
        };
        image.onerror = function () {
          URL.revokeObjectURL(url);
          reject(new Error('Unable to capture chart image.'));
        };
        image.src = url;
      });
    }
    function textSnapshotImage(element) {
      const lines = [];
      const rows = Array.from(element.querySelectorAll('tr')).slice(0, 80);
      if (rows.length) {
        rows.forEach(function (row) {
          lines.push(Array.from(row.children).map(function (cell) {
            return cell.textContent.trim().replace(/\s+/g, ' ');
          }).join(' | '));
        });
      } else {
        element.textContent.split(/\n+/).forEach(function (line) {
          const clean = line.trim().replace(/\s+/g, ' ');
          if (clean) lines.push(clean);
        });
      }
      const canvas = document.createElement('canvas');
      const width = 1200;
      const height = Math.max(180, Math.min(1800, 42 + lines.length * 22));
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d');
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, width, height);
      context.fillStyle = '#102a43';
      context.font = '14px Segoe UI, Arial, sans-serif';
      lines.slice(0, 80).forEach(function (line, index) {
        context.fillText(line.slice(0, 150), 18, 28 + index * 22);
      });
      return canvas.toDataURL('image/png');
    }
    function formatText(text) {
      const lines = String(text || '').split(/\r?\n/);
      const output = [];
      let paragraph = [];
      let bullets = [];
      function flushParagraph() {
        if (paragraph.length) {
          output.push('<p>' + escapeHtml(paragraph.join(' ')) + '</p>');
          paragraph = [];
        }
      }
      function flushBullets() {
        if (bullets.length) {
          output.push('<ul>' + bullets.map(function (item) { return '<li>' + escapeHtml(item) + '</li>'; }).join('') + '</ul>');
          bullets = [];
        }
      }
      for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index].trim();
        if (!line) {
          flushParagraph();
          flushBullets();
          continue;
        }
        if (/^\|.+\|$/.test(line)) {
          flushParagraph();
          flushBullets();
          const tableLines = [];
          while (index < lines.length && /^\|.+\|$/.test(lines[index].trim())) {
            tableLines.push(lines[index].trim());
            index += 1;
          }
          index -= 1;
          output.push(markdownTable(tableLines));
        } else if (/^[-*]\s+/.test(line)) {
          flushParagraph();
          bullets.push(line.replace(/^[-*]\s+/, ''));
        } else {
          flushBullets();
          paragraph.push(line);
        }
      }
      flushParagraph();
      flushBullets();
      return output.join('') || '<p>No response text was returned.</p>';
    }
    function markdownTable(lines) {
      const useful = lines.filter(function (line) {
        return !/^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|$/.test(line);
      });
      const rows = useful.map(function (line) {
        return line.split('|').slice(1, -1).map(function (cell) { return cell.trim(); });
      });
      if (!rows.length) return '';
      const header = rows.shift();
      return '<table><thead><tr>' + header.map(function (cell) {
        return '<th>' + escapeHtml(cell) + '</th>';
      }).join('') + '</tr></thead><tbody>' + rows.map(function (row) {
        return '<tr>' + row.map(function (cell) { return '<td>' + escapeHtml(cell) + '</td>'; }).join('') + '</tr>';
      }).join('') + '</tbody></table>';
    }
    document.querySelectorAll('[data-insight-target]').forEach(function (target) {
      const openButton = target.querySelector('[data-insight-open]');
      const panel = target.querySelector('[data-insight-panel]');
      if (!openButton || !panel) return;
      openButton.addEventListener('click', function () {
        const initializedPanel = ensurePanel(target);
        initializedPanel.hidden = !initializedPanel.hidden;
      });
    });
  })();
</script>
"""


def _fynsp_views(reviewed: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    frame = reviewed.copy()
    if "fynsp_cycle" not in frame.columns:
        frame["fynsp_cycle"] = ""
    cycle = frame["fynsp_cycle"].map(_clean_text)
    return [
        ("combined", "Combined", frame),
        ("fynsp-2028", "FYNSP 2028", frame.loc[cycle.eq("FYNSP 2028")].copy()),
        ("fynsp-2027", "FYNSP 2027", frame.loc[cycle.eq("FYNSP 2027")].copy()),
    ]


def _dashboard_view_html(
    *,
    view_id: str,
    view_label: str,
    reviewed_classifications: pd.DataFrame,
    office_claim_validation: pd.DataFrame,
    challenge_coverage: pd.DataFrame,
    office_challenge_evidence: pd.DataFrame,
    claims: pd.DataFrame,
    ontology_graph: nx.MultiDiGraph | None,
    default_visible: bool,
) -> str:
    positive = _positive_funding_rows(reviewed_classifications)
    role_map = _claim_role_map(claims)

    challenge_summary = _challenge_funding_summary(
        positive,
        challenge_coverage,
        claims,
    )
    contributions = _office_challenge_contributions(positive, role_map, claims)
    office_summary = _office_portfolio_summary(positive, claims)
    office_challenge = _office_challenge_portfolio(positive, role_map, claims)
    lead_summary = _lead_claim_funding_summary(positive, claims, contributions)
    pair_opportunities = _office_pair_opportunities(contributions)
    disruptiveness_rows = _disruptiveness_rows(positive)
    unclaimed_evidence = _unclaimed_evidence(office_challenge_evidence)
    appendix_claims = _appendix_claim_validation(office_claim_validation)
    appendix_rows = _appendix_reviewed_rows(reviewed_classifications)

    totals_by_level = {
        level: float(challenge_summary[level].sum()) if level in challenge_summary.columns else 0.0
        for level in FUNDING_LEVELS
    }
    total_funding = sum(totals_by_level.values())
    funded_challenges = int(challenge_summary["total_funding"].gt(0).sum())
    contributing_offices = int(positive["normalized_funding_office"].nunique()) if not positive.empty else 0
    program_office_players = int(
        pd.concat(
            [
                positive["normalized_funding_office"] if not positive.empty else pd.Series(dtype=str),
                claims["claimed_office"] if not claims.empty else pd.Series(dtype=str),
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .nunique()
    )
    multi_office_challenges = int(challenge_summary["contributing_office_count"].gt(1).sum())
    hidden_attr = "" if default_visible else " hidden"
    metric_cards = "".join(
        [_metric("Total BASELINE + UFR + ROT", _format_money(total_funding))]
        + [_metric(level, _format_money(totals_by_level[level])) for level in FUNDING_LEVELS]
        + [
            _metric("Funded Challenges", f"{funded_challenges:,}"),
            _metric("Contributing Offices", f"{contributing_offices:,}"),
            _metric("Program Office Players", f"{program_office_players:,}"),
            _metric("Multi-Office Challenges", f"{multi_office_challenges:,}"),
        ]
    )
    mission_funding_mix_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="mission-funding-mix",
        title="Mission Funding Mix",
        target_type="table",
        visual_html=_mission_funding_mix_html(challenge_summary),
        raw_data=challenge_summary,
        metadata={
            "description": "Funding by Genesis Mission Challenge across BASELINE, UFR, and ROT.",
            "labels": ["Genesis Mission Challenge", "Funding Mix", "Total", *FUNDING_LEVELS],
            "legend": list(FUNDING_LEVELS),
        },
        ontology_graph=ontology_graph,
    )
    contribution_matrix_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="challenge-office-matrix",
        title="Challenge-by-Office Contribution Matrix",
        target_type="table",
        visual_html=_challenge_office_matrix_html(contributions, challenge_summary),
        raw_data=contributions,
        metadata={
            "description": "Funding and role contribution matrix by mission challenge and program office.",
            "labels": ["Challenge", "Program Offices", "Funding Mix", "Role", "Rows", "WBS"],
            "legend": [*FUNDING_LEVELS, "Lead", "Partner", "Actual contributor"],
        },
        ontology_graph=ontology_graph,
    )
    collaboration_posture_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="collaboration-posture",
        title="Collaboration Posture",
        target_type="chart",
        visual_html=_collaboration_bubble_html(challenge_summary),
        raw_data=challenge_summary,
        metadata={
            "description": "Bubble chart comparing program office player count, total funding, and UFR/ROT share.",
            "x_axis": "Program office players",
            "size": "Total funding",
            "color": "Funding posture from BASELINE-heavy to UFR/ROT-heavy",
        },
        ontology_graph=ontology_graph,
    )
    collaboration_groups_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="challenge-collaboration-groups",
        title="Challenge Collaboration Groups",
        target_type="cards",
        visual_html=_challenge_collaboration_cards_html(challenge_summary, contributions, claims),
        raw_data={
            "challenge_summary": challenge_summary,
            "contributions": contributions,
            "claims": claims,
        },
        metadata={
            "description": "Challenge cards grouping claimed leads, partners, and actual funding contributors.",
            "labels": ["Lead", "Claimed partners", "Other funding contributors", "Suggested collaboration group"],
        },
        ontology_graph=ontology_graph,
    )
    project_disruptiveness_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="project-funding-disruptiveness",
        title="Project Funding Disruptiveness",
        target_type="chart",
        visual_html=_project_disruptiveness_html(disruptiveness_rows, view_id=view_id),
        raw_data=disruptiveness_rows,
        metadata={
            "description": "Packed tilemap of row-level funding share and AI-directness score.",
            "x_axis": "AI-directness score, with more directly AI work to the left",
            "y_axis": "Share of office funding in selected mode",
            "filters": ["Funding Mode", "Program Office", "Genesis Mission Challenge"],
            "legend": [f"{level} tile" for level in FUNDING_LEVELS],
        },
        ontology_graph=ontology_graph,
    )
    office_portfolios_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="office-portfolios",
        title="Office Portfolios",
        target_type="cards",
        visual_html=_office_portfolios_html(office_summary, office_challenge, positive),
        raw_data={
            "office_summary": office_summary,
            "office_challenge": office_challenge,
            "positive_programming_rows": positive,
        },
        metadata={
            "description": "Office cards showing funded challenges, claimed roles, funding mix, and citations.",
            "labels": ["Funded challenges", "Lead claims", "Partner claims", *FUNDING_LEVELS],
        },
        ontology_graph=ontology_graph,
    )
    lead_funding_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="lead-office-funding",
        title="Lead Office Funding by Claimed Mission",
        target_type="table",
        visual_html=_lead_claim_funding_html(lead_summary),
        raw_data=lead_summary,
        metadata={
            "description": "Lead program office funding against claimed Genesis Mission Challenges.",
            "labels": ["Lead Office", "Genesis Mission Challenge", "Funding Mix", "Total", *FUNDING_LEVELS],
        },
        ontology_graph=ontology_graph,
    )
    challenge_coverage_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="challenge-coverage",
        title="Challenge Coverage",
        target_type="table",
        visual_html=_challenge_coverage_html(challenge_summary),
        raw_data=challenge_summary,
        metadata={
            "description": "Funding coverage, claimed offices, actual contributors, and coordination signal.",
            "labels": ["Genesis Mission Challenge", "Total", *FUNDING_LEVELS, "Office Players", "Funding Offices"],
        },
        ontology_graph=ontology_graph,
    )
    appendix_claims_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="appendix-office-claim-validation",
        title="Office Claim Validation",
        target_type="table",
        visual_html=_table_html(appendix_claims),
        raw_data=appendix_claims,
        metadata={"description": "Validation status and evidence for claimed office roles."},
        ontology_graph=ontology_graph,
    )
    appendix_evidence_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="appendix-unclaimed-positive-evidence",
        title="Unclaimed Positive Evidence",
        target_type="table",
        visual_html=_table_html(unclaimed_evidence),
        raw_data=unclaimed_evidence,
        metadata={"description": "Positive funding evidence that was not matched to claimed offices."},
        ontology_graph=ontology_graph,
    )
    appendix_pairs_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="appendix-office-pair-opportunities",
        title="Office Pair Collaboration Opportunities",
        target_type="table",
        visual_html=_office_pair_opportunities_html(pair_opportunities),
        raw_data=pair_opportunities,
        metadata={"description": "Office pairs that share funded mission areas."},
        ontology_graph=ontology_graph,
    )
    appendix_reviewed_rows_html = _insight_target_html(
        view_id=view_id,
        view_label=view_label,
        target_key="appendix-reviewed-row-level-classifications",
        title="Reviewed Row-Level Classifications",
        target_type="table",
        visual_html=_table_html(appendix_rows),
        raw_data=appendix_rows,
        metadata={"description": "Reviewed row-level classifications, SME corrections, and AI-directness fields."},
        ontology_graph=ontology_graph,
    )

    return f"""
  <div class="fynsp-view" data-fynsp-view="{_ha(view_id)}"{hidden_attr}>
    <section>
      <h2>{_h(view_label)} Executive Mission Funding Snapshot</h2>
      <div class="metric-grid">
        {metric_cards}
      </div>
    </section>

    <section>
      <h2>Mission Funding Mix</h2>
      <p class="section-note">How much funding is available in BASELINE, UFR, and ROT funds for each Genesis Mission Challenge?</p>
      {_legend()}
      {mission_funding_mix_html}
    </section>

    <section>
      <h2>Challenge-by-Office Contribution Matrix</h2>
      <p class="section-note">Which program offices contribute BASELINE, UFR, and ROT funds to each challenge, regardless of whether they are the lead office?</p>
      {contribution_matrix_html}
    </section>

    <section>
      <h2>Collaboration Posture</h2>
      <p class="section-note">Challenges farther right have more program office players, including claimed leads/partners and actual funding contributors; larger circles have more total funding; warmer circles depend more on UFR/ROT funds.</p>
      {collaboration_posture_html}
    </section>

    <section>
      <h2>Challenge Collaboration Groups</h2>
      <p class="section-note">These cards group the offices that should be coordinating around each challenge: claimed leads, claimed partners, and other actual funding contributors.</p>
      {collaboration_groups_html}
    </section>

    <section>
      <h2>Project Funding Disruptiveness</h2>
      <p class="section-note">Each square is one programming row. Higher tiles capture a larger share of the selected funding mode for their office; farther-right tiles are more directly AI/ML work. High-share, low-AI rows are the most disruptive candidates to tax or rescope.</p>
      {project_disruptiveness_html}
    </section>

    <section>
      <h2>Office Portfolios</h2>
      <p class="section-note">Each office view shows the funded Genesis Mission Challenges in its portfolio, the BASELINE/UFR/ROT mix, and programming-data citations for traceability.</p>
      {office_portfolios_html}
    </section>

    <section>
      <h2>Lead Office Funding by Claimed Mission</h2>
      <p class="section-note">How much funding are lead program offices contributing to their claimed Genesis Mission Challenge?</p>
      {lead_funding_html}
    </section>

    <section>
      <h2>Challenge Coverage</h2>
      <p class="section-note">How much funding coverage is there for each Genesis Mission Challenge, and which program offices are contributing?</p>
      {challenge_coverage_html}
    </section>

    <section class="appendix">
      <h2>Appendix: Validation and Traceability</h2>
      <p class="section-note">Validation, confidence, review, and row-level FYNSP programming details are available here.</p>
      <div class="appendix-actions">
        <button type="button" onclick="setAppendix(true)">Open all</button>
        <button type="button" onclick="setAppendix(false)">Close all</button>
      </div>
      <details>
        <summary>Office Claim Validation</summary>
        {appendix_claims_html}
      </details>
      <details>
        <summary>Unclaimed Positive Evidence</summary>
        {appendix_evidence_html}
      </details>
      <details>
        <summary>Office Pair Collaboration Opportunities</summary>
        {appendix_pairs_html}
      </details>
      <details>
        <summary>Reviewed Row-Level Classifications</summary>
        {appendix_reviewed_rows_html}
      </details>
    </section>
  </div>
"""


def _positive_funding_rows(reviewed: pd.DataFrame) -> pd.DataFrame:
    frame = reviewed.copy()
    frame["funding_level"] = frame["funding_level"].astype(str).str.upper().str.strip()
    frame = frame.loc[frame["funding_level"].isin(FUNDING_LEVELS)].copy()
    for column in (*FY_COLUMNS, "five_year_total"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if "programming_record_id" not in frame.columns:
        frame["programming_record_id"] = frame.index.astype(str)
    return frame


def _claims_frame(office_claim_validation: pd.DataFrame) -> pd.DataFrame:
    claims = office_claim_validation.copy()
    if claims.empty:
        return pd.DataFrame(columns=["challenge_name", "claimed_office", "claim_role"])
    claims["claimed_office"] = claims["claimed_office"].map(_clean_text)
    claims["claim_role"] = claims["claim_role"].map(lambda value: _clean_text(value).lower())
    claims["challenge_name"] = claims["challenge_name"].map(_clean_text)
    return claims.loc[
        claims["challenge_name"].ne("") & claims["claimed_office"].ne(""),
        ["challenge_name", "claimed_office", "claim_role"],
    ].drop_duplicates()


def _claim_role_map(claims: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {
        (_clean_text(row["challenge_name"]), _clean_text(row["claimed_office"])): _clean_text(row["claim_role"]).lower()
        for _, row in claims.iterrows()
    }


def _challenge_funding_summary(
    positive: pd.DataFrame,
    challenge_coverage: pd.DataFrame,
    claims: pd.DataFrame,
) -> pd.DataFrame:
    if positive.empty:
        funding = pd.DataFrame(columns=["challenge_name", *FUNDING_LEVELS])
        counts = pd.DataFrame(
            columns=["challenge_name", "project_rows", "wbs_count", "contributing_office_count"]
        )
    else:
        funding = (
            positive.pivot_table(
                index="predicted_challenge",
                columns="funding_level",
                values="five_year_total",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
            .rename(columns={"predicted_challenge": "challenge_name"})
        )
        counts = (
            positive.groupby("predicted_challenge", as_index=False)
            .agg(
                project_rows=("programming_record_id", "nunique"),
                wbs_count=("wbs_name", "nunique"),
                contributing_office_count=("normalized_funding_office", "nunique"),
            )
            .rename(columns={"predicted_challenge": "challenge_name"})
        )
    for level in FUNDING_LEVELS:
        if level not in funding.columns:
            funding[level] = 0.0

    base_challenges = pd.DataFrame(
        {
            "challenge_name": sorted(
                set(_series_text(challenge_coverage.get("challenge_name")))
                | set(_series_text(funding.get("challenge_name")))
            )
        }
    )
    summary = (
        base_challenges.merge(funding[["challenge_name", *FUNDING_LEVELS]], on="challenge_name", how="left")
        .merge(counts, on="challenge_name", how="left")
    )
    for column in [*FUNDING_LEVELS, "project_rows", "wbs_count", "contributing_office_count"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0)
    summary["total_funding"] = summary[list(FUNDING_LEVELS)].sum(axis=1)
    summary["ufr_share"] = summary.apply(
        lambda row: 0.0
        if row["total_funding"] <= 0
        else (float(row["UFR"]) + float(row["ROT"])) / float(row["total_funding"]),
        axis=1,
    )

    claim_summary = _claim_summary(claims)
    actual_offices = _actual_offices_by_challenge(positive)
    summary = summary.merge(claim_summary, on="challenge_name", how="left").merge(
        actual_offices,
        on="challenge_name",
        how="left",
    )
    for column in ["lead_offices", "partner_offices", "claimed_offices", "actual_funding_offices"]:
        summary[column] = summary[column].fillna("")
    summary["program_office_players"] = summary.apply(
        lambda row: _join_sorted(
            _split_semicolon_list(row["claimed_offices"])
            + _split_semicolon_list(row["actual_funding_offices"])
        ),
        axis=1,
    )
    summary["program_office_player_count"] = summary["program_office_players"].map(
        lambda value: len(_split_semicolon_list(value))
    )
    summary["collaboration_signal"] = summary.apply(_collaboration_signal, axis=1)
    return summary.sort_values(["total_funding", "challenge_name"], ascending=[False, True]).reset_index(drop=True)


def _claim_summary(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame(
            columns=["challenge_name", "lead_offices", "partner_offices", "claimed_offices"]
        )

    rows = []
    for challenge, group in claims.groupby("challenge_name"):
        lead = _join_sorted(group.loc[group["claim_role"].eq("lead"), "claimed_office"])
        partners = _join_sorted(group.loc[group["claim_role"].eq("partner"), "claimed_office"])
        claimed = _join_sorted(group["claimed_office"])
        rows.append(
            {
                "challenge_name": challenge,
                "lead_offices": lead,
                "partner_offices": partners,
                "claimed_offices": claimed,
            }
        )
    return pd.DataFrame(rows)


def _actual_offices_by_challenge(positive: pd.DataFrame) -> pd.DataFrame:
    if positive.empty:
        return pd.DataFrame(columns=["challenge_name", "actual_funding_offices"])
    return (
        positive.groupby("predicted_challenge")["normalized_funding_office"]
        .agg(_join_sorted)
        .reset_index()
        .rename(
            columns={
                "predicted_challenge": "challenge_name",
                "normalized_funding_office": "actual_funding_offices",
            }
        )
    )


def _office_challenge_contributions(
    positive: pd.DataFrame,
    role_map: dict[tuple[str, str], str],
    claims: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = [
        "challenge_name",
        "normalized_funding_office",
        *FUNDING_LEVELS,
        "total_funding",
        "project_rows",
        "wbs_count",
        "wbs_names",
        "role",
        "role_rank",
        "role_label",
    ]
    if positive.empty:
        funding = pd.DataFrame(columns=["challenge_name", "normalized_funding_office", *FUNDING_LEVELS])
        counts = pd.DataFrame(columns=["challenge_name", "normalized_funding_office", "project_rows", "wbs_count", "wbs_names"])
    else:
        funding = (
            positive.pivot_table(
                index=["predicted_challenge", "normalized_funding_office"],
                columns="funding_level",
                values="five_year_total",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
            .rename(columns={"predicted_challenge": "challenge_name"})
        )
        counts = (
            positive.groupby(["predicted_challenge", "normalized_funding_office"], as_index=False)
            .agg(
                project_rows=("programming_record_id", "nunique"),
                wbs_count=("wbs_name", "nunique"),
                wbs_names=("wbs_name", _join_sorted),
            )
            .rename(columns={"predicted_challenge": "challenge_name"})
        )

    claim_keys = (
        claims[["challenge_name", "claimed_office", "claim_role"]]
        .rename(columns={"claimed_office": "normalized_funding_office", "claim_role": "claimed_role"})
        if not claims.empty
        else pd.DataFrame(columns=["challenge_name", "normalized_funding_office", "claimed_role"])
    )
    funding_keys = funding[["challenge_name", "normalized_funding_office"]]
    all_keys = pd.concat(
        [funding_keys, claim_keys[["challenge_name", "normalized_funding_office"]]],
        ignore_index=True,
    ).drop_duplicates()
    if all_keys.empty:
        return pd.DataFrame(columns=output_columns)

    frame = (
        all_keys.merge(funding, on=["challenge_name", "normalized_funding_office"], how="left")
        .merge(counts, on=["challenge_name", "normalized_funding_office"], how="left")
        .merge(claim_keys, on=["challenge_name", "normalized_funding_office"], how="left")
    )
    for level in FUNDING_LEVELS:
        if level not in frame.columns:
            frame[level] = 0.0
        frame[level] = pd.to_numeric(frame[level], errors="coerce").fillna(0.0)
    for column in ["project_rows", "wbs_count"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    frame["wbs_names"] = frame["wbs_names"].fillna("")
    frame["total_funding"] = frame[list(FUNDING_LEVELS)].sum(axis=1)
    frame["role"] = frame.apply(
        lambda row: row["claimed_role"]
        if _clean_text(row.get("claimed_role"))
        else role_map.get((row["challenge_name"], row["normalized_funding_office"]), "contributor"),
        axis=1,
    )
    frame["role_rank"] = frame["role"].map(_role_rank)
    frame["role_label"] = frame["role"].map(_role_label)
    return frame[output_columns].sort_values(
        ["challenge_name", "role_rank", "total_funding"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _office_portfolio_summary(positive: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "normalized_funding_office",
        *FUNDING_LEVELS,
        "total_funding",
        "funded_challenge_count",
        "claimed_challenge_count",
        "lead_claim_count",
        "partner_claim_count",
        "project_rows",
        "wbs_count",
    ]
    if positive.empty:
        funding = pd.DataFrame(columns=["normalized_funding_office", *FUNDING_LEVELS])
        counts = pd.DataFrame(columns=["normalized_funding_office", "funded_challenge_count", "project_rows", "wbs_count"])
    else:
        funding = (
            positive.pivot_table(
                index="normalized_funding_office",
                columns="funding_level",
                values="five_year_total",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
        )
        counts = positive.groupby("normalized_funding_office", as_index=False).agg(
            funded_challenge_count=("predicted_challenge", "nunique"),
            project_rows=("programming_record_id", "nunique"),
            wbs_count=("wbs_name", "nunique"),
        )
    for level in FUNDING_LEVELS:
        if level not in funding.columns:
            funding[level] = 0.0

    if claims.empty:
        claim_counts = pd.DataFrame(columns=["normalized_funding_office", "claimed_challenge_count", "lead_claim_count", "partner_claim_count"])
    else:
        claim_counts = (
            claims.groupby("claimed_office", as_index=False)
            .agg(
                claimed_challenge_count=("challenge_name", "nunique"),
                lead_claim_count=("claim_role", lambda values: sum(_clean_text(value).lower() == "lead" for value in values)),
                partner_claim_count=("claim_role", lambda values: sum(_clean_text(value).lower() == "partner" for value in values)),
            )
            .rename(columns={"claimed_office": "normalized_funding_office"})
        )

    offices = pd.concat(
        [
            funding[["normalized_funding_office"]],
            claim_counts[["normalized_funding_office"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    if offices.empty:
        return pd.DataFrame(columns=output_columns)

    frame = (
        offices.merge(funding, on="normalized_funding_office", how="left")
        .merge(counts, on="normalized_funding_office", how="left")
        .merge(claim_counts, on="normalized_funding_office", how="left")
    )
    for level in FUNDING_LEVELS:
        frame[level] = pd.to_numeric(frame[level], errors="coerce").fillna(0.0)
    for column in [
        "funded_challenge_count",
        "claimed_challenge_count",
        "lead_claim_count",
        "partner_claim_count",
        "project_rows",
        "wbs_count",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    frame["total_funding"] = frame[list(FUNDING_LEVELS)].sum(axis=1)
    return frame[output_columns].sort_values(
        ["funded_challenge_count", "claimed_challenge_count", "total_funding"],
        ascending=False,
    ).reset_index(drop=True)


def _office_challenge_portfolio(
    positive: pd.DataFrame,
    role_map: dict[tuple[str, str], str],
    claims: pd.DataFrame,
) -> pd.DataFrame:
    return _office_challenge_contributions(positive, role_map, claims).sort_values(
        ["normalized_funding_office", "role_rank", "total_funding"],
        ascending=[True, True, False],
    )


def _lead_claim_funding_summary(
    positive: pd.DataFrame,
    claims: pd.DataFrame,
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    leads = claims.loc[claims["claim_role"].eq("lead")].rename(
        columns={"claimed_office": "lead_program_office"}
    )
    if leads.empty:
        return pd.DataFrame(
            columns=[
                "challenge_name",
                "lead_program_office",
                *FUNDING_LEVELS,
                "total_funding",
                "project_rows",
                "wbs_count",
                "collaboration_signal",
            ]
        )

    lead_funding = _office_challenge_contributions(
        positive,
        _claim_role_map(
            leads.rename(columns={"lead_program_office": "claimed_office"})[
                ["challenge_name", "claimed_office", "claim_role"]
            ]
        ),
        leads.rename(columns={"lead_program_office": "claimed_office"})[
            ["challenge_name", "claimed_office", "claim_role"]
        ],
    )
    lead_funding = lead_funding.rename(columns={"normalized_funding_office": "lead_program_office"})
    lead_funding = lead_funding.loc[
        lead_funding["role"].eq("lead"),
        ["challenge_name", "lead_program_office", *FUNDING_LEVELS, "total_funding", "project_rows", "wbs_count"],
    ]
    summary = leads[["challenge_name", "lead_program_office"]].merge(
        lead_funding,
        on=["challenge_name", "lead_program_office"],
        how="left",
    )
    for column in [*FUNDING_LEVELS, "total_funding", "project_rows", "wbs_count"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0)

    contributor_counts = contributions.groupby("challenge_name")["normalized_funding_office"].nunique()
    summary["collaboration_signal"] = summary["challenge_name"].map(
        lambda challenge: "Multi-office mission area"
        if int(contributor_counts.get(challenge, 0)) > 1
        else "Lead-only funding found"
    )
    return summary.sort_values(["total_funding", "challenge_name"], ascending=[False, True]).reset_index(drop=True)


def _office_pair_opportunities(contributions: pd.DataFrame) -> pd.DataFrame:
    if contributions.empty:
        return pd.DataFrame(
            columns=["office_a", "office_b", "shared_challenge_count", "shared_challenges", "pair_total"]
        )

    rows = []
    for challenge, group in contributions.groupby("challenge_name"):
        office_totals = {
            row["normalized_funding_office"]: float(row["total_funding"])
            for _, row in group.iterrows()
        }
        for office_a, office_b in combinations(sorted(office_totals), 2):
            pair_total = office_totals[office_a] + office_totals[office_b]
            if pair_total <= 0:
                continue
            rows.append(
                {
                    "office_a": office_a,
                    "office_b": office_b,
                    "challenge_name": challenge,
                    "pair_challenge_total": pair_total,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["office_a", "office_b", "shared_challenge_count", "shared_challenges", "pair_total"]
        )

    pairs = pd.DataFrame(rows)
    summary = (
        pairs.groupby(["office_a", "office_b"], as_index=False)
        .agg(
            shared_challenge_count=("challenge_name", "nunique"),
            shared_challenges=("challenge_name", _join_sorted),
            pair_total=("pair_challenge_total", "sum"),
        )
        .sort_values(["shared_challenge_count", "pair_total"], ascending=False)
        .reset_index(drop=True)
    )
    return summary


def _unclaimed_evidence(office_challenge_evidence: pd.DataFrame) -> pd.DataFrame:
    if office_challenge_evidence.empty:
        return pd.DataFrame()
    frame = office_challenge_evidence.copy()
    is_claimed = frame["is_claimed"].astype(str).str.casefold().isin({"true", "1", "yes"})
    frame = frame.loc[~is_claimed].copy()
    frame["five_year_total"] = pd.to_numeric(frame["five_year_total"], errors="coerce").fillna(0.0)
    frame["five_year_total_display"] = frame["five_year_total"].map(_format_money)
    frame = frame.sort_values("five_year_total", ascending=False)
    return frame[
        [
            "challenge_name",
            "normalized_funding_office",
            "evidence_row_count",
            "evidence_wbs_count",
            "five_year_total_display",
            "evidence_wbs_names",
        ]
    ]


def _appendix_claim_validation(office_claim_validation: pd.DataFrame) -> pd.DataFrame:
    frame = office_claim_validation.copy()
    if frame.empty:
        return frame
    frame["five_year_total"] = pd.to_numeric(frame["five_year_total"], errors="coerce").fillna(0.0)
    frame["five_year_total_display"] = frame["five_year_total"].map(_format_money)
    columns = [
        "challenge_name",
        "claimed_office",
        "claim_role",
        "validation_status",
        "evidence_row_count",
        "evidence_wbs_count",
        "five_year_total_display",
        "high_confidence_count",
        "medium_confidence_count",
        "low_confidence_count",
        "review_flag_count",
        "evidence_wbs_names",
    ]
    return frame[[column for column in columns if column in frame.columns]]


def _appendix_reviewed_rows(reviewed_classifications: pd.DataFrame) -> pd.DataFrame:
    frame = reviewed_classifications.copy()
    if frame.empty:
        return frame
    frame["five_year_total"] = pd.to_numeric(frame["five_year_total"], errors="coerce").fillna(0.0)
    frame["five_year_total_display"] = frame["five_year_total"].map(_format_money)
    columns = [
        "source_file",
        "source_row_number",
        "fynsp_cycle",
        "wbs_name",
        "normalized_funding_office",
        "funding_level",
        "original_predicted_challenge",
        "predicted_challenge",
        "confidence",
        "review_flag",
        "ai_directness_score",
        "ai_directness_label",
        "ai_directness_confidence",
        "ai_directness_rationale",
        "sme_review_decision",
        "sme_override_applied",
        "sme_notes",
        "sme_reviewer",
        "sme_review_date",
        "five_year_total_display",
        "scope_description",
    ]
    return frame[[column for column in columns if column in frame.columns]].sort_values(
        [column for column in ["predicted_challenge", "normalized_funding_office", "wbs_name"] if column in frame.columns]
    )


def _disruptiveness_rows(positive: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "programming_record_id",
        "source_file",
        "source_row_number",
        "fynsp_cycle",
        "wbs_name",
        "normalized_funding_office",
        "funding_level",
        "predicted_challenge",
        "five_year_total",
        "ai_directness_score",
        "ai_directness_label",
        "ai_directness_confidence",
        "ai_directness_rationale",
        "scope_description",
    ]
    if positive.empty:
        return pd.DataFrame(columns=columns)

    frame = positive.copy()
    frame = frame.loc[
        frame["funding_level"].astype(str).str.upper().isin(FUNDING_LEVELS)
    ].copy()
    frame["five_year_total"] = pd.to_numeric(
        frame["five_year_total"],
        errors="coerce",
    ).fillna(0.0)
    for column, default in {
        "ai_directness_score": 0,
        "ai_directness_label": "not_ai_ml",
        "ai_directness_confidence": "low",
        "ai_directness_rationale": "AI-directness classification was not available.",
        "scope_description": "",
    }.items():
        if column not in frame.columns:
            frame[column] = default

    frame["ai_directness_score"] = (
        pd.to_numeric(frame["ai_directness_score"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=100)
        .astype(int)
    )
    for column in [
        "ai_directness_label",
        "ai_directness_confidence",
        "ai_directness_rationale",
    ]:
        frame[column] = frame[column].map(_clean_text)
    frame["ai_directness_label"] = frame["ai_directness_label"].replace(
        {"": "not_ai_ml"}
    )
    return frame[[column for column in columns if column in frame.columns]].sort_values(
        ["normalized_funding_office", "funding_level", "five_year_total"],
        ascending=[True, True, False],
    )


def _project_disruptiveness_html(rows: pd.DataFrame, *, view_id: str) -> str:
    if rows.empty:
        return "<p>No BASELINE, UFR, or ROT programming rows were found for disruptiveness scoring.</p>"

    payload = _disruptiveness_payload(rows)
    data_json = json.dumps(payload, ensure_ascii=False, sort_keys=True).replace(
        "</",
        "<\\/",
    )
    office_options = _select_options(
        sorted({row["office"] for row in payload["rows"] if row["office"]})
    )
    challenge_options = _select_options(
        sorted({row["challenge"] for row in payload["rows"] if row["challenge"]})
    )
    tile_count = len(payload["rows"])
    mode_buttons = "".join(
        f"<button type=\"button\" data-disruptiveness-mode=\"{level}\">{level}</button>"
        for level in FUNDING_LEVELS
    )
    legend_items = "".join(
        f"<span><span class=\"legend-swatch\" style=\"background:var(--{level.lower()})\"></span>{level} tile</span>"
        for level in FUNDING_LEVELS
    )
    return f"""
    <div id="disruptiveness-tool-{_ha(view_id)}" class="viz-block disruptiveness-tool" data-disruptiveness-row-count="{tile_count}" data-tile-layout="packed-grid" data-x-axis="ai-directness-reversed">
      <div class="tool-controls">
        <div class="tool-control">
          <label>Funding Mode</label>
          <div class="segmented" role="group" aria-label="Funding mode">
            <button type="button" class="active" data-disruptiveness-mode="ALL">ALL FUNDS</button>
            {mode_buttons}
          </div>
        </div>
        <div class="tool-control">
          <label>Program Office</label>
          <select data-disruptiveness-office-filter>
            <option value="">All offices</option>
            {office_options}
          </select>
        </div>
        <div class="tool-control">
          <label>Genesis Mission Challenge</label>
          <select data-disruptiveness-challenge-filter>
            <option value="">All challenges</option>
            {challenge_options}
          </select>
        </div>
      </div>
      <div class="legend">
        {legend_items}
        <span data-disruptiveness-count>{tile_count:,} visible projects</span>
      </div>
      <div class="disruptiveness-layout">
        <svg class="disruptiveness-svg" viewBox="0 0 1040 570" role="img" aria-label="Packed tilemap of project funding disruptiveness">
          <rect x="78" y="36" width="896" height="448" fill="#ffffff" stroke="#d9e2ec"></rect>
          <g stroke="#edf2f7" stroke-width="1">
            <line x1="78" y1="372" x2="974" y2="372"></line>
            <line x1="78" y1="260" x2="974" y2="260"></line>
            <line x1="78" y1="148" x2="974" y2="148"></line>
            <line x1="302" y1="36" x2="302" y2="484"></line>
            <line x1="526" y1="36" x2="526" y2="484"></line>
            <line x1="750" y1="36" x2="750" y2="484"></line>
          </g>
          <g fill="#52606d" font-size="12">
            <text x="72" y="488" text-anchor="end">0%</text>
            <text x="72" y="376" text-anchor="end">25%</text>
            <text x="72" y="264" text-anchor="end">50%</text>
            <text x="72" y="152" text-anchor="end">75%</text>
            <text x="72" y="40" text-anchor="end">100%</text>
            <text x="78" y="510" text-anchor="middle">100</text>
            <text x="302" y="510" text-anchor="middle">75</text>
            <text x="526" y="510" text-anchor="middle">50</text>
            <text x="750" y="510" text-anchor="middle">25</text>
            <text x="974" y="510" text-anchor="middle">0</text>
            <text x="526" y="544" text-anchor="middle">AI-directness score (more directly AI to the left)</text>
            <text x="18" y="256" transform="rotate(-90 18 256)" text-anchor="middle">Share of office funding in selected mode</text>
          </g>
          <g fill="#102a43" font-size="12" font-weight="650">
            <text x="604" y="58">Highest disruption risk: large office share, weak AI fit</text>
            <text x="98" y="466">Lower disruption: small share, direct AI/ML fit</text>
          </g>
          <g data-disruptiveness-tiles></g>
        </svg>
        <aside class="detail-panel" aria-live="polite">
          <h3 data-disruptiveness-detail-title>Select a project tile</h3>
          <div class="detail-grid">
            <span class="label">Office</span><span data-disruptiveness-detail-office>-</span>
            <span class="label">Funding</span><span data-disruptiveness-detail-funding>-</span>
            <span class="label">Amount</span><span data-disruptiveness-detail-amount>-</span>
            <span class="label">Office share</span><span data-disruptiveness-detail-share>-</span>
            <span class="label">AI score</span><span data-disruptiveness-detail-score>-</span>
            <span class="label">AI label</span><span data-disruptiveness-detail-label>-</span>
            <span class="label">Challenge</span><span data-disruptiveness-detail-challenge>-</span>
            <span class="label">Source</span><span data-disruptiveness-detail-source>-</span>
          </div>
          <p class="muted" data-disruptiveness-detail-rationale>Use the filters to isolate an office, challenge, or funding mode.</p>
        </aside>
      </div>
      <script type="application/json" data-disruptiveness-data>{data_json}</script>
      {_project_disruptiveness_script(view_id)}
    </div>
    """


def _disruptiveness_payload(rows: pd.DataFrame) -> dict[str, Any]:
    frame = rows.copy()
    office_level_totals = (
        frame.pivot_table(
            index="normalized_funding_office",
            columns="funding_level",
            values="five_year_total",
            aggfunc="sum",
            fill_value=0.0,
        )
        if not frame.empty
        else pd.DataFrame()
    )
    for level in FUNDING_LEVELS:
        if level not in office_level_totals.columns:
            office_level_totals[level] = 0.0
    office_level_totals["ALL"] = office_level_totals[list(FUNDING_LEVELS)].sum(axis=1)

    payload_rows = []
    for _, row in frame.iterrows():
        office = _clean_text(row.get("normalized_funding_office"))
        funding_level = _clean_text(row.get("funding_level")).upper()
        amount = float(row.get("five_year_total") or 0.0)
        totals = office_level_totals.loc[office] if office in office_level_totals.index else {}
        level_totals = {
            level: float(totals.get(level, 0.0)) if hasattr(totals, "get") else 0.0
            for level in FUNDING_LEVELS
        }
        all_total = float(totals.get("ALL", 0.0)) if hasattr(totals, "get") else 0.0
        level_shares = {
            level.lower(): (
                amount / level_totals[level]
                if funding_level == level and level_totals[level] > 0
                else 0.0
            )
            for level in FUNDING_LEVELS
        }
        payload_rows.append(
            {
                "id": _clean_text(row.get("programming_record_id")),
                "source_file": _clean_text(row.get("source_file")) or "programming_data.csv",
                "source_row_number": _clean_text(row.get("source_row_number")),
                "wbs_name": _clean_text(row.get("wbs_name")) or "Unnamed WBS",
                "office": office,
                "funding_level": funding_level,
                "challenge": _clean_text(row.get("predicted_challenge")),
                "five_year_total": amount,
                "ai_directness_score": int(row.get("ai_directness_score") or 0),
                "ai_directness_label": _clean_text(row.get("ai_directness_label")) or "not_ai_ml",
                "ai_directness_confidence": _clean_text(row.get("ai_directness_confidence")) or "low",
                "ai_directness_rationale": _clean_text(row.get("ai_directness_rationale")),
                "office_total_all": all_total,
                **{f"office_total_{level.lower()}": total for level, total in level_totals.items()},
                "office_share_all": 0.0 if all_total <= 0 else amount / all_total,
                **{f"office_share_{level}": share for level, share in level_shares.items()},
            }
        )
    return {"rows": payload_rows}


def _select_options(values: list[str]) -> str:
    return "".join(f"<option value=\"{_ha(value)}\">{_h(value)}</option>" for value in values)


def _project_disruptiveness_script(view_id: str) -> str:
    return """
      <script>
        (function () {
          const container = document.getElementById('disruptiveness-tool-__VIEW_ID__');
          if (!container) return;
          const dataElement = container.querySelector('[data-disruptiveness-data]');
          const payload = JSON.parse(dataElement.textContent || '{"rows": []}');
          const rows = payload.rows || [];
          const layer = container.querySelector('[data-disruptiveness-tiles]');
          const count = container.querySelector('[data-disruptiveness-count]');
          const officeFilter = container.querySelector('[data-disruptiveness-office-filter]');
          const challengeFilter = container.querySelector('[data-disruptiveness-challenge-filter]');
          const modeButtons = Array.from(container.querySelectorAll('[data-disruptiveness-mode]'));
          const detail = {
            title: container.querySelector('[data-disruptiveness-detail-title]'),
            office: container.querySelector('[data-disruptiveness-detail-office]'),
            funding: container.querySelector('[data-disruptiveness-detail-funding]'),
            amount: container.querySelector('[data-disruptiveness-detail-amount]'),
            share: container.querySelector('[data-disruptiveness-detail-share]'),
            score: container.querySelector('[data-disruptiveness-detail-score]'),
            label: container.querySelector('[data-disruptiveness-detail-label]'),
            challenge: container.querySelector('[data-disruptiveness-detail-challenge]'),
            source: container.querySelector('[data-disruptiveness-detail-source]'),
            rationale: container.querySelector('[data-disruptiveness-detail-rationale]')
          };
          const chart = { left: 78, top: 36, cols: 32, rows: 16, tileSize: 28 };
          let mode = 'ALL';
          let selectedId = '';

          function money(value) {
            const number = Number(value || 0);
            if (Math.abs(number) >= 1000000000) return '$' + (number / 1000000000).toFixed(2) + 'B';
            if (Math.abs(number) >= 1000000) return '$' + (number / 1000000).toFixed(1) + 'M';
            return '$' + Math.round(number).toLocaleString();
          }
          function percent(value) {
            return (100 * Number(value || 0)).toFixed(1) + '%';
          }
          function labelText(value) {
            return String(value || '').replaceAll('_', ' ');
          }
          function rowShare(row) {
            if (mode !== 'ALL') return row['office_share_' + mode.toLowerCase()] || 0;
            return row.office_share_all || 0;
          }
          function rowVisible(row) {
            if (mode !== 'ALL' && row.funding_level !== mode) return false;
            if (officeFilter.value && row.office !== officeFilter.value) return false;
            if (challengeFilter.value && row.challenge !== challengeFilter.value) return false;
            return true;
          }
          function targetCell(row) {
            const score = Math.max(0, Math.min(100, Number(row.ai_directness_score || 0)));
            const share = Math.max(0, Math.min(1, Number(rowShare(row) || 0)));
            return {
              col: Math.round(((100 - score) / 100) * (chart.cols - 1)),
              row: Math.round((1 - share) * (chart.rows - 1))
            };
          }
          function nearestOpenCell(target, occupied) {
            let best = null;
            let bestDistance = Infinity;
            for (let row = 0; row < chart.rows; row += 1) {
              for (let col = 0; col < chart.cols; col += 1) {
                const key = col + ',' + row;
                if (occupied.has(key)) continue;
                const distance = Math.pow(col - target.col, 2) + Math.pow(row - target.row, 2);
                if (
                  distance < bestDistance ||
                  (distance === bestDistance && best && row < best.row) ||
                  (distance === bestDistance && best && row === best.row && col > best.col)
                ) {
                  best = { col: col, row: row, key: key };
                  bestDistance = distance;
                }
              }
            }
            return best || { col: target.col, row: target.row, key: target.col + ',' + target.row };
          }
          function packTiles(visible) {
            const occupied = new Set();
            return visible
              .slice()
              .sort(function (left, right) {
                const leftShare = rowShare(left);
                const rightShare = rowShare(right);
                if (rightShare !== leftShare) return rightShare - leftShare;
                if (left.ai_directness_score !== right.ai_directness_score) return left.ai_directness_score - right.ai_directness_score;
                return Number(right.five_year_total || 0) - Number(left.five_year_total || 0);
              })
              .map(function (row) {
                const target = targetCell(row);
                const placed = nearestOpenCell(target, occupied);
                occupied.add(placed.key);
                return { row: row, target: target, placed: placed };
              });
          }
          function setDetail(row) {
            const share = rowShare(row);
            detail.title.textContent = row.wbs_name || 'Unnamed WBS';
            detail.office.textContent = row.office || '-';
            detail.funding.textContent = row.funding_level || '-';
            detail.amount.textContent = money(row.five_year_total);
            detail.share.textContent = percent(share);
            detail.score.textContent = String(row.ai_directness_score || 0);
            detail.label.textContent = labelText(row.ai_directness_label);
            detail.challenge.textContent = row.challenge || '-';
            detail.source.textContent = (row.source_file || 'programming_data.csv') + ' row ' + (row.source_row_number || '-');
            detail.rationale.textContent = row.ai_directness_rationale || 'No AI-directness rationale available.';
          }
          function render() {
            layer.replaceChildren();
            const visible = rows.filter(rowVisible);
            const packed = packTiles(visible);
            count.textContent = visible.length.toLocaleString() + ' visible project' + (visible.length === 1 ? '' : 's');
            if (visible.length && !visible.some(function (row) { return row.id === selectedId; })) {
              selectedId = visible[0].id || '';
              setDetail(visible[0]);
            }
            packed.forEach(function (tile) {
              const row = tile.row;
              const score = Math.max(0, Math.min(100, Number(row.ai_directness_score || 0)));
              const share = Math.max(0, Math.min(1, Number(rowShare(row) || 0)));
              const x = chart.left + tile.placed.col * chart.tileSize;
              const y = chart.top + tile.placed.row * chart.tileSize;
              const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
              rect.setAttribute('x', x.toFixed(1));
              rect.setAttribute('y', y.toFixed(1));
              rect.setAttribute('width', String(chart.tileSize));
              rect.setAttribute('height', String(chart.tileSize));
              rect.setAttribute('rx', '2');
              rect.setAttribute('tabindex', '0');
              rect.setAttribute('role', 'button');
              rect.setAttribute('class', 'tile-' + String(row.funding_level || 'baseline').toLowerCase() + (row.id === selectedId ? ' tile-selected' : ''));
              rect.setAttribute('opacity', '0.82');
              rect.dataset.projectId = row.id || '';
              rect.dataset.targetCol = String(tile.target.col);
              rect.dataset.targetRow = String(tile.target.row);
              rect.dataset.placedCol = String(tile.placed.col);
              rect.dataset.placedRow = String(tile.placed.row);
              const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
              title.textContent = row.wbs_name + ': ' + money(row.five_year_total) + ', ' + percent(share) + ' of office ' + mode + ', true AI score ' + score;
              rect.appendChild(title);
              rect.addEventListener('click', function () {
                selectedId = row.id || '';
                setDetail(row);
                render();
              });
              rect.addEventListener('focus', function () {
                selectedId = row.id || '';
                setDetail(row);
                render();
              });
              layer.appendChild(rect);
            });
          }
          modeButtons.forEach(function (button) {
            button.addEventListener('click', function () {
              mode = button.dataset.disruptivenessMode || 'ALL';
              modeButtons.forEach(function (item) { item.classList.toggle('active', item === button); });
              render();
            });
          });
          officeFilter.addEventListener('change', render);
          challengeFilter.addEventListener('change', render);
          if (rows.length) setDetail(rows[0]);
          render();
        })();
      </script>
    """.replace("__VIEW_ID__", _ha(view_id))


def _mission_funding_mix_html(challenge_summary: pd.DataFrame) -> str:
    if challenge_summary.empty:
        return "<p>No BASELINE, UFR, or ROT funding rows were found.</p>"

    max_total = max(float(challenge_summary["total_funding"].max()), 1.0)
    rows = []
    for _, row in challenge_summary.iterrows():
        total = float(row["total_funding"])
        funding_values = _funding_values(row)
        width = 3 if total <= 0 else max(6, 100 * total / max_total)
        funding_attrs = " ".join(
            f"data-{level.lower()}-total=\"{funding_values[level]:.2f}\""
            for level in FUNDING_LEVELS
        )
        funding_cells = "".join(
            f"<td class=\"num\">{_format_money(funding_values[level])}</td>"
            for level in FUNDING_LEVELS
        )
        rows.append(
            "<tr "
            f"data-challenge=\"{_ha(row['challenge_name'])}\" "
            f"{funding_attrs} "
            f"data-total=\"{total:.2f}\" "
            f"data-contributor-count=\"{int(row['contributing_office_count'])}\">"
            f"<td>{_h(row['challenge_name'])}</td>"
            f"<td>{_stacked_bar(total, funding_values, width)}</td>"
            f"<td class=\"num\">{_format_money(total)}</td>"
            f"{funding_cells}"
            f"<td class=\"num\">{int(row['contributing_office_count'])}</td>"
            f"<td class=\"num\">{int(row['program_office_player_count'])}</td>"
            f"<td>{_h(row['collaboration_signal'])}</td>"
            "</tr>"
        )
    return (
        "<div class=\"viz-block\"><table class=\"data-table\">"
        "<thead><tr><th>Genesis Mission Challenge</th><th>Funding Mix</th><th>Total</th>"
        f"{_funding_header_html()}<th>Funding Offices</th><th>Office Players</th><th>Collaboration Signal</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _challenge_office_matrix_html(
    contributions: pd.DataFrame,
    challenge_summary: pd.DataFrame,
) -> str:
    if contributions.empty:
        return "<p>No office contribution rows were found.</p>"

    office_order = (
        contributions.groupby("normalized_funding_office")["total_funding"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    challenge_order = challenge_summary["challenge_name"].tolist()
    lookup = {
        (row["challenge_name"], row["normalized_funding_office"]): row
        for _, row in contributions.iterrows()
    }

    headers = "".join(f"<th>{_h(office)}</th>" for office in office_order)
    rows = []
    for _, challenge_row in challenge_summary.iterrows():
        challenge = challenge_row["challenge_name"]
        cells = []
        for office in office_order:
            row = lookup.get((challenge, office))
            if row is None:
                cells.append("<td class=\"muted\">-</td>")
                continue
            total = float(row["total_funding"])
            funding_values = _funding_values(row)
            role = _clean_text(row["role"])
            cells.append(
                "<td>"
                "<div class=\"matrix-cell\">"
                f"<div class=\"cell-total\">{_format_money(total)}</div>"
                f"{_mini_stack(total, funding_values)}"
                f"<div>{_role_pill(role)}</div>"
                f"<div class=\"muted\">{int(row['project_rows'])} rows / {int(row['wbs_count'])} WBS</div>"
                "</div>"
                "</td>"
            )
        rows.append(
            "<tr "
            f"data-challenge=\"{_ha(challenge)}\" "
            f"data-contributor-count=\"{int(challenge_row['contributing_office_count'])}\" "
            f"data-office-player-count=\"{int(challenge_row['program_office_player_count'])}\">"
            f"<td><strong>{_h(challenge)}</strong><br><span class=\"muted\">{_format_money(challenge_row['total_funding'])}</span></td>"
            + "".join(cells)
            + "</tr>"
        )
    return (
        "<div class=\"viz-block\"><table class=\"data-table matrix-table\">"
        f"<thead><tr><th>Challenge</th>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _collaboration_bubble_html(challenge_summary: pd.DataFrame) -> str:
    if challenge_summary.empty:
        return "<p>No challenge funding rows were found.</p>"

    rows = challenge_summary.sort_values(["total_funding", "challenge_name"], ascending=[False, True]).reset_index(drop=True)
    width = 1120
    left = 330
    right = 70
    top = 42
    row_gap = 46
    height = max(170, top + row_gap * len(rows) + 48)
    max_count = max(int(rows["program_office_player_count"].max()), 1)
    max_total = max(float(rows["total_funding"].max()), 1.0)
    axis_width = width - left - right

    tick_count = max_count
    ticks = []
    for tick in range(0, tick_count + 1):
        x = left + (axis_width * tick / max_count)
        ticks.append(
            f"<line x1=\"{x:.1f}\" y1=\"26\" x2=\"{x:.1f}\" y2=\"{height - 34}\" stroke=\"#e5e7eb\" />"
            f"<text x=\"{x:.1f}\" y=\"{height - 12}\" text-anchor=\"middle\" font-size=\"11\" fill=\"#52606d\">{tick}</text>"
        )

    points = []
    for index, row in rows.iterrows():
        funding_count = int(row["contributing_office_count"])
        player_count = int(row["program_office_player_count"])
        total = float(row["total_funding"])
        ufr_share = float(row["ufr_share"])
        x = left + (axis_width * player_count / max_count if max_count else 0)
        y = top + index * row_gap
        radius = 6 + 24 * sqrt(total / max_total) if total > 0 else 5
        color = _ufr_share_color(ufr_share, total)
        points.append(
            f"<text x=\"12\" y=\"{y + 4:.1f}\" font-size=\"12\" fill=\"#1f2933\">{_h(_shorten(row['challenge_name'], 58))}</text>"
            f"<circle cx=\"{x:.1f}\" cy=\"{y:.1f}\" r=\"{radius:.1f}\" fill=\"{color}\" opacity=\"0.82\" "
            f"stroke=\"#ffffff\" stroke-width=\"1.5\" data-challenge=\"{_ha(row['challenge_name'])}\" "
            f"data-total=\"{total:.2f}\" data-ufr-share=\"{ufr_share:.4f}\" />"
            f"<text x=\"{x + radius + 8:.1f}\" y=\"{y + 4:.1f}\" font-size=\"11\" fill=\"#52606d\">"
            f"{_format_money(total)} / {player_count} player{'s' if player_count != 1 else ''} "
            f"({funding_count} funded) / {ufr_share:.0%} UFR/ROT</text>"
        )

    return (
        "<div class=\"viz-block\">"
        f"<svg class=\"svg-chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" "
        "aria-label=\"Bubble chart of collaboration posture by challenge\">"
        f"{''.join(ticks)}"
        f"<text x=\"{left + axis_width / 2:.1f}\" y=\"16\" text-anchor=\"middle\" font-size=\"12\" fill=\"#52606d\">Program office players</text>"
        f"{''.join(points)}"
        "</svg>"
        "<div class=\"legend\"><span><span class=\"legend-swatch\" style=\"background:#3f6f9f\"></span>BASELINE-heavy</span>"
        "<span><span class=\"legend-swatch\" style=\"background:#9b7354\"></span>Mixed</span>"
        "<span><span class=\"legend-swatch\" style=\"background:#d97706\"></span>UFR/ROT-heavy</span></div>"
        "</div>"
    )


def _challenge_collaboration_cards_html(
    challenge_summary: pd.DataFrame,
    contributions: pd.DataFrame,
    claims: pd.DataFrame,
) -> str:
    if challenge_summary.empty:
        return "<p>No challenge collaboration groups were found.</p>"

    claim_groups = {challenge: group for challenge, group in claims.groupby("challenge_name")}
    contribution_groups = {challenge: group for challenge, group in contributions.groupby("challenge_name")}
    cards = []
    for _, challenge_row in challenge_summary.iterrows():
        challenge = challenge_row["challenge_name"]
        claimed = claim_groups.get(challenge, pd.DataFrame(columns=claims.columns))
        actual = contribution_groups.get(challenge, pd.DataFrame(columns=contributions.columns))
        leads = _join_sorted(claimed.loc[claimed["claim_role"].eq("lead"), "claimed_office"])
        partners = _join_sorted(claimed.loc[claimed["claim_role"].eq("partner"), "claimed_office"])
        actual_offices = actual.sort_values(["role_rank", "total_funding"], ascending=[True, False])[
            "normalized_funding_office"
        ].tolist()
        claimed_offices = set(claimed["claimed_office"].tolist())
        other_contributors = [office for office in actual_offices if office not in claimed_offices]
        collaboration_group = _clean_text(challenge_row.get("program_office_players")) or _join_sorted(actual_offices)
        contributor_rows = []
        for _, row in actual.sort_values(["role_rank", "total_funding"], ascending=[True, False]).iterrows():
            contributor_rows.append(
                "<tr>"
                f"<td>{_h(row['normalized_funding_office'])}</td>"
                f"<td>{_role_pill(row['role'])}</td>"
                f"<td class=\"num\">{_format_money(row['total_funding'])}</td>"
                f"<td>{_mini_stack(row['total_funding'], _funding_values(row))}</td>"
                "</tr>"
            )
        if not contributor_rows:
            contributor_table = "<p class=\"muted\">No BASELINE/UFR/ROT funding contributors found.</p>"
        else:
            contributor_table = (
                "<table class=\"data-table\"><thead><tr><th>Office</th><th>Role</th><th>Total</th><th>Mix</th></tr></thead>"
                f"<tbody>{''.join(contributor_rows)}</tbody></table>"
            )
        cards.append(
            "<article class=\"card\" "
            f"data-challenge=\"{_ha(challenge)}\" "
            f"data-contributor-count=\"{int(challenge_row['contributing_office_count'])}\">"
            f"<h3>{_h(challenge)}</h3>"
            f"<p><strong>{_format_money(challenge_row['total_funding'])}</strong> across "
            f"{int(challenge_row['program_office_player_count'])} program office player"
            f"{'s' if int(challenge_row['program_office_player_count']) != 1 else ''}; "
            f"{int(challenge_row['contributing_office_count'])} have direct BASELINE/UFR/ROT funding.</p>"
            f"{_labeled_chips('Lead', leads)}"
            f"{_labeled_chips('Claimed partners', partners)}"
            f"{_labeled_chips('Other funding contributors', _join_sorted(other_contributors))}"
            f"{_labeled_chips('Suggested collaboration group', collaboration_group)}"
            f"{contributor_table}"
            "</article>"
        )
    return f"<div class=\"cards\">{''.join(cards)}</div>"


def _office_pair_opportunities_html(pair_opportunities: pd.DataFrame) -> str:
    if pair_opportunities.empty:
        return "<p>No multi-office funded challenges were found.</p>"

    max_total = max(float(pair_opportunities["pair_total"].max()), 1.0)
    rows = []
    for _, row in pair_opportunities.iterrows():
        width = max(5, 100 * float(row["pair_total"]) / max_total)
        rows.append(
            "<tr "
            f"data-office-a=\"{_ha(row['office_a'])}\" "
            f"data-office-b=\"{_ha(row['office_b'])}\" "
            f"data-shared-challenge-count=\"{int(row['shared_challenge_count'])}\">"
            f"<td><strong>{_h(row['office_a'])} + {_h(row['office_b'])}</strong></td>"
            f"<td class=\"num\">{int(row['shared_challenge_count'])}</td>"
            f"<td>{_h(row['shared_challenges'])}</td>"
            f"<td><div class=\"bar-wrap\"><div class=\"bar-fill\" style=\"width:{width:.1f}%; background:#627d98\"></div></div></td>"
            f"<td class=\"num\">{_format_money(row['pair_total'])}</td>"
            "</tr>"
        )
    return (
        "<div class=\"viz-block\"><table class=\"data-table\">"
        "<thead><tr><th>Office Pair</th><th>Shared Challenges</th><th>Mission Areas</th>"
        "<th>Relative Combined Funding</th><th>Combined Funding</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _office_portfolios_html(
    office_summary: pd.DataFrame,
    office_challenge: pd.DataFrame,
    positive: pd.DataFrame,
) -> str:
    if office_summary.empty:
        return "<p>No office portfolios were found.</p>"

    max_office_total = max(float(office_summary["total_funding"].max()), 1.0)
    cards = []
    for _, office_row in office_summary.iterrows():
        office = office_row["normalized_funding_office"]
        office_challenges = office_challenge.loc[
            office_challenge["normalized_funding_office"].eq(office)
        ].sort_values("total_funding", ascending=False)
        portfolio_rows = []
        for _, row in office_challenges.iterrows():
            width = max(6, 100 * float(row["total_funding"]) / max_office_total)
            portfolio_rows.append(
                "<div class=\"portfolio-row\">"
                f"<div>{_h(_shorten(row['challenge_name'], 72))}<br>{_role_pill(row['role'])}</div>"
                f"<div>{_stacked_bar(row['total_funding'], _funding_values(row), width)}</div>"
                f"<div class=\"num\">{_format_money(row['total_funding'])}</div>"
                "</div>"
            )

        office_rows = positive.loc[positive["normalized_funding_office"].eq(office)].copy()
        top_challenges = office_challenges["challenge_name"].head(3).tolist()
        funded_count = int(office_row["funded_challenge_count"])
        claimed_count = int(office_row["claimed_challenge_count"])
        narrative = (
            f"{_h(office)} has {funded_count} funded Genesis Mission Challenge"
            f"{'s' if funded_count != 1 else ''} and {claimed_count} claimed role"
            f"{'s' if claimed_count != 1 else ''}, with "
            f"{_format_money(office_row['total_funding'])} across {int(office_row['project_rows'])} programming rows. "
            f"Mission areas shown: {_h('; '.join(top_challenges) if top_challenges else 'none')}."
        )
        citations = "".join(f"<li>{_h(citation)}</li>" for citation in _top_citations(office_rows, limit=5))
        if not citations:
            citations = "<li>No direct BASELINE/UFR/ROT programming rows found for this claimed office.</li>"
        funding_metrics = "".join(
            f"<div><strong>{_format_money(office_row[level])}</strong><br>{level}</div>"
            for level in FUNDING_LEVELS
        )
        cards.append(
            "<article class=\"card office-card\" "
            f"data-office=\"{_ha(office)}\" "
            f"data-funded-challenge-count=\"{int(office_row['funded_challenge_count'])}\" "
            f"data-claimed-challenge-count=\"{int(office_row['claimed_challenge_count'])}\" "
            f"data-office-total=\"{float(office_row['total_funding']):.2f}\">"
            f"<h3>{_h(office)}</h3>"
            "<div class=\"office-summary\">"
            f"<div><strong>{int(office_row['funded_challenge_count'])}</strong><br>funded challenges</div>"
            f"<div><strong>{int(office_row['lead_claim_count'])}</strong><br>lead claims</div>"
            f"<div><strong>{int(office_row['partner_claim_count'])}</strong><br>partner claims</div>"
            f"{funding_metrics}"
            f"<div><strong>{int(office_row['wbs_count'])}</strong><br>WBS areas</div>"
            "</div>"
            f"<p>{narrative}</p>"
            f"{''.join(portfolio_rows)}"
            "<details><summary>Programming-data citations</summary>"
            f"<ul class=\"citation-list\">{citations}</ul>"
            "</details>"
            "</article>"
        )
    return f"<div class=\"cards\">{''.join(cards)}</div>"


def _lead_claim_funding_html(lead_summary: pd.DataFrame) -> str:
    if lead_summary.empty:
        return "<p>No claimed lead office rows were found.</p>"

    max_total = max(float(lead_summary["total_funding"].max()), 1.0)
    rows = []
    for _, row in lead_summary.iterrows():
        total = float(row["total_funding"])
        width = 3 if total <= 0 else max(6, 100 * total / max_total)
        funding_values = _funding_values(row)
        funding_cells = "".join(
            f"<td class=\"num\">{_format_money(funding_values[level])}</td>"
            for level in FUNDING_LEVELS
        )
        rows.append(
            "<tr>"
            f"<td>{_h(row['lead_program_office'])}</td>"
            f"<td>{_h(row['challenge_name'])}</td>"
            f"<td>{_stacked_bar(total, funding_values, width)}</td>"
            f"<td class=\"num\">{_format_money(total)}</td>"
            f"{funding_cells}"
            f"<td class=\"num\">{int(row['project_rows'])}</td>"
            f"<td class=\"num\">{int(row['wbs_count'])}</td>"
            f"<td>{_h(row['collaboration_signal'])}</td>"
            "</tr>"
        )
    return (
        "<div class=\"viz-block\"><table class=\"data-table\">"
        "<thead><tr><th>Lead Office</th><th>Genesis Mission Challenge</th><th>Funding Mix</th>"
        f"<th>Total</th>{_funding_header_html()}<th>Rows</th><th>WBS</th><th>Signal</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _challenge_coverage_html(challenge_summary: pd.DataFrame) -> str:
    if challenge_summary.empty:
        return "<p>No challenge coverage rows were found.</p>"

    rows = []
    for _, row in challenge_summary.iterrows():
        funding_values = _funding_values(row)
        funding_cells = "".join(
            f"<td class=\"num\">{_format_money(funding_values[level])}</td>"
            for level in FUNDING_LEVELS
        )
        rows.append(
            "<tr>"
            f"<td>{_h(row['challenge_name'])}</td>"
            f"<td class=\"num\">{_format_money(row['total_funding'])}</td>"
            f"{funding_cells}"
            f"<td class=\"num\">{int(row['program_office_player_count'])}</td>"
            f"<td class=\"num\">{int(row['contributing_office_count'])}</td>"
            f"<td>{_h(row['lead_offices'] or 'None identified')}</td>"
            f"<td>{_h(row['partner_offices'] or 'None identified')}</td>"
            f"<td>{_h(row['actual_funding_offices'] or 'No BASELINE/UFR/ROT funding found')}</td>"
            f"<td>{_h(row['collaboration_signal'])}</td>"
            "</tr>"
        )
    return (
        "<table class=\"data-table\">"
        f"<thead><tr><th>Genesis Mission Challenge</th><th>Total</th>{_funding_header_html()}"
        "<th>Office Players</th><th>Funding Offices</th><th>Claimed Lead</th><th>Claimed Partners</th><th>Actual Contributors</th><th>Signal</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class=\"muted\">No rows.</p>"
    return frame.to_html(index=False, escape=True, border=0, classes="data-table")


def _metadata_html(paths_module: Any, metadata_text: str | None) -> str:
    suffix = "Executive views use BASELINE, UFR, and ROT rows only."
    if metadata_text is not None:
        text = _clean_text(metadata_text)
        separator = "" if text.endswith((".", "!", "?")) else "."
        return f"{_h(text)}{separator} {suffix}"
    return (
        "Generated from Programming Ontology artifacts in "
        f"<code>{_h(str(paths_module.PROCESSED_DATA_DIR))}</code>. {suffix}"
    )


def _funding_values(row: Any) -> dict[str, float]:
    return {
        level: _numeric_value(row.get(level, 0.0) if hasattr(row, "get") else 0.0)
        for level in FUNDING_LEVELS
    }


def _funding_header_html() -> str:
    return "".join(f"<th>{_h(level)}</th>" for level in FUNDING_LEVELS)


def _stacked_bar(total: Any, funding_values: dict[str, float], width: float = 100.0) -> str:
    total = float(total or 0)
    segments = "".join(
        f"<span class=\"{level.lower()}\" style=\"width:{(0.0 if total <= 0 else 100 * funding_values.get(level, 0.0) / total):.1f}%\"></span>"
        for level in FUNDING_LEVELS
    )
    return (
        "<div class=\"bar-wrap\">"
        f"<div class=\"bar-fill\" style=\"width:{width:.1f}%\">"
        f"{segments}"
        "</div></div>"
    )


def _mini_stack(total: Any, funding_values: dict[str, float]) -> str:
    total = float(total or 0)
    segments = "".join(
        f"<span class=\"{level.lower()}\" style=\"width:{(0.0 if total <= 0 else 100 * funding_values.get(level, 0.0) / total):.1f}%\"></span>"
        for level in FUNDING_LEVELS
    )
    return (
        "<div class=\"mini-stack\">"
        f"{segments}"
        "</div>"
    )


def _role_pill(role: Any) -> str:
    role = _clean_text(role).lower()
    class_name = {
        "lead": "role-lead",
        "partner": "role-partner",
    }.get(role, "role-contributor")
    return f"<span class=\"role-pill {class_name}\">{_h(_role_label(role))}</span>"


def _role_label(role: Any) -> str:
    role = _clean_text(role).lower()
    if role == "lead":
        return "Lead"
    if role == "partner":
        return "Partner"
    return "Actual contributor"


def _role_rank(role: Any) -> int:
    role = _clean_text(role).lower()
    if role == "lead":
        return 0
    if role == "partner":
        return 1
    return 2


def _labeled_chips(label: str, csv_values: str) -> str:
    values = [value.strip() for value in _clean_text(csv_values).split(";") if value.strip()]
    if not values:
        values = ["None identified"]
    chips = "".join(f"<span class=\"chip\">{_h(value)}</span>" for value in values)
    return f"<div><strong>{_h(label)}:</strong><div class=\"chip-list\">{chips}</div></div>"


def _top_citations(rows: pd.DataFrame, *, limit: int) -> list[str]:
    if rows.empty:
        return []
    rows = rows.sort_values("five_year_total", ascending=False).head(limit)
    citations = []
    for _, row in rows.iterrows():
        source_file = _clean_text(row.get("source_file")) or "programming_data.csv"
        source_row = _clean_text(row.get("source_row_number"))
        row_ref = f"{source_file} row {source_row}" if source_row else source_file
        challenge = _clean_text(row.get("predicted_challenge"))
        wbs = _clean_text(row.get("wbs_name")) or "Unnamed WBS"
        funding_level = _clean_text(row.get("funding_level"))
        total = _format_money(row.get("five_year_total"))
        citations.append(f"{row_ref}: {wbs}; {challenge}; {funding_level}; {total}")
    return citations


def _metric(label: str, value: str) -> str:
    return f"<div class=\"metric\"><span class=\"label\">{_h(label)}</span><span class=\"value\">{_h(value)}</span></div>"


def _legend() -> str:
    funding_items = "".join(
        f"<span><span class=\"legend-swatch\" style=\"background:var(--{level.lower()})\"></span>{level}</span>"
        for level in FUNDING_LEVELS
    )
    return (
        "<div class=\"legend\">"
        f"{funding_items}"
        "<span><span class=\"role-pill role-lead\">Lead</span></span>"
        "<span><span class=\"role-pill role-partner\">Partner</span></span>"
        "<span><span class=\"role-pill role-contributor\">Actual contributor</span></span>"
        "</div>"
    )


def _collaboration_signal(row: pd.Series) -> str:
    count = int(row.get("contributing_office_count", 0) or 0)
    actual = _clean_text(row.get("actual_funding_offices"))
    players = _clean_text(row.get("program_office_players"))
    player_count = int(row.get("program_office_player_count", 0) or 0)
    if player_count > 1:
        return f"Coordinate {players}"
    if count == 1:
        return f"Single-office execution: {actual}"
    if player_count == 1:
        return f"Claimed mission ownership: {players}"
    return "No BASELINE/UFR/ROT funding found"


def _ufr_share_color(ufr_share: float, total: float) -> str:
    if total <= 0:
        return "#bcccdc"
    if ufr_share >= 0.67:
        return "#d97706"
    if ufr_share >= 0.34:
        return "#9b7354"
    return "#3f6f9f"


def _format_money(value: Any) -> str:
    number = _numeric_value(value)
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000_000:
        return f"{sign}${number / 1_000_000_000:,.2f}B"
    if number >= 1_000_000:
        return f"{sign}${number / 1_000_000:,.1f}M"
    return f"{sign}${number:,.0f}"


def _numeric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _shorten(value: Any, limit: int) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _join_sorted(values: Any) -> str:
    if isinstance(values, pd.Series):
        raw_values = values.tolist()
    else:
        raw_values = list(values) if values is not None and not isinstance(values, str) else [values]
    cleaned = sorted({text for text in (_clean_text(value) for value in raw_values) if text})
    return "; ".join(cleaned)


def _series_text(series: Any) -> list[str]:
    if series is None:
        return []
    try:
        values = series.dropna().tolist()
    except AttributeError:
        return []
    return [text for text in (_clean_text(value) for value in values) if text]


def _split_semicolon_list(value: Any) -> list[str]:
    return [part.strip() for part in _clean_text(value).split(";") if part.strip()]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _h(value: Any) -> str:
    return escape(_clean_text(value), quote=False)


def _ha(value: Any) -> str:
    return escape(_clean_text(value), quote=True)
