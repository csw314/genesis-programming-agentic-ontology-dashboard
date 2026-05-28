"""Static HTML report rendering for the classification dashboard."""

from __future__ import annotations

from html import escape
from itertools import combinations
from math import sqrt
from typing import Any

import pandas as pd


FUNDING_LEVELS = ("BASELINE", "UFR")
FY_COLUMNS = ("fy2028", "fy2029", "fy2030", "fy2031", "fy2032")


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

    positive = _positive_funding_rows(reviewed_classifications)
    claims = _claims_frame(office_claim_validation)
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
    unclaimed_evidence = _unclaimed_evidence(office_challenge_evidence)
    appendix_claims = _appendix_claim_validation(office_claim_validation)
    appendix_rows = _appendix_reviewed_rows(reviewed_classifications)

    total_baseline = float(challenge_summary["BASELINE"].sum())
    total_ufr = float(challenge_summary["UFR"].sum())
    total_funding = total_baseline + total_ufr
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
    metadata_html = _metadata_html(paths_module, metadata_text)

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
      .data-table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
<div class="page">
  <h1>Genesis Mission Funding and Collaboration Dashboard</h1>
  <p class="meta">{metadata_html}</p>

  <section>
    <h2>Executive Mission Funding Snapshot</h2>
    <div class="metric-grid">
      {_metric("Total BASELINE + UFR", _format_money(total_funding))}
      {_metric("BASELINE", _format_money(total_baseline))}
      {_metric("UFR", _format_money(total_ufr))}
      {_metric("Funded Challenges", f"{funded_challenges:,}")}
      {_metric("Contributing Offices", f"{contributing_offices:,}")}
      {_metric("Program Office Players", f"{program_office_players:,}")}
      {_metric("Multi-Office Challenges", f"{multi_office_challenges:,}")}
    </div>
  </section>

  <section>
    <h2>Mission Funding Mix</h2>
    <p class="section-note">How much funding is available in BASELINE funds versus UFR funds for each Genesis Mission Challenge?</p>
    {_legend()}
    {_mission_funding_mix_html(challenge_summary)}
  </section>

  <section>
    <h2>Challenge-by-Office Contribution Matrix</h2>
    <p class="section-note">Which program offices contribute BASELINE and UFR funds to each challenge, regardless of whether they are the lead office?</p>
    {_challenge_office_matrix_html(contributions, challenge_summary)}
  </section>

  <section>
    <h2>Collaboration Posture</h2>
    <p class="section-note">Challenges farther right have more program office players, including claimed leads/partners and actual funding contributors; larger circles have more total funding; orange-heavy circles depend more on UFR funds.</p>
    {_collaboration_bubble_html(challenge_summary)}
  </section>

  <section>
    <h2>Challenge Collaboration Groups</h2>
    <p class="section-note">These cards group the offices that should be coordinating around each challenge: claimed leads, claimed partners, and other actual funding contributors.</p>
    {_challenge_collaboration_cards_html(challenge_summary, contributions, claims)}
  </section>

  <section>
    <h2>Office Pair Collaboration Opportunities</h2>
    <p class="section-note">Ranked pairs show offices already funding the same challenge areas and therefore likely candidates for joint execution planning.</p>
    {_office_pair_opportunities_html(pair_opportunities)}
  </section>

  <section>
    <h2>Office Portfolios</h2>
    <p class="section-note">Each office view shows the funded Genesis Mission Challenges in its portfolio, the BASELINE/UFR mix, and programming-data citations for traceability.</p>
    {_office_portfolios_html(office_summary, office_challenge, positive)}
  </section>

  <section>
    <h2>Lead Office Funding by Claimed Mission</h2>
    <p class="section-note">How much funding are lead program offices contributing to their claimed Genesis Mission Challenge?</p>
    {_lead_claim_funding_html(lead_summary)}
  </section>

  <section>
    <h2>Challenge Coverage</h2>
    <p class="section-note">How much funding coverage is there for each Genesis Mission Challenge, and which program offices are contributing?</p>
    {_challenge_coverage_html(challenge_summary)}
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
      {_table_html(appendix_claims)}
    </details>
    <details>
      <summary>Unclaimed Positive Evidence</summary>
      {_table_html(unclaimed_evidence)}
    </details>
    <details>
      <summary>Reviewed Row-Level Classifications</summary>
      {_table_html(appendix_rows)}
    </details>
  </section>
</div>
<script>
  function setAppendix(open) {{
    document.querySelectorAll('.appendix details').forEach(function(detail) {{
      detail.open = open;
    }});
  }}
</script>
</body>
</html>
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
        funding = pd.DataFrame(columns=["challenge_name", "BASELINE", "UFR"])
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
        base_challenges.merge(funding[["challenge_name", "BASELINE", "UFR"]], on="challenge_name", how="left")
        .merge(counts, on="challenge_name", how="left")
    )
    for column in ["BASELINE", "UFR", "project_rows", "wbs_count", "contributing_office_count"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0)
    summary["total_funding"] = summary["BASELINE"] + summary["UFR"]
    summary["ufr_share"] = summary.apply(
        lambda row: 0.0 if row["total_funding"] <= 0 else float(row["UFR"]) / float(row["total_funding"]),
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
        "BASELINE",
        "UFR",
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
    frame["total_funding"] = frame["BASELINE"] + frame["UFR"]
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
        "BASELINE",
        "UFR",
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
    frame["total_funding"] = frame["BASELINE"] + frame["UFR"]
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
                "BASELINE",
                "UFR",
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
        ["challenge_name", "lead_program_office", "BASELINE", "UFR", "total_funding", "project_rows", "wbs_count"],
    ]
    summary = leads[["challenge_name", "lead_program_office"]].merge(
        lead_funding,
        on=["challenge_name", "lead_program_office"],
        how="left",
    )
    for column in ["BASELINE", "UFR", "total_funding", "project_rows", "wbs_count"]:
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
        "wbs_name",
        "normalized_funding_office",
        "funding_level",
        "original_predicted_challenge",
        "predicted_challenge",
        "confidence",
        "review_flag",
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


def _mission_funding_mix_html(challenge_summary: pd.DataFrame) -> str:
    if challenge_summary.empty:
        return "<p>No BASELINE or UFR funding rows were found.</p>"

    max_total = max(float(challenge_summary["total_funding"].max()), 1.0)
    rows = []
    for _, row in challenge_summary.iterrows():
        total = float(row["total_funding"])
        baseline = float(row["BASELINE"])
        ufr = float(row["UFR"])
        width = 3 if total <= 0 else max(6, 100 * total / max_total)
        rows.append(
            "<tr "
            f"data-challenge=\"{_ha(row['challenge_name'])}\" "
            f"data-baseline-total=\"{baseline:.2f}\" "
            f"data-ufr-total=\"{ufr:.2f}\" "
            f"data-total=\"{total:.2f}\" "
            f"data-contributor-count=\"{int(row['contributing_office_count'])}\">"
            f"<td>{_h(row['challenge_name'])}</td>"
            f"<td>{_stacked_bar(total, baseline, ufr, width)}</td>"
            f"<td class=\"num\">{_format_money(total)}</td>"
            f"<td class=\"num\">{_format_money(baseline)}</td>"
            f"<td class=\"num\">{_format_money(ufr)}</td>"
            f"<td class=\"num\">{int(row['contributing_office_count'])}</td>"
            f"<td class=\"num\">{int(row['program_office_player_count'])}</td>"
            f"<td>{_h(row['collaboration_signal'])}</td>"
            "</tr>"
        )
    return (
        "<div class=\"viz-block\"><table class=\"data-table\">"
        "<thead><tr><th>Genesis Mission Challenge</th><th>Funding Mix</th><th>Total</th>"
        "<th>BASELINE</th><th>UFR</th><th>Funding Offices</th><th>Office Players</th><th>Collaboration Signal</th></tr></thead>"
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
            baseline = float(row["BASELINE"])
            ufr = float(row["UFR"])
            role = _clean_text(row["role"])
            cells.append(
                "<td>"
                "<div class=\"matrix-cell\">"
                f"<div class=\"cell-total\">{_format_money(total)}</div>"
                f"{_mini_stack(total, baseline, ufr)}"
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
            f"({funding_count} funded) / {ufr_share:.0%} UFR</text>"
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
        "<span><span class=\"legend-swatch\" style=\"background:#d97706\"></span>UFR-heavy</span></div>"
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
                f"<td>{_mini_stack(row['total_funding'], row['BASELINE'], row['UFR'])}</td>"
                "</tr>"
            )
        if not contributor_rows:
            contributor_table = "<p class=\"muted\">No BASELINE/UFR funding contributors found.</p>"
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
            f"{int(challenge_row['contributing_office_count'])} have direct BASELINE/UFR funding.</p>"
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
                f"<div>{_stacked_bar(row['total_funding'], row['BASELINE'], row['UFR'], width)}</div>"
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
            citations = "<li>No direct BASELINE/UFR programming rows found for this claimed office.</li>"
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
            f"<div><strong>{_format_money(office_row['BASELINE'])}</strong><br>BASELINE</div>"
            f"<div><strong>{_format_money(office_row['UFR'])}</strong><br>UFR</div>"
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
        rows.append(
            "<tr>"
            f"<td>{_h(row['lead_program_office'])}</td>"
            f"<td>{_h(row['challenge_name'])}</td>"
            f"<td>{_stacked_bar(total, row['BASELINE'], row['UFR'], width)}</td>"
            f"<td class=\"num\">{_format_money(total)}</td>"
            f"<td class=\"num\">{_format_money(row['BASELINE'])}</td>"
            f"<td class=\"num\">{_format_money(row['UFR'])}</td>"
            f"<td class=\"num\">{int(row['project_rows'])}</td>"
            f"<td class=\"num\">{int(row['wbs_count'])}</td>"
            f"<td>{_h(row['collaboration_signal'])}</td>"
            "</tr>"
        )
    return (
        "<div class=\"viz-block\"><table class=\"data-table\">"
        "<thead><tr><th>Lead Office</th><th>Genesis Mission Challenge</th><th>Funding Mix</th>"
        "<th>Total</th><th>BASELINE</th><th>UFR</th><th>Rows</th><th>WBS</th><th>Signal</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _challenge_coverage_html(challenge_summary: pd.DataFrame) -> str:
    if challenge_summary.empty:
        return "<p>No challenge coverage rows were found.</p>"

    rows = []
    for _, row in challenge_summary.iterrows():
        rows.append(
            "<tr>"
            f"<td>{_h(row['challenge_name'])}</td>"
            f"<td class=\"num\">{_format_money(row['total_funding'])}</td>"
            f"<td class=\"num\">{_format_money(row['BASELINE'])}</td>"
            f"<td class=\"num\">{_format_money(row['UFR'])}</td>"
            f"<td class=\"num\">{int(row['program_office_player_count'])}</td>"
            f"<td class=\"num\">{int(row['contributing_office_count'])}</td>"
            f"<td>{_h(row['lead_offices'] or 'None identified')}</td>"
            f"<td>{_h(row['partner_offices'] or 'None identified')}</td>"
            f"<td>{_h(row['actual_funding_offices'] or 'No BASELINE/UFR funding found')}</td>"
            f"<td>{_h(row['collaboration_signal'])}</td>"
            "</tr>"
        )
    return (
        "<table class=\"data-table\">"
        "<thead><tr><th>Genesis Mission Challenge</th><th>Total</th><th>BASELINE</th><th>UFR</th>"
        "<th>Office Players</th><th>Funding Offices</th><th>Claimed Lead</th><th>Claimed Partners</th><th>Actual Contributors</th><th>Signal</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class=\"muted\">No rows.</p>"
    return frame.to_html(index=False, escape=True, border=0, classes="data-table")


def _metadata_html(paths_module: Any, metadata_text: str | None) -> str:
    suffix = "Executive views use BASELINE and UFR rows only."
    if metadata_text is not None:
        text = _clean_text(metadata_text)
        separator = "" if text.endswith((".", "!", "?")) else "."
        return f"{_h(text)}{separator} {suffix}"
    return (
        "Generated from Programming Ontology artifacts in "
        f"<code>{_h(str(paths_module.PROCESSED_DATA_DIR))}</code>. {suffix}"
    )


def _stacked_bar(total: Any, baseline: Any, ufr: Any, width: float = 100.0) -> str:
    total = float(total or 0)
    baseline = float(baseline or 0)
    ufr = float(ufr or 0)
    baseline_width = 0.0 if total <= 0 else 100 * baseline / total
    ufr_width = 0.0 if total <= 0 else 100 * ufr / total
    return (
        "<div class=\"bar-wrap\">"
        f"<div class=\"bar-fill\" style=\"width:{width:.1f}%\">"
        f"<span class=\"baseline\" style=\"width:{baseline_width:.1f}%\"></span>"
        f"<span class=\"ufr\" style=\"width:{ufr_width:.1f}%\"></span>"
        "</div></div>"
    )


def _mini_stack(total: Any, baseline: Any, ufr: Any) -> str:
    total = float(total or 0)
    baseline = float(baseline or 0)
    ufr = float(ufr or 0)
    baseline_width = 0.0 if total <= 0 else 100 * baseline / total
    ufr_width = 0.0 if total <= 0 else 100 * ufr / total
    return (
        "<div class=\"mini-stack\">"
        f"<span class=\"baseline\" style=\"width:{baseline_width:.1f}%\"></span>"
        f"<span class=\"ufr\" style=\"width:{ufr_width:.1f}%\"></span>"
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
    return (
        "<div class=\"legend\">"
        "<span><span class=\"legend-swatch\" style=\"background:var(--baseline)\"></span>BASELINE</span>"
        "<span><span class=\"legend-swatch\" style=\"background:var(--ufr)\"></span>UFR</span>"
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
    return "No BASELINE/UFR funding found"


def _ufr_share_color(ufr_share: float, total: float) -> str:
    if total <= 0:
        return "#bcccdc"
    if ufr_share >= 0.67:
        return "#d97706"
    if ufr_share >= 0.34:
        return "#9b7354"
    return "#3f6f9f"


def _format_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000_000:
        return f"{sign}${number / 1_000_000_000:,.2f}B"
    if number >= 1_000_000:
        return f"{sign}${number / 1_000_000:,.1f}M"
    return f"{sign}${number:,.0f}"


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
