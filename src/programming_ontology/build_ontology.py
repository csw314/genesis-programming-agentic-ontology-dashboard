"""Build NetworkX ontology graphs from the raw project CSV files.

The module contains small, reusable functions for:

* reading the known CSV files into pandas DataFrames,
* converting each file into a typed NetworkX graph ontology,
* combining all CSV-derived ontologies into one graph, and
* exporting/exploring the resulting graph.

The ontology is represented as a ``networkx.MultiDiGraph``. Nodes carry a
``node_type`` and human-readable ``label``. Edges carry a typed ``relation``.
"""

from __future__ import annotations

from collections import Counter, deque
from hashlib import sha1
from pathlib import Path
import csv
import io
import re
from typing import Any, Iterable

import networkx as nx
import pandas as pd

try:
    from .paths import (
        COMBINED_PROGRAMMING_DATA_PATH,
        MSTC_DATA_PATH,
        PROGRAM_OFFICE_MAPPING_PATH,
        RAW_DATA_DIR,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    COMBINED_PROGRAMMING_DATA_PATH = (
        Path(__file__).resolve().parents[2] / "data" / "processed" / "combined_programming_data.csv"
    )
    MSTC_DATA_PATH = RAW_DATA_DIR / "MSTC_data.csv"
    PROGRAM_OFFICE_MAPPING_PATH = RAW_DATA_DIR / "ProgramOffice_Mapping.csv"


PROGRAMMING_COLUMNS = [
    "Sub Office Number",
    "STAT L3 (Programming)",
    "STAT L4 (Programming)",
    "STAT L5 (Programming)",
    "WBS Name",
    "Funding Levels",
    "Scope Description",
]

PROGRAMMING_EXTRA_NODE_TYPES = {
    "programming_record_id": "programming_record_id",
    "source_file": "programming_source_file",
    "source_row_number": "programming_source_row_number",
    "fynsp_cycle": "fynsp_cycle",
    "fynsp_years": "fynsp_years",
    "FYNSP": "fynsp_years",
    "Submission Type": "submission_type",
    "Scenario": "planning_scenario",
    "WBS": "wbs_code",
    "Program Value": "program_value_code",
    "BNR Subtask Code": "bnr_subtask_code",
    "Fund Value": "fund_value",
    "WBS Level": "wbs_level",
    "Program Int. Area": "program_interest_area",
    "Process Imp. Area": "process_improvement_area",
    "Acquisition Type": "acquisition_type",
    "Acquisition ID": "acquisition_id",
    "Acquisition Name": "acquisition_name",
    "Acquisition Start Date": "acquisition_start_date",
    "Acquisition End Date": "acquisition_end_date",
    "Program Request": "program_request",
    "Account Integrator Priority": "account_integrator_priority",
    "Account Integrator Decision": "account_integrator_decision",
    "Program Priority": "program_priority",
    "DOE Priority Tier": "doe_priority_tier",
    "FY2027": "fy2027_budget",
    "FY2028": "fy2028_budget",
    "FY2029": "fy2029_budget",
    "FY2030": "fy2030_budget",
    "FY2031": "fy2031_budget",
    "FY2032": "fy2032_budget",
}

PROGRAMMING_CLASSIFIER_CONTEXT_COLUMNS = {
    "Program Int. Area",
    "Process Imp. Area",
    "Program Request",
    "Acquisition Type",
    "Acquisition Name",
}

MSTC_COLUMNS = [
    "Agency",
    "Challenge Name",
    "Problem Statement",
    "Proposed Solution",
    "NNSA Advantage",
    "Expected Outcome",
]

MSTC_COLUMN_ALIASES = {
    "Owner": "Agency",
    "Genesis Mission Challenge": "Challenge Name",
    "Challenge": "Problem Statement",
    "AI Solution": "Proposed Solution",
    "Justification": "NNSA Advantage",
    "National Impact": "Expected Outcome",
}

PROGRAM_OFFICE_COLUMNS = [
    "Genesis Mission Science & Tech Challenge",
    "Lead NNSA Organization",
    "Partner/Collaborator Organizations",
]

PROGRAM_OFFICE_COLUMN_ALIASES = {
    "Genesis Mission Challenge": "Genesis Mission Science & Tech Challenge",
    "Lead Program Office": "Lead NNSA Organization",
    "Partner Program Offices": "Partner/Collaborator Organizations",
}

MSTC_FILENAMES = {
    "mstc_data.csv",
}

PROGRAM_OFFICE_FILENAMES = {
    "programoffice_mapping.csv",
}

GENERATED_CSV_FILENAMES = {
    "programming_challenge_classifications.csv",
}

CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin1")

OFFICE_CODE_ALIASES = {
    "NAIM": "NAIM",
    "NAESH": "NAESH",
    "NAPAS": "NA-PAS",
    "NAMB": "NA-MB",
}
CHALLENGE_LEAD_OFFICE_OVERRIDES = {
    "Accelerating Business and Management Processes for National Security Missions": "NA-MB",
    "Accelerating Nuclear Enterprise Acquisition": "NA-PAS",
}


def read_csv(csv_path: str | Path, *, schema: str | None = None) -> pd.DataFrame:
    """Read one CSV file into a cleaned pandas DataFrame.

    ``MSTC_data.csv`` supports both the original headerless format and the
    newer headed format. Other CSV files are read with their first row as the
    header. If you add a future headerless file, pass ``schema="headerless"``
    to generate ``Column 1``, ``Column 2``, etc.
    """

    path = Path(csv_path)
    lower_name = path.name.lower()

    if schema == "mstc" or lower_name in MSTC_FILENAMES:
        return _read_mstc_csv(path)
    elif schema == "program_office" or lower_name in PROGRAM_OFFICE_FILENAMES:
        df = _read_csv_with_fallback(path)
        df = _clean_dataframe(df).rename(columns=PROGRAM_OFFICE_COLUMN_ALIASES)
        return _repair_program_office_dataframe(df)
    elif schema == "headerless" or not _csv_appears_to_have_header(path):
        column_count = _count_csv_columns(path)
        names = [f"Column {index}" for index in range(1, column_count + 1)]
        df = _read_csv_with_fallback(path, header=None, names=names)
    else:
        df = _read_csv_with_fallback(path)

    return _clean_dataframe(df)


def load_csvs(
    folder: str | Path = RAW_DATA_DIR,
    *,
    pattern: str = "*.csv",
) -> dict[str, pd.DataFrame]:
    """Read every matching CSV in ``folder`` and return ``{filename: DataFrame}``."""

    base = Path(folder)
    return {
        csv_path.name: read_csv(csv_path)
        for csv_path in sorted(base.glob(pattern))
        if csv_path.is_file() and csv_path.name.lower() not in GENERATED_CSV_FILENAMES
    }


def build_ontology_from_csv(
    csv_path: str | Path,
    *,
    schema: str | None = None,
) -> nx.MultiDiGraph:
    """Read one CSV file and build an ontology graph for it."""

    path = Path(csv_path)
    df = read_csv(path, schema=schema)
    return build_ontology_from_dataframe(df, source_name=path.name, schema=schema)


def build_ontology_from_dataframe(
    df: pd.DataFrame,
    *,
    source_name: str = "dataframe",
    schema: str | None = None,
) -> nx.MultiDiGraph:
    """Build an ontology graph from one DataFrame.

    Known schemas receive semantic node and edge types. Unknown DataFrames use
    a generic row-column-value ontology so new CSV files still become graphs.
    """

    graph = _new_graph(source_name)

    if schema == "programming" or _has_columns(df, PROGRAMMING_COLUMNS):
        _add_programming_rows(graph, df, source_name)
    elif schema == "mstc" or _has_columns(df, MSTC_COLUMNS):
        _add_mstc_rows(graph, df, source_name)
    elif schema == "program_office" or _has_columns(df, PROGRAM_OFFICE_COLUMNS):
        _add_program_office_rows(graph, df, source_name)
    else:
        _add_generic_rows(graph, df, source_name)

    return graph


def build_ontologies_for_folder(
    folder: str | Path = RAW_DATA_DIR,
    *,
    pattern: str = "*.csv",
) -> dict[str, nx.MultiDiGraph]:
    """Build one ontology graph per CSV file in ``folder``."""

    base = Path(folder)
    return {
        csv_path.name: build_ontology_from_csv(csv_path)
        for csv_path in sorted(base.glob(pattern))
        if csv_path.is_file() and csv_path.name.lower() not in GENERATED_CSV_FILENAMES
    }


def build_combined_ontology(
    folder: str | Path = RAW_DATA_DIR,
    *,
    pattern: str = "*.csv",
) -> nx.MultiDiGraph:
    """Build a single ontology graph by composing all CSV graphs in ``folder``."""

    combined = _new_graph("combined")
    source_files = []

    graphs = (
        build_project_ontologies()
        if Path(folder) == RAW_DATA_DIR and pattern == "*.csv"
        else build_ontologies_for_folder(folder, pattern=pattern)
    )
    for source_name, graph in graphs.items():
        combined = nx.compose(combined, graph)
        source_files.extend(_split_delimited(graph.graph.get("source_files", source_name)))

    combined.graph["name"] = "combined"
    combined.graph["label"] = "Combined CSV Ontology"
    combined.graph["source_files"] = "; ".join(dict.fromkeys(source_files))
    return combined


def build_project_ontologies() -> dict[str, nx.MultiDiGraph]:
    """Build the canonical project graph inputs, including preprocessed programming."""

    from .preprocess_programming import preprocess_programming_data

    preprocess_programming_data()

    return {
        MSTC_DATA_PATH.name: build_ontology_from_csv(MSTC_DATA_PATH, schema="mstc"),
        COMBINED_PROGRAMMING_DATA_PATH.name: build_ontology_from_csv(
            COMBINED_PROGRAMMING_DATA_PATH,
            schema="programming",
        ),
        PROGRAM_OFFICE_MAPPING_PATH.name: build_ontology_from_csv(
            PROGRAM_OFFICE_MAPPING_PATH,
            schema="program_office",
        ),
    }


def summarize_graph(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Return useful counts for a graph."""

    node_types = Counter(
        data.get("node_type", "unknown") for _, data in graph.nodes(data=True)
    )
    edge_relations = Counter(
        data.get("relation", "related_to") for _, _, _, data in graph.edges(keys=True, data=True)
    )

    return {
        "name": graph.graph.get("label", graph.graph.get("name", "ontology")),
        "source_files": graph.graph.get("source_files", ""),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": dict(sorted(node_types.items())),
        "edge_relations": dict(sorted(edge_relations.items())),
    }


def graph_to_tables(graph: nx.MultiDiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert graph nodes and edges into DataFrames for notebook inspection."""

    node_rows = []
    for node_id, data in graph.nodes(data=True):
        node_rows.append({"node_id": node_id, **data})

    edge_rows = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        edge_rows.append(
            {
                "source": source,
                "target": target,
                "key": key,
                **data,
                "source_label": graph.nodes[source].get("label", source),
                "target_label": graph.nodes[target].get("label", target),
            }
        )

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def find_nodes(
    graph: nx.MultiDiGraph,
    text: str,
    *,
    node_type: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find nodes whose label contains ``text``."""

    query = text.casefold()
    matches: list[dict[str, Any]] = []

    for node_id, data in graph.nodes(data=True):
        if node_type and data.get("node_type") != node_type:
            continue

        label = str(data.get("label", ""))
        if query in label.casefold():
            matches.append({"node_id": node_id, **data})

        if len(matches) >= limit:
            break

    return matches


def neighborhood_subgraph(
    graph: nx.MultiDiGraph,
    node_id: str,
    *,
    depth: int = 1,
    direction: str = "both",
) -> nx.MultiDiGraph:
    """Return a subgraph around ``node_id`` for quick visualization."""

    if node_id not in graph:
        raise KeyError(f"Node not found: {node_id}")
    if depth < 0:
        raise ValueError("depth must be zero or greater")
    if direction not in {"both", "predecessors", "successors"}:
        raise ValueError("direction must be 'both', 'predecessors', or 'successors'")

    visited = {node_id}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])

    while queue:
        current, distance = queue.popleft()
        if distance >= depth:
            continue

        neighbors: set[str] = set()
        if direction in {"both", "successors"}:
            neighbors.update(graph.successors(current))
        if direction in {"both", "predecessors"}:
            neighbors.update(graph.predecessors(current))

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return graph.subgraph(visited).copy()


def export_graph(graph: nx.MultiDiGraph, output_path: str | Path) -> Path:
    """Export a graph to GraphML, GEXF, or edge-list CSV based on extension."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".graphml":
        nx.write_graphml(graph, path)
    elif suffix == ".gexf":
        nx.write_gexf(graph, path)
    elif suffix == ".csv":
        _, edges = graph_to_tables(graph)
        edges.to_csv(path, index=False)
    else:
        raise ValueError("Supported export formats: .graphml, .gexf, .csv")

    return path


def _new_graph(source_name: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.graph["name"] = source_name
    graph.graph["label"] = f"{source_name} Ontology"
    graph.graph["source_files"] = source_name
    return graph


def _read_csv_with_fallback(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: UnicodeError | None = None

    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Unable to read CSV: {path}")


def _read_text_with_fallback(path: Path, *, max_chars: int | None = None) -> str:
    last_error: UnicodeError | None = None

    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return handle.read(max_chars) if max_chars is not None else handle.read()
        except UnicodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Unable to read text file: {path}")


def _read_mstc_csv(path: Path) -> pd.DataFrame:
    headed_df = _clean_dataframe(_read_csv_with_fallback(path))
    headed_df = headed_df.rename(columns=MSTC_COLUMN_ALIASES)

    if _has_columns(headed_df, MSTC_COLUMNS):
        return headed_df[MSTC_COLUMNS]

    headerless_df = _read_csv_with_fallback(path, header=None, names=MSTC_COLUMNS)
    return _clean_dataframe(headerless_df)


def _add_programming_rows(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
    source_name: str,
) -> None:
    dataset = _add_concept_node(graph, "dataset", source_name, source_name=source_name)

    for row_number, row in df.iterrows():
        record_source_name = _clean_value(row.get("source_file")) or source_name
        source_row_number = _to_int(row.get("source_row_number")) or int(row_number + 1)
        relation_row_number = source_row_number - 1
        record = _add_record_node(
            graph,
            "programming_record",
            record_source_name,
            relation_row_number,
            explicit_node_id=_clean_value(row.get("programming_record_id")),
            source_row_number=source_row_number,
        )
        _add_relation(
            graph,
            dataset,
            record,
            "contains_record",
            record_source_name,
            relation_row_number,
        )
        _append_graph_source(graph, record_source_name)

        office = _add_concept_node(
            graph, "sub_office", row["Sub Office Number"], source_name=source_name
        )
        stat_l3 = _add_concept_node(
            graph, "stat_l3", row["STAT L3 (Programming)"], source_name=source_name
        )
        stat_l4 = _add_concept_node(
            graph, "stat_l4", row["STAT L4 (Programming)"], source_name=source_name
        )
        stat_l5 = _add_concept_node(
            graph, "stat_l5", row["STAT L5 (Programming)"], source_name=source_name
        )
        wbs = _add_concept_node(graph, "wbs", row["WBS Name"], source_name=source_name)
        funding = _add_concept_node(
            graph, "funding_level", row["Funding Levels"], source_name=source_name
        )
        scope = _add_concept_node(
            graph,
            "scope_description",
            row["Scope Description"],
            source_name=source_name,
        )

        row_edges = [
            (record, office, "belongs_to_sub_office"),
            (record, stat_l3, "classified_by_stat_l3"),
            (record, stat_l4, "classified_by_stat_l4"),
            (record, stat_l5, "classified_by_stat_l5"),
            (record, wbs, "has_wbs"),
            (record, funding, "has_funding_level"),
            (record, scope, "has_scope_description"),
            (office, stat_l3, "has_stat_l3"),
            (stat_l3, stat_l4, "has_stat_l4"),
            (stat_l4, stat_l5, "has_stat_l5"),
            (stat_l5, wbs, "has_wbs"),
            (wbs, funding, "uses_funding_level"),
            (wbs, scope, "described_by_scope"),
        ]
        for source, target, relation in row_edges:
            _add_relation(graph, source, target, relation, record_source_name, relation_row_number)

        for column in _programming_extra_columns(df):
            label = _clean_value(row[column])
            if label is None:
                continue

            slug = _slugify(column)
            attribute = _add_concept_node(
                graph,
                PROGRAMMING_EXTRA_NODE_TYPES.get(column, f"programming_{slug}"),
                label,
                source_name=source_name,
            )
            graph.nodes[attribute]["column_name"] = str(column)
            graph.nodes[attribute]["programming_context"] = "true"
            graph.nodes[attribute]["classification_context"] = str(
                column in PROGRAMMING_CLASSIFIER_CONTEXT_COLUMNS
            ).lower()
            _add_relation(
                graph,
                record,
                attribute,
                f"has_programming_{slug}",
                record_source_name,
                relation_row_number,
            )
            _add_relation(
                graph,
                wbs,
                attribute,
                f"wbs_has_programming_{slug}",
                record_source_name,
                relation_row_number,
            )


def _add_mstc_rows(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
    source_name: str,
) -> None:
    dataset = _add_concept_node(graph, "dataset", source_name, source_name=source_name)

    for row_number, row in df.iterrows():
        record = _add_record_node(graph, "mstc_record", source_name, row_number)
        _add_relation(graph, dataset, record, "contains_record", source_name, row_number)

        agency = _add_concept_node(graph, "agency", row["Agency"], source_name=source_name)
        challenge = _add_concept_node(
            graph, "mstc_challenge", row["Challenge Name"], source_name=source_name
        )
        problem = _add_concept_node(
            graph,
            "problem_statement",
            row["Problem Statement"],
            source_name=source_name,
        )
        solution = _add_concept_node(
            graph,
            "proposed_solution",
            row["Proposed Solution"],
            source_name=source_name,
        )
        advantage = _add_concept_node(
            graph,
            "nnsa_advantage",
            row["NNSA Advantage"],
            source_name=source_name,
        )
        outcome = _add_concept_node(
            graph,
            "expected_outcome",
            row["Expected Outcome"],
            source_name=source_name,
        )

        row_edges = [
            (record, agency, "from_agency"),
            (record, challenge, "addresses_challenge"),
            (record, problem, "has_problem_statement"),
            (record, solution, "has_proposed_solution"),
            (record, advantage, "has_nnsa_advantage"),
            (record, outcome, "has_expected_outcome"),
            (agency, challenge, "sponsors_challenge"),
            (challenge, problem, "has_problem_statement"),
            (challenge, solution, "has_proposed_solution"),
            (challenge, advantage, "has_nnsa_advantage"),
            (challenge, outcome, "has_expected_outcome"),
        ]
        for source, target, relation in row_edges:
            _add_relation(graph, source, target, relation, source_name, row_number)


def _add_program_office_rows(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
    source_name: str,
) -> None:
    dataset = _add_concept_node(graph, "dataset", source_name, source_name=source_name)

    for row_number, row in df.iterrows():
        challenge_label = _clean_value(row["Genesis Mission Science & Tech Challenge"])
        if challenge_label is None:
            continue

        record = _add_record_node(
            graph, "program_office_mapping_record", source_name, row_number
        )
        _add_relation(graph, dataset, record, "contains_record", source_name, row_number)

        challenge = _add_concept_node(
            graph, "mstc_challenge", challenge_label, source_name=source_name
        )
        _add_relation(graph, record, challenge, "maps_challenge", source_name, row_number)

        lead_label = CHALLENGE_LEAD_OFFICE_OVERRIDES.get(
            challenge_label,
            _normalize_office_code(row["Lead NNSA Organization"]),
        )
        if lead_label is not None:
            lead = _add_concept_node(
                graph, "sub_office", lead_label, source_name=source_name
            )
            _add_relation(
                graph, record, lead, "has_lead_program_office", source_name, row_number
            )
            _add_relation(
                graph, challenge, lead, "led_by_program_office", source_name, row_number
            )
            _add_relation(
                graph, lead, challenge, "leads_challenge", source_name, row_number
            )

        for partner_label in _split_office_list(row["Partner/Collaborator Organizations"]):
            partner = _add_concept_node(
                graph, "sub_office", partner_label, source_name=source_name
            )
            _add_relation(
                graph,
                record,
                partner,
                "has_partner_program_office",
                source_name,
                row_number,
            )
            _add_relation(
                graph,
                challenge,
                partner,
                "has_partner_organization",
                source_name,
                row_number,
            )
            _add_relation(
                graph,
                partner,
                challenge,
                "partners_on_challenge",
                source_name,
                row_number,
            )


def _add_generic_rows(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
    source_name: str,
) -> None:
    dataset = _add_concept_node(graph, "dataset", source_name, source_name=source_name)

    column_nodes = {
        column: _add_concept_node(graph, "column", column, source_name=source_name)
        for column in df.columns
    }
    for column_node in column_nodes.values():
        _add_relation(graph, dataset, column_node, "has_column", source_name)

    for row_number, row in df.iterrows():
        record = _add_record_node(graph, "record", source_name, row_number)
        _add_relation(graph, dataset, record, "contains_record", source_name, row_number)

        for column, value in row.items():
            label = _clean_value(value)
            if label is None:
                continue

            value_node = _add_concept_node(
                graph,
                f"value_{_slugify(str(column))}",
                label,
                source_name=source_name,
            )
            relation = f"has_{_slugify(str(column))}"
            _add_relation(graph, record, value_node, relation, source_name, row_number)
            _add_relation(
                graph,
                value_node,
                column_nodes[column],
                "value_in_column",
                source_name,
                row_number,
            )


def _add_concept_node(
    graph: nx.MultiDiGraph,
    node_type: str,
    label: Any,
    *,
    source_name: str,
) -> str:
    clean_label = _clean_value(label)
    if clean_label is None:
        clean_label = "Unknown"

    node_id = _node_id(node_type, clean_label)

    if node_id not in graph:
        graph.add_node(
            node_id,
            label=clean_label,
            node_type=node_type,
            source_files=source_name,
        )
    else:
        _append_node_source(graph, node_id, source_name)

    return node_id


def _add_record_node(
    graph: nx.MultiDiGraph,
    node_type: str,
    source_name: str,
    row_number: int,
    *,
    explicit_node_id: str | None = None,
    source_row_number: int | None = None,
) -> str:
    display_row_number = int(source_row_number if source_row_number is not None else row_number + 1)
    node_id = explicit_node_id or f"{node_type}:{_slugify(source_name)}:{display_row_number}"
    if node_id not in graph:
        graph.add_node(
            node_id,
            label=f"{source_name} row {display_row_number}",
            node_type=node_type,
            source_files=source_name,
            row_number=display_row_number,
        )
    else:
        _append_node_source(graph, node_id, source_name)
    return node_id


def _add_relation(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    relation: str,
    source_name: str,
    row_number: int | None = None,
) -> None:
    edge_key = relation
    row_label = "" if row_number is None else str(row_number + 1)

    if graph.has_edge(source, target, key=edge_key):
        edge = graph.edges[source, target, edge_key]
        edge["weight"] = int(edge.get("weight", 1)) + 1
        edge["source_files"] = _append_delimited(edge.get("source_files", ""), source_name)
        if row_label:
            edge["source_rows"] = _append_delimited(edge.get("source_rows", ""), row_label)
    else:
        graph.add_edge(
            source,
            target,
            key=edge_key,
            relation=relation,
            weight=1,
            source_files=source_name,
            source_rows=row_label,
        )


def _append_node_source(graph: nx.MultiDiGraph, node_id: str, source_name: str) -> None:
    data = graph.nodes[node_id]
    data["source_files"] = _append_delimited(data.get("source_files", ""), source_name)


def _append_graph_source(graph: nx.MultiDiGraph, source_name: str) -> None:
    graph.graph["source_files"] = _append_delimited(
        graph.graph.get("source_files", ""), source_name
    )


def _append_delimited(existing: str, value: str, *, delimiter: str = "; ") -> str:
    values = [item for item in existing.split(delimiter) if item]
    if value not in values:
        values.append(value)
    return delimiter.join(values)


def _split_delimited(value: str, *, delimiter: str = "; ") -> list[str]:
    return [item for item in str(value or "").split(delimiter) if item]


def _node_id(node_type: str, label: str) -> str:
    slug = _slugify(label)
    digest = sha1(label.casefold().encode("utf-8")).hexdigest()[:10]
    if len(slug) > 48:
        slug = slug[:48].strip("_")
    return f"{node_type}:{slug or digest}:{digest}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().casefold())
    return slug.strip("_")


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].map(_clean_value)

    return cleaned.dropna(how="all").reset_index(drop=True)


def _repair_program_office_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        row_data = row.to_dict()
        challenge = _clean_value(row_data.get("Genesis Mission Science & Tech Challenge"))
        lead = _clean_value(row_data.get("Lead NNSA Organization"))
        partners = _clean_value(row_data.get("Partner/Collaborator Organizations"))

        if challenge and lead is None and partners is None and rows:
            previous = rows[-1]
            previous_challenge = previous["Genesis Mission Science & Tech Challenge"]
            separator = " " if str(previous_challenge).endswith(",") else ", "
            previous["Genesis Mission Science & Tech Challenge"] = (
                f"{previous_challenge}{separator}{challenge}"
            )
            continue

        rows.append(row_data)

    return pd.DataFrame(rows, columns=PROGRAM_OFFICE_COLUMNS).reset_index(drop=True)


def _clean_value(value: Any) -> str | None:
    if pd.isna(value):
        return None

    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned or None


def _to_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _normalize_office_code(value: Any) -> str | None:
    label = _clean_value(value)
    if label is None:
        return None

    label = label.upper()
    compact = label.replace("-", "").replace(" ", "")
    if compact.startswith("NA") and compact[2:].isdigit():
        return f"NA{int(compact[2:]):02d}"
    if compact in OFFICE_CODE_ALIASES:
        return OFFICE_CODE_ALIASES[compact]

    return label


def _split_office_list(value: Any) -> list[str]:
    label = _clean_value(value)
    if label is None:
        return []

    offices = []
    for part in label.split(","):
        office = _normalize_office_code(part)
        if office and office not in offices:
            offices.append(office)

    return offices


def _has_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    return set(required_columns).issubset(set(df.columns))


def _programming_extra_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in PROGRAMMING_COLUMNS]


def _csv_appears_to_have_header(path: Path) -> bool:
    sample = _read_text_with_fallback(path, max_chars=4096)

    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        return True


def _count_csv_columns(path: Path) -> int:
    first_row = next(csv.reader(io.StringIO(_read_text_with_fallback(path, max_chars=4096))))
    return len(first_row)


if __name__ == "__main__":
    graph = build_combined_ontology()
    summary = summarize_graph(graph)
    print(f"{summary['name']}: {summary['nodes']} nodes, {summary['edges']} edges")
    print("Node types:", summary["node_types"])
    print("Edge relations:", summary["edge_relations"])
