"""Preprocess FYNSP programming extracts into one provenance-preserving table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import pandas as pd

from . import paths
from .build_ontology import read_csv
from .fiscal_years import (
    detect_fiscal_year_columns,
    fiscal_year_columns_for_cycles,
    fiscal_year_labels_for_cycle,
)


PROVENANCE_COLUMNS = (
    "programming_record_id",
    "source_file",
    "source_row_number",
    "fynsp_cycle",
    "fynsp_years",
)
BASE_PROGRAMMING_COLUMN_ORDER = (
    *PROVENANCE_COLUMNS,
    "FYNSP",
    "Submission Type",
    "Scenario",
    "Sub Office Number",
    "Funding Levels",
    "STAT L3 (Programming)",
    "STAT L4 (Programming)",
    "STAT L5 (Programming)",
    "WBS",
    "WBS Name",
    "WBS Level",
    "Program Value",
    "BNR Subtask Code",
    "Fund Value",
    "Program Request",
    "Program Int. Area",
    "Process Imp. Area",
    "Scope Description",
    "Acquisition Type",
    "Acquisition ID",
    "Acquisition Name",
    "Acquisition Start Date",
    "Acquisition End Date",
    "Account Integrator Priority",
    "Account Integrator Decision",
    "Program Priority",
    "DOE Priority Tier",
)


@dataclass(frozen=True)
class ProgrammingSource:
    """One raw FYNSP programming extract."""

    path: Path
    fynsp_cycle: str
    fynsp_years: str


DEFAULT_SOURCES = (
    ProgrammingSource(
        paths.PROGRAMMING_DATA_PATH,
        fynsp_cycle="FYNSP 2028",
        fynsp_years="28-32",
    ),
    ProgrammingSource(
        paths.PROGRAMMING_DATA_FYNSP_2027_PATH,
        fynsp_cycle="FYNSP 2027",
        fynsp_years="27-31",
    ),
)


def combine_programming_data(
    sources: Sequence[ProgrammingSource] = DEFAULT_SOURCES,
) -> pd.DataFrame:
    """Return a combined programming table with source-row provenance."""

    frames = [_normalize_source(source) for source in sources if source.path.exists()]
    if not frames:
        raise FileNotFoundError("No programming source files were found.")

    combined = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in frames],
        ignore_index=True,
    )
    for column in _programming_column_order(combined.columns):
        if column not in combined.columns:
            combined[column] = pd.NA
    ordered_columns = [
        column
        for column in _programming_column_order(combined.columns)
        if column in combined.columns
    ]
    ordered_columns.extend(
        column for column in combined.columns if column not in ordered_columns
    )
    return combined[ordered_columns]


def preprocess_programming_data(
    output_path: str | Path = paths.COMBINED_PROGRAMMING_DATA_PATH,
    *,
    sources: Sequence[ProgrammingSource] = DEFAULT_SOURCES,
) -> Path:
    """Write the combined programming CSV and return its path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combine_programming_data(sources).to_csv(output, index=False)
    return output


def _normalize_source(source: ProgrammingSource) -> pd.DataFrame:
    df = read_csv(source.path, schema="programming").copy()
    source_file = source.path.name
    source_rows = pd.Series(range(1, len(df) + 1), index=df.index)

    df["programming_record_id"] = [
        f"programming_record:{_slugify(source_file)}:{row_number}"
        for row_number in source_rows
    ]
    df["source_file"] = source_file
    df["source_row_number"] = source_rows
    df["fynsp_cycle"] = source.fynsp_cycle
    df["fynsp_years"] = source.fynsp_years

    if "FYNSP" not in df.columns:
        df["FYNSP"] = source.fynsp_years
    else:
        df["FYNSP"] = df["FYNSP"].fillna("").astype(str).str.strip()
        df.loc[df["FYNSP"].eq(""), "FYNSP"] = source.fynsp_years

    if "Funding Levels" in df.columns:
        df["Funding Levels"] = df["Funding Levels"].fillna("").astype(str).str.upper().str.strip()

    for column in _source_fiscal_year_columns(source, df.columns):
        if column not in df.columns:
            df[column] = pd.NA

    for column in _programming_column_order(df.columns):
        if column not in df.columns:
            df[column] = pd.NA

    return df


def _source_fiscal_year_columns(
    source: ProgrammingSource,
    columns: Sequence[str],
) -> tuple[str, ...]:
    detected = set(detect_fiscal_year_columns(columns))
    expected = set(fiscal_year_labels_for_cycle(source.fynsp_cycle))
    expected.update(fiscal_year_labels_for_cycle(source.fynsp_years))
    return fiscal_year_columns_for_cycles(
        [*detected, *expected],
        [source.fynsp_cycle, source.fynsp_years],
    )


def _programming_column_order(columns: Sequence[str]) -> tuple[str, ...]:
    fiscal_years = detect_fiscal_year_columns(columns)
    return (*BASE_PROGRAMMING_COLUMN_ORDER, *fiscal_years)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().casefold()).strip("_")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for programming preprocessing."""

    parser = argparse.ArgumentParser(
        description="Combine FYNSP programming extracts into one processed CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.COMBINED_PROGRAMMING_DATA_PATH,
        help="Output CSV path. Defaults to data/processed/combined_programming_data.csv.",
    )
    args = parser.parse_args(argv)

    output = preprocess_programming_data(args.output)
    print(f"Wrote combined programming data: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
