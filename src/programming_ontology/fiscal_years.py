"""Helpers for FYNSP fiscal-year amount columns."""

from __future__ import annotations

import re
from typing import Iterable, Any


FISCAL_YEAR_RE = re.compile(r"^fy(?P<year>\d{4})$", re.IGNORECASE)
FYNSP_CYCLE_RE = re.compile(r"\bFYNSP\s+(?P<year>\d{2}|\d{4})\b", re.IGNORECASE)
FYNSP_RANGE_RE = re.compile(r"(?P<start>\d{2}|\d{4})\s*[-/]\s*(?P<end>\d{2}|\d{4})")


def fiscal_year_label(column: Any) -> str:
    """Return the canonical ``FY####`` label for a fiscal-year column."""

    match = FISCAL_YEAR_RE.match(str(column).strip())
    return f"FY{match.group('year')}" if match else ""


def fiscal_year_key(column: Any) -> str:
    """Return the normalized lower-case key for a fiscal-year column."""

    label = fiscal_year_label(column)
    return label.casefold() if label else ""


def is_fiscal_year_column(column: Any) -> bool:
    """Return whether a column name is shaped like ``FY####``."""

    return bool(fiscal_year_label(column))


def detect_fiscal_year_columns(columns: Iterable[Any]) -> tuple[str, ...]:
    """Return fiscal-year columns sorted by actual fiscal year."""

    found = {
        fiscal_year_label(column): str(column)
        for column in columns
        if fiscal_year_label(column)
    }
    return tuple(found[label] for label in sorted(found, key=_label_year))


def normalized_fiscal_year_columns(columns: Iterable[Any]) -> tuple[str, ...]:
    """Return detected fiscal-year columns normalized to ``fy####`` keys."""

    return tuple(fiscal_year_key(column) for column in detect_fiscal_year_columns(columns))


def fiscal_year_labels(columns: Iterable[Any]) -> tuple[str, ...]:
    """Return canonical ``FY####`` labels for fiscal-year columns."""

    return tuple(fiscal_year_label(column) for column in detect_fiscal_year_columns(columns))


def fiscal_year_labels_for_cycle(value: Any, *, span: int = 5) -> tuple[str, ...]:
    """Return expected FY labels for a value like ``FYNSP 2028`` or ``28-32``."""

    text = str(value or "").strip()
    if not text:
        return ()

    match = FYNSP_CYCLE_RE.search(text)
    if match:
        start_year = _full_year(match.group("year"))
        return tuple(f"FY{year}" for year in range(start_year, start_year + span))

    match = FYNSP_RANGE_RE.search(text)
    if match:
        start_year = _full_year(match.group("start"))
        end_year = _full_year(match.group("end"))
        if end_year < start_year:
            end_year += 100
        return tuple(f"FY{year}" for year in range(start_year, end_year + 1))

    return ()


def fiscal_year_columns_for_cycles(
    columns: Iterable[Any],
    cycles: Iterable[Any],
) -> tuple[str, ...]:
    """Return available fiscal-year columns implied by one or more FYNSP cycles."""

    available = {
        fiscal_year_label(column): str(column)
        for column in columns
        if fiscal_year_label(column)
    }
    desired = {
        label
        for cycle in cycles
        for label in fiscal_year_labels_for_cycle(cycle)
    }
    if desired:
        selected = [label for label in sorted(desired, key=_label_year) if label in available]
        if selected:
            return tuple(available[label] for label in selected)
    return tuple(available[label] for label in sorted(available, key=_label_year))


def _label_year(label: str) -> int:
    return int(label.removeprefix("FY"))


def _full_year(value: str) -> int:
    year = int(value)
    return 2000 + year if year < 100 else year
