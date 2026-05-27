"""Publish team-shareable Genesis dashboard HTML artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from . import paths
from ._dashboard_report import build_classification_dashboard_html


TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LATEST_FILENAME = "genesis_dashboard_latest.html"


@dataclass(frozen=True)
class PublishedDashboard:
    """Paths created by a dashboard publish run."""

    timestamp: str
    timestamped_html_path: Path
    latest_html_path: Path
    zip_path: Path | None = None


def publish_dashboard(
    *,
    output_dir: str | Path | None = None,
    create_zip: bool = False,
    timestamp: str | None = None,
    metadata: str | None = None,
) -> PublishedDashboard:
    """Publish timestamped and latest dashboard HTML files from processed artifacts."""

    resolved_timestamp, timestamp_dt = _resolve_timestamp(timestamp)
    publish_dir = Path(output_dir) if output_dir is not None else paths.PUBLISH_OUTPUT_DIR
    publish_dir.mkdir(parents=True, exist_ok=True)

    metadata_text = metadata or (
        "Generated from Programming Ontology data snapshot on "
        f"{timestamp_dt:%Y-%m-%d %H:%M}"
    )
    html = _build_publish_html(metadata_text)

    timestamped_html_path = publish_dir / f"genesis_dashboard_{resolved_timestamp}.html"
    latest_html_path = publish_dir / LATEST_FILENAME
    timestamped_html_path.write_text(html, encoding="utf-8")
    latest_html_path.write_text(html, encoding="utf-8")

    zip_path = None
    if create_zip:
        zip_path = publish_dir / f"genesis_dashboard_{resolved_timestamp}.zip"
        _write_zip_bundle(
            zip_path=zip_path,
            html_path=timestamped_html_path,
            metadata_text=metadata_text,
        )

    return PublishedDashboard(
        timestamp=resolved_timestamp,
        timestamped_html_path=timestamped_html_path,
        latest_html_path=latest_html_path,
        zip_path=zip_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Publish the Genesis dashboard as team-shareable static HTML."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.PUBLISH_OUTPUT_DIR,
        help="Directory for published dashboard files. Defaults to outputs/publish.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a zip bundle containing the timestamped HTML and README.",
    )
    parser.add_argument(
        "--timestamp",
        help="Optional deterministic timestamp in YYYYMMDD_HHMMSS format.",
    )
    parser.add_argument(
        "--metadata",
        help="Optional metadata line to show in the published dashboard.",
    )
    args = parser.parse_args(argv)

    published = publish_dashboard(
        output_dir=args.output_dir,
        create_zip=args.zip,
        timestamp=args.timestamp,
        metadata=args.metadata,
    )
    print(f"Published timestamped dashboard: {published.timestamped_html_path}")
    print(f"Updated latest dashboard: {published.latest_html_path}")
    if published.zip_path is not None:
        print(f"Created zip bundle: {published.zip_path}")
    return 0


def _build_publish_html(metadata_text: str) -> str:
    return build_classification_dashboard_html(
        paths_module=paths,
        reviewed_classifications=pd.read_csv(
            paths.REVIEWED_PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH
        ),
        office_claim_validation=pd.read_csv(paths.OFFICE_CLAIM_VALIDATION_PATH),
        challenge_coverage=pd.read_csv(paths.CHALLENGE_COVERAGE_SUMMARY_PATH),
        office_challenge_evidence=pd.read_csv(paths.OFFICE_CHALLENGE_EVIDENCE_PATH),
        metadata_text=metadata_text,
    )


def _resolve_timestamp(timestamp: str | None) -> tuple[str, datetime]:
    if timestamp is None:
        timestamp_dt = datetime.now()
        return timestamp_dt.strftime(TIMESTAMP_FORMAT), timestamp_dt

    try:
        timestamp_dt = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    except ValueError as error:
        raise ValueError(
            "timestamp must use YYYYMMDD_HHMMSS format, for example 20260527_134500"
        ) from error
    return timestamp, timestamp_dt


def _write_zip_bundle(
    *,
    zip_path: Path,
    html_path: Path,
    metadata_text: str,
) -> None:
    readme = _readme_text(html_path.name, metadata_text)
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.write(html_path, arcname=html_path.name)
        archive.writestr("README.txt", readme)


def _readme_text(html_filename: str, metadata_text: str) -> str:
    return f"""Genesis Mission Funding and Collaboration Dashboard

Open {html_filename} in a modern web browser.

This HTML dashboard is self-contained. Team members do not need Python, the
Programming Ontology repository, or access to the local data/processed folder to
view it.

Snapshot:
{metadata_text}

Recommended sharing workflow:
1. Publish the dashboard.
2. Upload {html_filename} or this zip bundle to the team-accessible location.
3. Replace the shared latest copy whenever the underlying programming artifacts
   are refreshed.
"""


if __name__ == "__main__":
    raise SystemExit(main())
