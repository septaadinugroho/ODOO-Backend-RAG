#!/usr/bin/env python3
"""Validate and load the class corpus.

    python scripts/load_data.py                                  # uses CORPUS_CSV
    python scripts/load_data.py data/sample_corpus.csv
    python scripts/load_data.py path/to/corpus.xlsx
    python scripts/load_data.py --dry-run                        # validate only
    python scripts/load_data.py --replace                        # wipe first

Exit codes let this sit in CI:
    0  loaded (warnings may still be present)
    1  batch rejected — nothing written
    2  cannot reach the database
    3  loaded, but a V8 hard warning fired (fee_rm/schedule had values)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.infrastructure.db import create_all, ping, session_scope  # noqa: E402
from app.core.ingest import (  # noqa: E402
    HARD_WARNING,
    IngestReport,
    load_rows,
    mirror_synonyms,
    read_rows,
    validate,
)
from app.infrastructure.models import ClassRecord, Provider  # noqa: E402
from app.core.synonyms import SynonymDictionary  # noqa: E402


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=settings.corpus_csv,
        help=f"CSV or XLSX to load (default: {settings.corpus_csv})",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate, write nothing")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete existing class_records and providers first",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    if not path.exists():
        print(f"Corpus file not found: {path}", file=sys.stderr)
        print(
            "\nPoint CORPUS_CSV in .env at your file, or pass the path as an "
            "argument. To try the pipeline right now:\n"
            "  python scripts/load_data.py data/sample_corpus.csv",
            file=sys.stderr,
        )
        return 1

    print(f"Reading {path} ...")
    rows, header = read_rows(path)

    report = IngestReport(source=str(path))
    report.rows_read = len(rows)
    accepted = validate(rows, header, report)

    if args.dry_run:
        print("\n=== DRY RUN — nothing written ===")
        report.rows_loaded = len(accepted)
        print(report.summary())
        return 1 if report.batch_rejected else 0

    if report.batch_rejected:
        print("\n=== BATCH REJECTED — nothing written ===", file=sys.stderr)
        print(report.summary(), file=sys.stderr)
        return 1

    if not accepted:
        print("\nNo rows survived validation; nothing to load.", file=sys.stderr)
        print(report.summary(), file=sys.stderr)
        return 1

    if not ping():
        print(
            "Cannot connect to the database. Run `docker compose up -d` first.",
            file=sys.stderr,
        )
        return 2

    create_all()

    with session_scope() as session:
        if args.replace:
            deleted = session.query(ClassRecord).delete()
            session.query(Provider).delete()
            session.flush()
            print(f"Deleted {deleted} existing class records.")

        load_rows(session, accepted, report)

        syn_path = Path(settings.synonyms_path)
        if not syn_path.is_absolute():
            syn_path = Path(__file__).resolve().parent.parent / syn_path
        if syn_path.exists():
            mirrored = mirror_synonyms(session, SynonymDictionary(syn_path).all_entries())
            print(f"Mirrored {mirrored} synonym entries into SQL.")

    print()
    print(report.summary())

    if report.by_severity(HARD_WARNING):
        print(
            "\nV8 fired. fee_rm and/or schedule arrived with values, but they "
            "were loaded as NULL to keep the v1 API contract intact (FR-16, "
            "FR-17). Decide explicitly before exposing them.",
            file=sys.stderr,
        )
        return 3

    print("\nDone. Start the API with: uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
