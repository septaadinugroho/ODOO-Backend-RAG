#!/usr/bin/env python3
"""Create the schema and seed the closed-vocabulary dimension tables.

    python scripts/init_db.py              # create missing tables, then seed
    python scripts/init_db.py --drop       # drop everything first (destructive)
    python scripts/init_db.py --check      # connectivity + row counts only

Safe to re-run: table creation is `CREATE TABLE IF NOT EXISTS`, and seeding
inserts only values that are not already present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.infrastructure.db import create_all, drop_all, ping, session_scope  # noqa: E402
from app.core.ingest import mirror_synonyms, seed_vocabulary  # noqa: E402
from app.infrastructure.models import (  # noqa: E402
    ClassRecord,
    Curriculum,
    DeliveryMode,
    InstructionLanguage,
    Level,
    Provider,
    Subject,
    Synonym,
)
from app.core.synonyms import SynonymDictionary  # noqa: E402

TABLES = (
    ("subjects", Subject),
    ("levels", Level),
    ("delivery_modes", DeliveryMode),
    ("instruction_languages", InstructionLanguage),
    ("curricula", Curriculum),
    ("providers", Provider),
    ("class_records", ClassRecord),
    ("synonyms", Synonym),
)


def counts() -> None:
    with session_scope() as session:
        print("\nRow counts:")
        for name, model in TABLES:
            pk = list(model.__table__.primary_key.columns)[0]
            total = session.execute(select(func.count()).select_from(model)).scalar()
            print(f"  {name:24s} {total}")
            del pk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop", action="store_true", help="DROP all tables first")
    parser.add_argument("--check", action="store_true", help="only report status")
    args = parser.parse_args()

    settings = get_settings()
    # Mask the password so the URL is safe to show in a terminal or a demo.
    shown = settings.database_url
    if "@" in shown and "//" in shown:
        head, tail = shown.split("//", 1)
        creds, host = tail.split("@", 1)
        user = creds.split(":", 1)[0]
        shown = f"{head}//{user}:***@{host}"
    print(f"Database: {shown}")

    if not ping():
        print(
            "\nCannot connect. Start PostgreSQL first:\n"
            "  docker compose up -d\n"
            "Then check DATABASE_URL in .env (this project uses port 5433 so it "
            "does not clash with an existing Postgres on 5432).",
            file=sys.stderr,
        )
        return 2

    print("Connection OK.")

    if args.check:
        counts()
        return 0

    if args.drop:
        confirm = input("This DROPS every table. Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 1
        drop_all()
        print("Dropped all tables.")

    create_all()
    print("Schema created (existing tables left alone).")

    with session_scope() as session:
        seeded = seed_vocabulary(session)
        print("Seeded closed vocabulary:")
        for name, added in seeded.items():
            print(f"  {name:24s} +{added}")

        path = Path(settings.synonyms_path)
        if path.exists():
            dictionary = SynonymDictionary(path)
            mirrored = mirror_synonyms(session, dictionary.all_entries())
            print(f"  synonyms (mirror)        {mirrored} from {path}")
        else:
            print(f"  synonyms                 skipped, {path} not found")

    counts()
    print("\nNext: python scripts/load_data.py data/sample_corpus.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
