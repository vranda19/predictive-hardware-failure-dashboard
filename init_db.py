#!/usr/bin/env python3
"""Database initialization and migration script.

Creates the SQLite telemetry table and indexes if they do not exist.
Can be executed directly: `python -m storage.init_db` or `python storage/init_db.py`
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path if executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs import load_config
from storage.db import init_db, get_db_connection, get_telemetry_count


def run_init(db_path: Path | str | None = None) -> None:
    config = load_config()
    target_db = db_path or config.get("storage", {}).get("db_path", "data/telemetry.db")
    print(f"[*] Initializing telemetry SQLite database at: {target_db}")

    init_db(target_db)

    # Verify table creation
    with get_db_connection(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='telemetry';")
        table_exists = cursor.fetchone() is not None

        cursor.execute("PRAGMA table_info(telemetry);")
        columns = [row["name"] for row in cursor.fetchall()]

    if table_exists:
        count = get_telemetry_count(target_db)
        print(f"[+] Success: 'telemetry' table is active.")
        print(f"[+] Verified columns ({len(columns)}): {', '.join(columns)}")
        print(f"[+] Current record count: {count}")
    else:
        print("[-] Error: Failed to create 'telemetry' table.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize telemetry SQLite database.")
    parser.add_argument("--db-path", type=str, default=None, help="Custom database path")
    args = parser.parse_args()

    run_init(args.db_path)
