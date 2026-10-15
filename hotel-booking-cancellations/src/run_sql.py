"""Run the SQL analysis on the SYNTHETIC hotel bookings data.

Loads data/guests.csv and data/bookings.csv into an in-memory SQLite database,
reads sql/queries.sql, runs every statement and prints its description and
the first rows of the result.

Run from the project root (after python src/data_generator.py):
    python src/run_sql.py
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

MAX_ROWS = 10  # rows printed per query
DESCRIPTION_PATTERN = re.compile(r"^--\s*(Q\d+:.*)$", re.MULTILINE)


def load_database() -> sqlite3.Connection:
    """Create an in-memory database with the guests and bookings tables."""
    for name in ("guests.csv", "bookings.csv"):
        if not (DATA_DIR / name).exists():
            raise FileNotFoundError(
                f"{DATA_DIR / name} not found. Run 'python src/data_generator.py' first."
            )
    conn = sqlite3.connect(":memory:")
    pd.read_csv(DATA_DIR / "guests.csv").to_sql("guests", conn, index=False)
    pd.read_csv(DATA_DIR / "bookings.csv").to_sql("bookings", conn, index=False)
    return conn


def split_statements(sql_text: str) -> list[str]:
    """Split the file into statements, dropping chunks that contain only comments."""
    statements = []
    for chunk in sql_text.split(";"):
        code_lines = [
            line for line in chunk.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if code_lines:
            statements.append(chunk.strip())
    return statements


def describe(statement: str, position: int) -> str:
    """Return the '-- Q<n>:' description of a statement, or a generic label."""
    match = DESCRIPTION_PATTERN.search(statement)
    return match.group(1).strip() if match else f"Query {position}"


def main() -> None:
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"{SQL_FILE} not found.")
    conn = load_database()
    statements = split_statements(SQL_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(statements)} queries from {SQL_FILE.relative_to(PROJECT_DIR)}")

    try:
        for i, statement in enumerate(statements, start=1):
            print("\n" + "=" * 70)
            print(describe(statement, i))
            print("=" * 70)
            result = pd.read_sql_query(statement, conn)
            if result.empty:
                print("(no rows)")
            else:
                print(result.head(MAX_ROWS).to_string(index=False))
                if len(result) > MAX_ROWS:
                    print(f"... {len(result) - MAX_ROWS} more rows not shown")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
