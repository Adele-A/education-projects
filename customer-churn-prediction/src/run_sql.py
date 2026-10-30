"""Run the analytical SQL queries from sql/queries.sql.

Loads the SYNTHETIC CSV tables from data/ into an in-memory SQLite database
(using pandas.to_sql), splits the SQL file into statements, runs each one and
prints its description and the first rows of the result.

Run from the project root:  python src/run_sql.py
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
    """Create an in-memory SQLite database with both CSV tables."""
    cust_path = DATA_DIR / "customers.csv"
    tick_path = DATA_DIR / "support_tickets.csv"
    if not cust_path.exists() or not tick_path.exists():
        raise FileNotFoundError(
            "Data files not found. Run `python src/data_generator.py` first."
        )
    conn = sqlite3.connect(":memory:")
    pd.read_csv(cust_path).to_sql("customers", conn, index=False)
    pd.read_csv(tick_path).to_sql("support_tickets", conn, index=False)
    return conn


def has_sql(statement: str) -> bool:
    """Return True if the text contains something other than comments and blanks."""
    lines = [ln.strip() for ln in statement.splitlines()]
    return any(ln and not ln.startswith("--") for ln in lines)


def split_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (description, statement) pairs.

    Statements are separated by semicolons. The description is the comment
    line starting with '-- Q<n>:'.
    """
    queries = []
    for chunk in sql_text.split(";"):
        if not has_sql(chunk):
            continue
        match = DESCRIPTION_PATTERN.search(chunk)
        description = match.group(1) if match else "(no description)"
        queries.append((description, chunk.strip()))
    return queries


def main() -> None:
    """Load the data, run every query and print the results."""
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_FILE}")

    conn = load_database()
    queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"Loaded tables into in-memory SQLite. Running {len(queries)} queries.")

    try:
        for description, statement in queries:
            print("\n" + "=" * 70)
            print(description)
            print("=" * 70)
            result = pd.read_sql_query(statement, conn)
            print(f"({len(result)} rows, showing up to {MAX_ROWS})")
            print(result.head(MAX_ROWS).to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
