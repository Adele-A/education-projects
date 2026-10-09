"""Run the SQL queries from sql/queries.sql on the synthetic data.

Loads data/customers.csv and data/subscription_events.csv into an in-memory
SQLite database (pandas.to_sql), splits sql/queries.sql into statements, runs
each one and prints its description and the first rows of the result.

All data is SYNTHETIC, produced by src/data_generator.py.

Run from the project root:
    python src/run_sql.py
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

from utils import load_data

PROJECT_DIR = Path(__file__).resolve().parents[1]
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"
MAX_ROWS = 12  # rows printed per query


def build_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database with both tables."""
    customers, events = load_data()
    # Store dates as plain 'YYYY-MM-DD' text so SQLite date functions work.
    customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")
    events["event_date"] = events["event_date"].dt.strftime("%Y-%m-%d")

    conn = sqlite3.connect(":memory:")
    customers.to_sql("customers", conn, index=False)
    events.to_sql("subscription_events", conn, index=False)
    return conn


def parse_queries(text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (description, statement) pairs.

    Statements end with a semicolon. The description is the comment line
    starting with '-- Q<n>:'. Other comment lines are ignored.
    """
    queries = []
    for chunk in text.split(";"):
        lines = chunk.strip().splitlines()
        desc = next((ln[2:].strip() for ln in lines if re.match(r"--\s*Q\d+:", ln)), None)
        sql = "\n".join(ln for ln in lines if not ln.strip().startswith("--")).strip()
        if sql and desc:
            queries.append((desc, sql))
    return queries


def main() -> None:
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"{SQL_FILE} not found.")
    conn = build_database()
    queries = parse_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(queries)} queries from {SQL_FILE.name}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for desc, sql in queries:
        print(f"\n{'=' * 70}\n{desc}\n{'=' * 70}")
        result = pd.read_sql_query(sql, conn)
        print(f"({len(result)} rows, showing up to {MAX_ROWS})")
        print(result.head(MAX_ROWS).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
