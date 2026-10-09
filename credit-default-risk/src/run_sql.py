"""Run the portfolio SQL queries on the SYNTHETIC credit default data.

Loads data/customers.csv and data/loans.csv into an in-memory SQLite database,
reads sql/queries.sql, runs every query and prints its description and the first rows.

Run from the project root:  python src/run_sql.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

MAX_ROWS = 10  # rows printed per query


def load_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database with the customers and loans tables."""
    conn = sqlite3.connect(":memory:")
    for table in ("customers", "loans"):
        path = DATA_DIR / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/data_generator.py' first.")
        pd.read_csv(path).to_sql(table, conn, index=False)
    return conn


def parse_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (description, statement) pairs.

    Statements are separated by semicolons. The description is the comment line
    starting with '-- Q<n>:'. Statements without such a line (e.g. only comments) are skipped.
    """
    queries = []
    for chunk in sql_text.split(";"):
        description = None
        body_lines = []
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("-- Q"):
                description = stripped[3:].strip()
            elif stripped.startswith("--") or not stripped:
                continue
            else:
                body_lines.append(line)
        if description and body_lines:
            queries.append((description, "\n".join(body_lines)))
    return queries


def main() -> None:
    """Load the data, run all queries and print the results."""
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 30)

    queries = parse_queries(SQL_FILE.read_text(encoding="utf-8"))
    if not queries:
        print("No queries found in sql/queries.sql")
        sys.exit(1)

    conn = load_database()
    try:
        print(f"Loaded tables into in-memory SQLite. Running {len(queries)} queries.")
        for description, statement in queries:
            result = pd.read_sql_query(statement, conn)
            print(f"\n{description}")
            print(f"({len(result)} rows, showing up to {MAX_ROWS})")
            print(result.head(MAX_ROWS).to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
