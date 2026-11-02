"""Run the SQL queries from sql/queries.sql on the synthetic CSV tables.

The CSV files are loaded into an in-memory SQLite database with pandas.to_sql.
The query file is split into statements, each statement is executed, and its
description (the "-- Q<n>:" comment) and the first rows of the result are printed.

Run from the project root:
    python src/run_sql.py
"""

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

MAX_ROWS = 10  # number of result rows to print per query

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

QUESTION_PATTERN = re.compile(r"^--\s*(Q\d+):\s*(.*)$")


def load_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database from the two CSV files."""
    products_path = DATA_DIR / "products.csv"
    sales_path = DATA_DIR / "weekly_sales.csv"
    if not products_path.exists() or not sales_path.exists():
        sys.exit("Data files not found. Run `python src/data_generator.py` first.")

    conn = sqlite3.connect(":memory:")
    # Dates stay as ISO text (YYYY-MM-DD), which sorts correctly in SQLite.
    pd.read_csv(products_path).to_sql("products", conn, index=False)
    pd.read_csv(sales_path).to_sql("weekly_sales", conn, index=False)
    return conn


def split_statements(sql_text: str) -> list[str]:
    """Split the SQL file into statements on semicolons.

    Comment lines must not contain semicolons. Chunks that only hold
    comments or whitespace are skipped.
    """
    statements = []
    for chunk in sql_text.split(";"):
        code_lines = [ln for ln in chunk.splitlines() if ln.strip() and not ln.strip().startswith("--")]
        if code_lines:
            statements.append(chunk.strip())
    return statements


def describe(statement: str, position: int) -> str:
    """Return the '-- Q<n>: ...' description of a statement."""
    for line in statement.splitlines():
        match = QUESTION_PATTERN.match(line.strip())
        if match:
            return f"{match.group(1)}: {match.group(2)}"
    return f"Query {position} (no description)"


def main() -> None:
    if not SQL_FILE.exists():
        sys.exit(f"SQL file not found: {SQL_FILE}")

    conn = load_database()
    try:
        for table in ["products", "weekly_sales"]:
            n_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Loaded table {table}: {n_rows} rows")

        statements = split_statements(SQL_FILE.read_text(encoding="utf-8"))
        print(f"Found {len(statements)} queries in {SQL_FILE.relative_to(PROJECT_DIR)}")

        for i, statement in enumerate(statements, start=1):
            print(f"\n{'=' * 70}\n{describe(statement, i)}\n{'=' * 70}")
            try:
                result = pd.read_sql_query(statement, conn)
            except Exception as exc:  # report the problem and continue with the next query
                print(f"Query failed: {exc}")
                continue
            print(f"Rows returned: {len(result)} (showing up to {MAX_ROWS})")
            print(result.head(MAX_ROWS).to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
