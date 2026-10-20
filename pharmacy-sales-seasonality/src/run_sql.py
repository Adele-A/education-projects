"""Run the SQL queries from sql/queries.sql on the SYNTHETIC pharmacy data.

Loads data/categories.csv and data/weekly_sales.csv into an in-memory SQLite
database, splits sql/queries.sql into statements, executes each one and prints
its description (the '-- Q<n>:' comment) and the first rows of the result.

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

MAX_ROWS = 12  # rows printed per query

QUESTION_RE = re.compile(r"^--\s*(Q\d+):\s*(.+)$", re.MULTILINE)


def load_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database with both CSV tables."""
    conn = sqlite3.connect(":memory:")
    categories = pd.read_csv(DATA_DIR / "categories.csv")
    sales = pd.read_csv(DATA_DIR / "weekly_sales.csv")
    # Keep dates as plain 'YYYY-MM-DD' text so SQLite date functions work
    sales["week_start"] = pd.to_datetime(sales["week_start"]).dt.strftime("%Y-%m-%d")
    categories.to_sql("categories", conn, index=False)
    sales.to_sql("weekly_sales", conn, index=False)
    return conn


def split_statements(sql_text: str) -> list[str]:
    """Split the SQL file on semicolons and keep only chunks that contain real SQL."""
    statements = []
    for chunk in sql_text.split(";"):
        code_lines = [ln for ln in chunk.splitlines()
                      if ln.strip() and not ln.strip().startswith("--")]
        if code_lines:
            statements.append(chunk.strip())
    return statements


def describe(statement: str, position: int) -> str:
    """Return 'Q<n>: question' taken from the comment before the statement."""
    match = QUESTION_RE.search(statement)
    if match:
        return f"{match.group(1)}: {match.group(2).strip()}"
    return f"Query {position}"


def main() -> None:
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"Missing {SQL_FILE}")
    conn = load_database()
    statements = split_statements(SQL_FILE.read_text())
    print(f"Loaded tables into in-memory SQLite, running {len(statements)} queries.")

    for i, stmt in enumerate(statements, start=1):
        print("\n" + "=" * 80)
        print(describe(stmt, i))
        print("=" * 80)
        result = pd.read_sql_query(stmt, conn)
        print(result.head(MAX_ROWS).to_string(index=False))
        if len(result) > MAX_ROWS:
            print(f"... ({len(result)} rows in total, first {MAX_ROWS} shown)")
        else:
            print(f"({len(result)} rows)")

    conn.close()


if __name__ == "__main__":
    main()
