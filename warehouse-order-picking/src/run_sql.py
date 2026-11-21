"""Run the analytical SQL queries on the SYNTHETIC warehouse picking data.

Loads data/pickers.csv and data/orders.csv into an in-memory SQLite database,
reads sql/queries.sql, runs every query and prints its description and the
first rows of the result.

Run from the project root:  python src/run_sql.py
(Run src/data_generator.py first.)
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

# Paths are computed relative to the project root, not the working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

MAX_ROWS = 10  # number of result rows to print per query

# Matches a description line such as "-- Q3: question text"
DESCRIPTION_RE = re.compile(r"^--\s*(Q\d+:.*)$")


def load_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database with the orders and pickers tables."""
    conn = sqlite3.connect(":memory:")
    # Dates are kept as text (YYYY-MM-DD) so SQLite date functions work on them
    pickers = pd.read_csv(DATA_DIR / "pickers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    pickers.to_sql("pickers", conn, index=False)
    orders.to_sql("orders", conn, index=False)
    return conn


def split_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (description, statement) pairs.

    Statements are separated by semicolons. The description is the comment
    line starting with '-- Q<n>:' that precedes the statement.
    """
    queries = []
    for chunk in sql_text.split(";"):
        description = ""
        code_lines = []
        for line in chunk.splitlines():
            stripped = line.strip()
            match = DESCRIPTION_RE.match(stripped)
            if match:
                description = match.group(1)
            elif stripped and not stripped.startswith("--"):
                code_lines.append(line)
        if code_lines:  # skip chunks that contain only comments
            queries.append((description or "(no description)", "\n".join(code_lines)))
    return queries


def main() -> None:
    """Load data, run all queries and print the results."""
    conn = load_database()
    queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"Loaded tables into in-memory SQLite, running {len(queries)} queries")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for description, statement in queries:
            print(f"\n{description}")
            result = pd.read_sql_query(statement, conn)
            if result.empty:
                print("(no rows returned)")
            else:
                print(result.head(MAX_ROWS).to_string(index=False))
                if len(result) > MAX_ROWS:
                    print(f"... {len(result) - MAX_ROWS} more rows")
    conn.close()


if __name__ == "__main__":
    main()
