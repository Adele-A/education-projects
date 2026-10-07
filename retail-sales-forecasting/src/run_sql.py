"""Run the SQL queries from sql/queries.sql on the synthetic sales data.

Loads data/stores.csv and data/weekly_sales.csv into an in-memory SQLite
database (pandas.to_sql), splits sql/queries.sql into statements, runs each
one and prints its description and the first rows of the result.

All data is synthetic. Run from the project root: python src/run_sql.py
(run python src/data_generator.py first to create the CSV files).
"""

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

MAX_ROWS = 10  # rows printed per query

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

DESCRIPTION_PATTERN = re.compile(r"^--\s*(Q\d+:.*)$")


def load_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database with the two CSV tables."""
    stores_path = DATA_DIR / "stores.csv"
    sales_path = DATA_DIR / "weekly_sales.csv"
    for path in (stores_path, sales_path):
        if not path.exists():
            sys.exit(f"Missing {path}. Run 'python src/data_generator.py' first.")

    stores = pd.read_csv(stores_path)
    sales = pd.read_csv(sales_path, parse_dates=["week_start"])
    # Store dates as plain YYYY-MM-DD text so SQLite date functions work.
    sales["week_start"] = sales["week_start"].dt.strftime("%Y-%m-%d")

    conn = sqlite3.connect(":memory:")
    stores.to_sql("stores", conn, index=False)
    sales.to_sql("weekly_sales", conn, index=False)
    return conn


def split_queries(text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (description, statement) pairs.

    Statements end with a semicolon. The description is the comment line
    that starts with '-- Q<n>:'. Statements without such a line are skipped
    (for example the file header).
    """
    queries = []
    for chunk in text.split(";"):
        description = None
        sql_lines = []
        for line in chunk.splitlines():
            match = DESCRIPTION_PATTERN.match(line.strip())
            if match:
                description = match.group(1)
            elif not line.strip().startswith("--"):
                sql_lines.append(line)
        statement = "\n".join(sql_lines).strip()
        if description and statement:
            queries.append((description, statement))
    return queries


def main() -> None:
    """Load the data, run every query and print the first rows."""
    if not SQL_FILE.exists():
        sys.exit(f"Missing {SQL_FILE}.")
    conn = load_database()
    queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"Loaded tables into in-memory SQLite. Running {len(queries)} queries.")

    for description, statement in queries:
        print(f"\n=== {description} ===")
        result = pd.read_sql_query(statement, conn)
        print(f"Rows returned: {len(result)} (showing up to {MAX_ROWS})")
        print(result.head(MAX_ROWS).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
