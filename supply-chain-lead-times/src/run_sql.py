"""Run the SQL queries in sql/queries.sql against the synthetic CSV data.

Loads suppliers.csv and purchase_orders.csv from data/ into an in-memory
SQLite database (pandas.to_sql), splits sql/queries.sql into statements,
runs each one and prints its description and the first rows of the result.
All data is SYNTHETIC.

Run from the project root:
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
QUESTION_RE = re.compile(r"^--\s*Q\d+:", re.IGNORECASE)


def build_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database holding both CSV tables.

    Dates are kept as YYYY-MM-DD text, which SQLite date functions accept.
    """
    suppliers_path = DATA_DIR / "suppliers.csv"
    orders_path = DATA_DIR / "purchase_orders.csv"
    for path in (suppliers_path, orders_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run python src/data_generator.py first."
            )

    conn = sqlite3.connect(":memory:")
    pd.read_csv(suppliers_path).to_sql("suppliers", conn, index=False)
    pd.read_csv(orders_path).to_sql("purchase_orders", conn, index=False)
    return conn


def split_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (description, statement) pairs.

    Statements are separated by semicolons. The description is the line
    starting with '-- Q<n>:'. Chunks without such a line (for example the
    file header) are skipped.
    """
    queries = []
    for chunk in sql_text.split(";"):
        lines = chunk.strip().splitlines()
        start = next((i for i, line in enumerate(lines)
                      if QUESTION_RE.match(line.strip())), None)
        if start is None:
            continue
        description = lines[start].strip().lstrip("-").strip()
        statement = "\n".join(lines[start + 1:]).strip()
        if statement:
            queries.append((description, statement))
    return queries


def main() -> None:
    """Load the data, run every query and print the first rows."""
    conn = build_database()
    queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"Loaded tables into in-memory SQLite. Running {len(queries)} queries "
          f"from {SQL_FILE.name}")

    for description, statement in queries:
        print("\n" + "=" * 80)
        print(description)
        print("=" * 80)
        result = pd.read_sql_query(statement, conn)
        print(f"({len(result)} rows, showing up to {MAX_ROWS})")
        print(result.head(MAX_ROWS).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
