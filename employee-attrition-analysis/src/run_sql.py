"""Run the analytical SQL queries from sql/queries.sql against the CSV data.

The two synthetic CSV tables are loaded into an in-memory SQLite database
(pandas.to_sql). The query file is split into statements; for each one the
description (the '-- Q<n>:' comment) and the first rows of the result are
printed.

All data is SYNTHETIC. Nothing is hard-coded: all numbers come from the queries.

Run from the project root (after generating the data):
    python src/data_generator.py
    python src/run_sql.py
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

# Paths are relative to the project root, not to the current working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

TABLES = {
    "employees": "employees.csv",
    "satisfaction_surveys": "satisfaction_surveys.csv",
}
MAX_ROWS = 10  # rows printed per query

DESCRIPTION_PATTERN = re.compile(r"^--\s*(Q\d+:.*)$", re.MULTILINE)


def load_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database and load the CSV files into it."""
    conn = sqlite3.connect(":memory:")
    for table, filename in TABLES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/data_generator.py' first."
            )
        pd.read_csv(path).to_sql(table, conn, index=False, if_exists="replace")
    return conn


def parse_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (description, statement) pairs.

    Statements are separated by semicolons (comments must not contain any).
    The description is the comment line starting with '-- Q<n>:'; all other
    comment lines are removed from the executed statement.
    """
    queries = []
    for chunk in sql_text.split(";"):
        match = DESCRIPTION_PATTERN.search(chunk)
        if match is None:
            continue  # header comments or trailing whitespace
        statement_lines = [
            line for line in chunk.splitlines() if not line.strip().startswith("--")
        ]
        statement = "\n".join(statement_lines).strip()
        if statement:
            queries.append((match.group(1).strip(), statement))
    return queries


def main() -> None:
    """Load the data, run every query and print the first rows of each result."""
    conn = load_database()
    try:
        for table in TABLES:
            n_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Loaded table {table}: {n_rows} rows")

        queries = parse_queries(SQL_FILE.read_text(encoding="utf-8"))
        print(f"Found {len(queries)} queries in {SQL_FILE.name}")

        for description, statement in queries:
            print("\n" + "=" * 70)
            print(description)
            print("=" * 70)
            result = pd.read_sql_query(statement, conn)
            print(result.head(MAX_ROWS).to_string(index=False))
            if len(result) > MAX_ROWS:
                print(f"... ({len(result) - MAX_ROWS} more rows)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
