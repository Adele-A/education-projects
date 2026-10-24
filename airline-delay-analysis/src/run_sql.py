"""Run the analytical SQL queries on the SYNTHETIC airline delay tables.

Loads the CSV files from data/ into an in-memory SQLite database, reads
sql/queries.sql, splits it into statements, runs each one and prints the
query description and the first rows of the result.

The data is synthetic and does not describe real flights or airlines.

Run from the project root (after python src/data_generator.py):
    python src/run_sql.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402

QUERY_FILE = utils.PROJECT_DIR / "sql" / "queries.sql"
MAX_ROWS = 10


def build_database(connection: sqlite3.Connection) -> None:
    """Load the three CSV tables into the SQLite connection."""
    # utils.load_tables keeps the carrier code "NA" from becoming missing
    airports, carriers, flights = utils.load_tables()
    airports.to_sql("airports", connection, index=False)
    carriers.to_sql("carriers", connection, index=False)
    flights.to_sql("flights", connection, index=False)


def parse_queries(text: str) -> list[tuple[str, str]]:
    """Split SQL text into (description, statement) pairs.

    Statements are separated by semicolons. Lines starting with '--' are
    comments and form the description. Comments therefore must not contain
    a semicolon.
    """
    queries = []
    for chunk in text.split(";"):
        comments, sql_lines = [], []
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                comments.append(stripped[2:].strip())
            elif stripped:
                sql_lines.append(line)
        if not sql_lines:
            continue  # trailing comments or blank chunk
        # Use the '-- Q<n>:' line as the description when present
        q_lines = [c for c in comments if c.startswith("Q")]
        description = q_lines[-1] if q_lines else " ".join(comments)
        queries.append((description, "\n".join(sql_lines)))
    return queries


def main() -> None:
    """Load data, run every query and print the first rows."""
    text = QUERY_FILE.read_text(encoding="utf-8")
    queries = parse_queries(text)
    print(f"Loaded {len(queries)} queries from {QUERY_FILE}")

    connection = sqlite3.connect(":memory:")
    try:
        build_database(connection)
        for description, sql in queries:
            print("\n" + "=" * 70)
            print(description)
            print("=" * 70)
            result = pd.read_sql_query(sql, connection)
            print(f"({len(result)} rows, showing up to {MAX_ROWS})")
            print(result.head(MAX_ROWS).to_string(index=False))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
