"""Run the SQL queries from sql/queries.sql on the SYNTHETIC air quality data.

Loads data/stations.csv and data/daily_measurements.csv into an in-memory
SQLite database (pandas.to_sql), splits sql/queries.sql into statements, runs
each one and prints its description and the first rows of the result.

Run from the project root:
    python src/run_sql.py
"""
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_generator import clean_text_files  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

MAX_ROWS = 10  # number of result rows to print per query

QUERY_MARKER = re.compile(r"^--\s*(Q\d+:.*)$")


def load_database():
    """Create an in-memory SQLite database with both CSV tables."""
    clean_text_files(DATA_DIR)  # replace old non-English notes/comments files
    stations_path = DATA_DIR / "stations.csv"
    meas_path = DATA_DIR / "daily_measurements.csv"
    if not stations_path.exists() or not meas_path.exists():
        raise FileNotFoundError(
            f"Data files not found in {DATA_DIR}. Run: python src/data_generator.py"
        )
    conn = sqlite3.connect(":memory:")
    # Dates stay as text (YYYY-MM-DD), which SQLite date functions understand.
    pd.read_csv(stations_path).to_sql("stations", conn, index=False)
    pd.read_csv(meas_path).to_sql("daily_measurements", conn, index=False)
    return conn


def split_queries(sql_text):
    """Split the SQL file into (description, statement) pairs.

    Statements are separated by semicolons. The description is the comment
    line that starts with '-- Q<n>:'. Other comment lines are dropped.
    Chunks without a '-- Q<n>:' line (e.g. trailing whitespace) are skipped.
    """
    queries = []
    for chunk in sql_text.split(";"):
        description = None
        sql_lines = []
        for line in chunk.splitlines():
            stripped = line.strip()
            match = QUERY_MARKER.match(stripped)
            if match:
                description = match.group(1)
            elif stripped.startswith("--") or not stripped:
                continue
            else:
                sql_lines.append(line)
        if description and sql_lines:
            queries.append((description, "\n".join(sql_lines)))
    return queries


def main():
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_FILE}")
    queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
    conn = load_database()
    print(f"Loaded tables into in-memory SQLite. Running {len(queries)} queries.")

    try:
        for description, sql in queries:
            print("\n" + "=" * 70)
            print(description)
            print("=" * 70)
            result = pd.read_sql_query(sql, conn)
            print(f"({len(result)} rows, showing first {min(len(result), MAX_ROWS)})")
            print(result.head(MAX_ROWS).to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
