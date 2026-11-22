"""Run the SQL queries from sql/queries.sql on the SYNTHETIC A/B test data.

Loads data/users.csv and data/orders.csv into an in-memory SQLite database
(pandas.to_sql), splits sql/queries.sql into statements, runs each one and
prints its description and the first rows of the result.

Run from the project root (after data_generator.py):
    python src/run_sql.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

MAX_ROWS = 10  # rows printed per query


def load_database(data_dir: Path = DATA_DIR) -> sqlite3.Connection:
    """Create an in-memory SQLite database with the users and orders tables.

    CSV files are read without date parsing, so dates stay as ISO text
    (YYYY-MM-DD), which SQLite date functions understand.
    """
    conn = sqlite3.connect(":memory:")
    for table in ["users", "orders"]:
        df = pd.read_csv(data_dir / f"{table}.csv")
        df.to_sql(table, conn, index=False)
    return conn


def split_queries(text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (description, sql) pairs.

    Statements are separated by semicolons. The description is the comment
    line starting with '-- Q<n>:' and any comment lines that follow it.
    Chunks without any SQL code are skipped.
    """
    queries = []
    for chunk in text.split(";"):
        description_lines, sql_lines = [], []
        in_description = False
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("-- Q") and ":" in stripped:
                description_lines = [stripped[3:].strip()]
                in_description = True
            elif stripped.startswith("--"):
                # Continuation comments belong to the query description
                if in_description:
                    description_lines.append(stripped[2:].strip())
            elif stripped:
                in_description = False
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip()
        if sql and description_lines:
            queries.append((" ".join(description_lines), sql))
    return queries


def main() -> None:
    conn = load_database()
    queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"Loaded tables into in-memory SQLite. Running {len(queries)} queries "
          f"from {SQL_FILE.relative_to(PROJECT_DIR)}")

    for description, sql in queries:
        print(f"\n{'=' * 70}\n{description}\n{'=' * 70}")
        result = pd.read_sql_query(sql, conn)
        print(result.head(MAX_ROWS).to_string(index=False))
        if len(result) > MAX_ROWS:
            print(f"... ({len(result)} rows in total, first {MAX_ROWS} shown)")
    conn.close()


if __name__ == "__main__":
    main()
