"""Run the SQL queries from sql/queries.sql on the SYNTHETIC energy data.

Run from the project root (after src/data_generator.py):
    python src/run_sql.py

The three CSV files in data/ are loaded into an in-memory SQLite database
with pandas.to_sql. sql/queries.sql is split into statements, and for each
one the '-- Q<n>:' description and the first rows of the result are printed.
Nothing is written to disk.
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

TABLES = ["households", "weather_hourly", "energy_hourly"]
MAX_ROWS = 10  # number of result rows to print per query

# A description line looks like: -- Q3: some question
DESCRIPTION_RE = re.compile(r"^--\s*(Q\d+):\s*(.+)$")

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def load_database() -> sqlite3.Connection:
    """Create an in-memory database and load every CSV table into it.

    Timestamps are kept as text so SQLite date functions can parse them.
    """
    conn = sqlite3.connect(":memory:")
    for name in TABLES:
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        df.to_sql(name, conn, index=False, if_exists="replace")
        print(f"loaded {name}: {len(df)} rows")
    return conn


def parse_queries(text: str) -> list[tuple[str, str, str]]:
    """Split the SQL file into (query id, description, sql) tuples.

    Statements are separated by a semicolon, so comments must not use one.
    Comment lines are removed from the SQL that is executed.
    """
    queries = []
    for chunk in text.split(";"):
        query_id, description, sql_lines = None, "", []
        for line in chunk.splitlines():
            stripped = line.strip()
            match = DESCRIPTION_RE.match(stripped)
            if match:
                query_id, description = match.group(1), match.group(2)
            elif stripped and not stripped.startswith("--"):
                sql_lines.append(line)
        if query_id and sql_lines:
            queries.append((query_id, description, "\n".join(sql_lines)))
    return queries


def main() -> None:
    conn = load_database()
    queries = parse_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"\nfound {len(queries)} queries in sql/queries.sql")

    for query_id, description, sql in queries:
        print(f"\n=== {query_id}: {description} ===")
        try:
            result = pd.read_sql_query(sql, conn)
        except Exception as exc:  # report the problem and continue with the next query
            print(f"query failed: {exc}")
            continue
        print(f"rows returned: {len(result)} (showing up to {MAX_ROWS})")
        print(result.head(MAX_ROWS).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
