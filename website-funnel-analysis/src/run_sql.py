"""Run the SQL queries from sql/queries.sql on the SYNTHETIC project data.

Loads data/sessions.csv and data/traffic_sources.csv into an in-memory SQLite
database (pandas.to_sql), splits queries.sql into statements, runs each one and
prints its description and the first rows of the result.

Run from the project root:  python src/run_sql.py
(Run python src/data_generator.py first to create the CSV files.)
"""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

MAX_ROWS = 10  # rows shown per query


def load_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database with the two CSV tables."""
    tables = {
        "sessions": DATA_DIR / "sessions.csv",
        "traffic_sources": DATA_DIR / "traffic_sources.csv",
    }
    for path in tables.values():
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found. Run: python src/data_generator.py")

    conn = sqlite3.connect(":memory:")
    for name, path in tables.items():
        pd.read_csv(path).to_sql(name, conn, index=False, if_exists="replace")
    return conn


def parse_queries(text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (description, sql) pairs.

    Statements end with a semicolon. Comment lines inside a statement block
    form the description ('-- Q<n>: ...' is the first one). Blocks without any
    executable SQL (for example the file header) are skipped.
    """
    queries = []
    for block in text.split(";"):
        comments, code = [], []
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                comments.append(stripped.lstrip("-").strip())
            elif stripped:
                code.append(line)
        if not code:
            continue
        # Header comments precede the first query: keep only the Q-line onward
        q_idx = next((i for i, c in enumerate(comments) if c.startswith("Q")), None)
        if q_idx is not None:
            comments = comments[q_idx:]
        description = " ".join(comments) if comments else "(no description)"
        queries.append((description, "\n".join(code)))
    return queries


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    if not SQL_FILE.exists():
        raise FileNotFoundError(f"{SQL_FILE} not found")

    conn = load_database()
    queries = parse_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"Loaded synthetic tables into in-memory SQLite. Running {len(queries)} queries.")

    for description, sql in queries:
        print(f"\n{'=' * 80}\n{description}\n{'-' * 80}")
        result = pd.read_sql_query(sql, conn)
        print(result.head(MAX_ROWS).to_string(index=False))
        if len(result) > MAX_ROWS:
            print(f"... ({len(result)} rows in total, first {MAX_ROWS} shown)")

    conn.close()


if __name__ == "__main__":
    main()
