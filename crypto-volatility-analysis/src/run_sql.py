"""Run the SQL analysis on the SYNTHETIC crypto tables.

Loads the CSV files from data/ into an in-memory SQLite database
(pandas.to_sql), splits sql/queries.sql into statements, runs each one
and prints its description and the first rows of the result.

The data is simulated and is NOT real market data.

Run from the project root:
    python src/run_sql.py
"""

import math
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

TABLES = ["assets", "market_regimes", "daily_prices"]
MAX_ROWS = 10  # rows printed per query


def safe_sqrt(x):
    """Square root that can be registered as a SQLite user-defined function.

    The volatility queries compute SQRT(AVG(r*r) - AVG(r)*AVG(r)). Floating-point
    cancellation can leave such a variance slightly negative when the true value
    is essentially zero. math.sqrt would raise on it and SQLite would abort the
    whole query, so tiny negative values are clamped to zero. NULL (None) is
    passed through unchanged.
    """
    if x is None:
        return None
    return math.sqrt(max(float(x), 0.0))


def load_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database from the CSV tables."""
    conn = sqlite3.connect(":memory:")
    # Make SQRT available regardless of how SQLite was compiled
    conn.create_function("SQRT", 1, safe_sqrt)
    for name in TABLES:
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/data_generator.py' first."
            )
        df = pd.read_csv(path)
        df.to_sql(name, conn, index=False)
        print(f"Loaded table {name}: {len(df)} rows")
    return conn


def split_queries(text: str) -> list:
    """Split the SQL file into (description, sql) pairs.

    Statements are separated by semicolons. The description is the comment
    line starting with '-- Q<n>:'. Statements without such a line (for
    example the header comments) are skipped.
    """
    queries = []
    for chunk in text.split(";"):
        description = None
        sql_lines = []
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("-- Q") and ":" in stripped:
                description = stripped[3:].strip()
            elif stripped.startswith("--") or not stripped:
                continue
            else:
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip()
        if description and sql:
            queries.append((description, sql))
    return queries


def main() -> None:
    """Load the data, run every query and print the first rows."""
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    if not SQL_FILE.exists():
        raise FileNotFoundError(f"{SQL_FILE} not found.")

    conn = load_database()
    queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"\nFound {len(queries)} queries in {SQL_FILE.name}")

    for description, sql in queries:
        print("\n" + "=" * 100)
        print(description)
        print("=" * 100)
        result = pd.read_sql_query(sql, conn)
        print(f"({len(result)} rows, showing up to {MAX_ROWS})")
        print(result.head(MAX_ROWS).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
