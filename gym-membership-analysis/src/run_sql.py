"""Run the SQL queries from sql/queries.sql on the SYNTHETIC gym dataset.

Loads data/members.csv and data/upgrades.csv into an in-memory SQLite
database (pandas.to_sql), splits sql/queries.sql into statements, runs each
one and prints its description and the first rows of the result.

Run from the project root:  python src/run_sql.py
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

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

# A description line looks like: -- Q3: question text
DESCRIPTION_PATTERN = re.compile(r"^--\s*(Q\d+):\s*(.+)$", re.MULTILINE)


def load_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database with the members and upgrades tables.

    Returns the open connection; the caller is responsible for closing it.
    """
    conn = sqlite3.connect(":memory:")
    # Dates are kept as ISO text so SQLite date functions work on them
    for table in ["members", "upgrades"]:
        path = DATA_DIR / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run python src/data_generator.py first.")
        df = pd.read_csv(path)
        df.to_sql(table, conn, index=False, if_exists="replace")
        print(f"Loaded table {table}: {len(df)} rows")
    return conn


def split_queries(sql_text: str) -> list[tuple[str, str, str]]:
    """Split the SQL file into (label, description, statement) tuples.

    Statements are separated by semicolons; chunks without a '-- Q<n>:'
    description (for example the file header) are skipped.
    """
    queries = []
    for chunk in sql_text.split(";"):
        match = DESCRIPTION_PATTERN.search(chunk)
        if not match:
            continue
        # Keep only the SQL that follows the description line
        statement = chunk[match.end():].strip()
        if statement:
            queries.append((match.group(1), match.group(2).strip(), statement))
    return queries


def main() -> None:
    """Load the CSV data, run every query in sql/queries.sql and print the results."""
    if not SQL_FILE.exists():
        print(f"SQL file not found: {SQL_FILE}")
        sys.exit(1)

    conn = load_database()
    queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"\nFound {len(queries)} queries in {SQL_FILE.name}")

    for label, description, statement in queries:
        print("\n" + "=" * 70)
        print(f"{label}: {description}")
        print("=" * 70)
        result = pd.read_sql_query(statement, conn)
        print(f"({len(result)} rows, showing up to {MAX_ROWS})")
        print(result.head(MAX_ROWS).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
