"""Run the SQL queries from sql/queries.sql on the SYNTHETIC data.

Loads data/customers.csv and data/transactions.csv into an in-memory SQLite
database (pandas.to_sql), splits sql/queries.sql into statements, runs each
one and prints its description and the first rows of the result.
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


def load_database():
    """Create an in-memory SQLite database with both tables."""
    cust_path = DATA_DIR / "customers.csv"
    tx_path = DATA_DIR / "transactions.csv"
    if not cust_path.exists() or not tx_path.exists():
        raise SystemExit("Data files not found. Run: python src/data_generator.py")

    conn = sqlite3.connect(":memory:")
    # The timestamp stays a text column ('YYYY-MM-DD HH:MM:SS'), which SQLite's
    # strftime() can parse directly
    pd.read_csv(cust_path).to_sql("customers", conn, index=False)
    pd.read_csv(tx_path).to_sql("transactions", conn, index=False)
    return conn


def parse_queries(text):
    """Split the SQL file into (description, statement) pairs.

    Statements end with a semicolon. The description is the comment line
    that starts with '-- Q<n>:'. Other comment lines are ignored.
    """
    queries = []
    for chunk in text.split(";"):
        description = None
        sql_lines = []
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                if re.match(r"--\s*Q\d+\s*:", stripped):
                    description = stripped.lstrip("-").strip()
            elif stripped:
                sql_lines.append(line)
        if sql_lines:  # skip empty trailing chunks
            statement = "\n".join(sql_lines).strip()
            queries.append((description or "(no description)", statement))
    return queries


def main():
    if not SQL_FILE.exists():
        raise SystemExit("SQL file not found: %s" % SQL_FILE)

    conn = load_database()
    queries = parse_queries(SQL_FILE.read_text(encoding="utf-8"))
    print("Loaded %d queries from %s" % (len(queries), SQL_FILE.name))

    for description, statement in queries:
        print("\n" + "=" * 70)
        print(description)
        print("=" * 70)
        result = pd.read_sql_query(statement, conn)
        if result.empty:
            print("(no rows)")
        else:
            print(result.head(MAX_ROWS).to_string(index=False))
            if len(result) > MAX_ROWS:
                print("... (%d rows in total, first %d shown)" % (len(result), MAX_ROWS))

    conn.close()


if __name__ == "__main__":
    main()
