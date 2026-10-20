"""Run the analytical SQL queries on the SYNTHETIC delivery data.

Loads data/couriers.csv and data/orders.csv into an in-memory SQLite
database (pandas.to_sql), splits sql/queries.sql into statements, runs each
one and prints its description (the '-- Q<n>:' comment) and the first rows.

Run from the project root (after python src/data_generator.py):
    python src/run_sql.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SQL_FILE = PROJECT_DIR / "sql" / "queries.sql"

MAX_ROWS = 10  # number of rows printed per query


def load_database(data_dir: Path = DATA_DIR) -> sqlite3.Connection:
    """Create an in-memory SQLite database with the orders and couriers tables.

    Args:
        data_dir: Folder containing orders.csv and couriers.csv.

    Returns:
        An open sqlite3 connection.

    Raises:
        FileNotFoundError: If a CSV file is missing.
    """
    orders_path = data_dir / "orders.csv"
    couriers_path = data_dir / "couriers.csv"
    if not orders_path.exists() or not couriers_path.exists():
        raise FileNotFoundError(
            "CSV files not found in data/. Run: python src/data_generator.py"
        )
    conn = sqlite3.connect(":memory:")
    # order_datetime stays a text column ('YYYY-MM-DD HH:MM:SS'), which SQLite date functions accept
    pd.read_csv(couriers_path).to_sql("couriers", conn, index=False)
    pd.read_csv(orders_path).to_sql("orders", conn, index=False)
    return conn


def split_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split an SQL script into (description, statement) pairs.

    Statements are separated by semicolons. The description is taken from the
    comment line that starts with '-- Q'. Chunks with no executable SQL
    (for example, only comments) are skipped.

    Args:
        sql_text: Full content of the SQL file.

    Returns:
        List of (description, statement) tuples in file order.
    """
    queries = []
    for chunk in sql_text.split(";"):
        lines = chunk.strip().splitlines()
        description = ""
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("-- Q"):
                description = stripped[3:].strip()
            elif stripped.startswith("--") or not stripped:
                continue
            else:
                code_lines.append(line)
        if code_lines:
            queries.append((description or "(no description)", chunk.strip()))
    return queries


def main() -> None:
    """Load the data, run every query from sql/queries.sql and print results."""
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    conn = load_database()
    queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
    print(f"Loaded tables into in-memory SQLite. Running {len(queries)} queries.")

    for description, statement in queries:
        print(f"\n{description}")
        result = pd.read_sql_query(statement, conn)
        print(f"({len(result)} rows, showing up to {MAX_ROWS})")
        print(result.head(MAX_ROWS).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
