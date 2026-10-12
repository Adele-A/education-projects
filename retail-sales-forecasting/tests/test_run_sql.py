"""Tests for the SQL file parser in src/run_sql.py.

The parser decides which statements are run and which description is printed
for each query, so a silent parsing error would hide a query from the report.
The tests use the real sql/queries.sql file and a small in-memory text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_sql


def test_split_real_queries_file_in_order():
    text = run_sql.SQL_FILE.read_text(encoding="utf-8")
    queries = run_sql.split_queries(text)
    assert len(queries) == 8
    # Descriptions must be Q1 to Q8 in file order
    labels = [description.split(":")[0] for description, _ in queries]
    assert labels == [f"Q{n}" for n in range(1, 9)]
    # Comment lines must not end up inside the executable statements
    for _, statement in queries:
        assert statement
        assert not any(line.strip().startswith("--") for line in statement.splitlines())


def test_split_skips_statements_without_description():
    text = "-- header comment\nSELECT 1;\n-- Q1: test question\nSELECT 2;\n"
    queries = run_sql.split_queries(text)
    assert queries == [("Q1: test question", "SELECT 2")]
