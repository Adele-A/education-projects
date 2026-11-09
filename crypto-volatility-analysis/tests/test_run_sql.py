"""Tests for the SQLite helper in src/run_sql.py.

These tests use an in-memory database only; no files are read or written.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import run_sql


def test_safe_sqrt_known_values():
    """Regular values are square-rooted and NULL stays NULL."""
    assert run_sql.safe_sqrt(4.0) == pytest.approx(2.0)
    assert run_sql.safe_sqrt(0.0) == 0.0
    assert run_sql.safe_sqrt(None) is None


def test_safe_sqrt_clamps_tiny_negative_variance():
    """Cancellation can leave a tiny negative variance; it must not raise."""
    assert run_sql.safe_sqrt(-1e-18) == 0.0


def test_sqlite_query_survives_negative_variance():
    """The registered SQRT function lets a SQL query run on a negative input."""
    conn = sqlite3.connect(":memory:")
    conn.create_function("SQRT", 1, run_sql.safe_sqrt)
    value = conn.execute("SELECT SQRT(-1e-12)").fetchone()[0]
    conn.close()
    assert value == 0.0
