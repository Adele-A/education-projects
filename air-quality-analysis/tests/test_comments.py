"""Tests for the English-only comments file written by src/data_generator.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_generator as dg  # noqa: E402


def test_write_comments_replaces_non_english_file(tmp_path):
    path = tmp_path / "comments.csv"
    path.write_text("comment_id,comment\n1,\u0422\u0435\u0441\u0442\n", encoding="utf-8")
    dg.write_comments(path)
    text = path.read_text(encoding="utf-8")
    assert text.isascii()
    assert text.splitlines()[0] == "comment_id,comment"
    assert len(text.splitlines()) == 2
