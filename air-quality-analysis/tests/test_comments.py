"""Tests for the English-only text files (src/data_generator.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_generator as dg  # noqa: E402


def test_clean_text_files_replaces_non_english_comments(tmp_path):
    comments = tmp_path / "comments.csv"
    comments.write_text("comment_id,comment\n1,\u0422\u0435\u0441\u0442\n", encoding="utf-8")
    dg.clean_text_files(tmp_path)
    text = comments.read_text(encoding="utf-8")
    assert text.isascii()
    assert text.splitlines()[0] == "comment_id,comment"
    # A missing notes file is not created by the cleanup.
    assert not (tmp_path / "notes.csv").exists()


def test_clean_text_files_replaces_non_english_notes(tmp_path):
    notes = tmp_path / "notes.csv"
    notes.write_text("note_id,note\n1,\u0422\u0435\u0441\u0442\n", encoding="utf-8")
    dg.clean_text_files(tmp_path)
    text = notes.read_text(encoding="utf-8")
    assert text.isascii()
    assert text.splitlines()[0] == "note_id,note"
