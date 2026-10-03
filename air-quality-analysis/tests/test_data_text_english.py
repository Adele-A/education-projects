"""Check that the project data files contain English-only text.

Importing data_generator replaces old non-English notes/comments files, so
this test also repairs leftovers from earlier versions of the data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_generator as dg  # noqa: E402
from clean_text import non_english_files  # noqa: E402


def test_data_folder_has_no_non_english_text():
    dg.clean_text_files(dg.DATA_DIR)
    assert non_english_files(dg.DATA_DIR) == []


def test_non_english_files_detects_and_cleanup_fixes(tmp_path):
    path = tmp_path / "comments.csv"
    path.write_text("comment_id,comment\n1,\u0422\u0435\u0441\u0442\n", encoding="utf-8")
    assert non_english_files(tmp_path) == [path]
    dg.clean_text_files(tmp_path)
    assert non_english_files(tmp_path) == []
