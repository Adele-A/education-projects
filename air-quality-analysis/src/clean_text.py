"""Replace old non-English text files in data/ with English-only versions.

Some earlier versions of data/notes.csv and data/comments.csv contained
non-English text. This script rewrites such files with the English-only
tables defined in src/data_generator.py and then verifies that every CSV
file in data/ is plain ASCII (English only).

Run from the project root:
    python src/clean_text.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_generator as dg  # noqa: E402


def non_english_files(data_dir=dg.DATA_DIR):
    """Return the CSV files in data_dir that contain non-ASCII bytes."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return [p for p in sorted(data_dir.glob("*.csv")) if dg.has_non_english_text(p)]


def main():
    dg.clean_text_files(dg.DATA_DIR)
    remaining = non_english_files()
    if remaining:
        names = ", ".join(p.name for p in remaining)
        raise SystemExit(f"Non-English text still present in: {names}")
    print(f"All CSV files in {dg.DATA_DIR} are English-only.")


if __name__ == "__main__":
    main()
