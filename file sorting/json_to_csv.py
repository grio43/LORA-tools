"""
json_to_csv.py
--------------
Walks a directory tree, loads every *.json file it finds, and writes all the
objects into one CSV file.

Requires: Python 3 (no external packages)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# ❑ 1. “Hard‑coded” configuration section
# ---------------------------------------------------------------------------

# Directory that holds the *.json files you want to convert -----------------
# Example: Windows r"C:\Users\me\data\json_files" or POSIX "/home/me/json"
SOURCE_DIR: Path = Path(r"K:\json backup from ready without tags")

# Name (or full path) of the output CSV file --------------------------------
#   • If you give just a filename, it will be created in the current directory
#   • Use Path("myfile.csv").resolve() if you want an absolute path
OUTPUT_CSV: Path = Path(r"K:\json backup from ready without tags")

# Set to True if you want to scan sub‑directories as well
RECURSIVE_SEARCH: bool = False

# ---------------------------------------------------------------------------
# ❑ 2. Helper functions
# ---------------------------------------------------------------------------

def find_json_files(root: Path, recursive: bool = False) -> List[Path]:
    """Return a list of *.json files under *root* (optionally including sub‑dirs)."""
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(root.glob(pattern))

def load_json(file_path: Path) -> Dict[str, Any]:
    """Safely load a JSON file; log and skip it if it cannot be parsed."""
    try:
        with file_path.open(encoding="utf‑8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logging.warning("Skipping %s: invalid JSON (%s)", file_path.name, exc)
        return {}  # empty → will be ignored later

# ---------------------------------------------------------------------------
# ❑ 3. Main conversion logic
# ---------------------------------------------------------------------------

def json_directory_to_csv(src_dir: Path, out_csv: Path, recursive: bool = False) -> None:
    """Convert all JSON objects in *src_dir* into one CSV file at *out_csv*."""
    records: List[Dict[str, Any]] = []

    # 1. Collect data from every JSON file
    for jp in find_json_files(src_dir, recursive):
        data = load_json(jp)
        if data:  # skip empty dicts from invalid files
            records.append(data)

    if not records:
        logging.error("No valid JSON objects found in %s", src_dir)
        return

    # 2. Build the complete header list (union of all keys)
    fieldnames: List[str] = []
    for rec in records:
        for key in rec:
            if key not in fieldnames:
                fieldnames.append(key)

    # 3. Write out the CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf‑8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    logging.info("✔ Wrote %d records to %s", len(records), out_csv.resolve())

# ---------------------------------------------------------------------------
# ❑ 4. Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    json_directory_to_csv(SOURCE_DIR, OUTPUT_CSV, recursive=RECURSIVE_SEARCH)
