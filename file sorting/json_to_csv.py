#!/usr/bin/env python3
"""
json_to_csv.py
--------------

Walk a directory tree, load every *.json file it finds and write all JSON
objects into one CSV file.

Pure standard‑library – requires only Python 3.x.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

# ────────────────────────── 1. User configuration ──────────────────────────

SOURCE_DIR      = Path(r"J:\New file\Ready for captions without tags")
OUTPUT_CSV      = Path(r"J:\json backup from ready without tags\merged.csv")
RECURSIVE_SEARCH = True                       # ← look in sub‑folders

# ────────────────────────── 2. Helper functions ────────────────────────────

def find_json_files(root: Path, *, recursive: bool = False) -> Iterable[Path]:
    """Yield every *.json file under *root* (optionally descending dirs)."""
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")
    pattern = "**/*.json" if recursive else "*.json"
    yield from root.glob(pattern)

def load_json(fp: Path) -> Dict[str, Any] | None:
    """Return the decoded JSON object from *fp* or None if the file is bad."""
    try:
        # utf‑8‑sig will strip a Unicode BOM transparently
        with fp.open(encoding="utf‑8‑sig") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logging.warning("Skipping %s – invalid JSON (%s)", fp, exc)
    except OSError as exc:
        logging.warning("Skipping %s – %s", fp, exc)
    return None

# ────────────────────────── 3. Main conversion ─────────────────────────────

def json_directory_to_csv(src_dir: Path, out_csv: Path, *, recursive: bool = False) -> None:
    """Convert every JSON file in *src_dir* to rows in *out_csv*."""
    records: List[Dict[str, Any]] = []

    # 3‑a  Collect data
    for fp in find_json_files(src_dir, recursive=recursive):
        data = load_json(fp)
        if isinstance(data, dict):            # one object per file
            records.append(data)
        elif isinstance(data, list):          # allow files that hold a list
            records.extend(d for d in data if isinstance(d, dict))

    if not records:
        logging.error("No valid JSON objects found in %s", src_dir)
        return
    logging.info("Found %d JSON object(s) in %s", len(records), src_dir)

    # 3‑b  Build the CSV header (union of all keys, order‑preserved)
    fieldnames: List[str] = []
    for rec in records:
        for key in rec:
            if key not in fieldnames:
                fieldnames.append(key)

    # 3‑c  Write the CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf‑8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    logging.info("✔ Wrote %d rows to %s", len(records), out_csv.resolve())

# ────────────────────────── 4. CLI entry point ─────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    json_directory_to_csv(SOURCE_DIR, OUTPUT_CSV, recursive=RECURSIVE_SEARCH)
