#!/usr/bin/env python3
"""
cleanup_exact_keyword_hardcoded.py

Recursively scans ROOT_DIR for *.json files.  
If a JSON file contains an exact keyword token (case‑insensitive),
the script deletes
  • the JSON file
  • any image file that shares the same stem.

The root directory is hard‑coded to avoid user path errors.
"""

import json
import re
import sys
from pathlib import Path
from typing import Iterable

# ---------- configuration ----------------------------------------------------

ROOT_DIR = Path(r"J:\New file\Danbooru2004\Images")   # <-- edit if you ever move data
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

# ---------- helpers ----------------------------------------------------------

def walk_json_files(root: Path) -> Iterable[Path]:
    """Yield every *.json file beneath root (recursively)."""
    yield from root.rglob("*.json")


def json_contains_exact_keyword(path: Path, keyword: str) -> bool:
    """
    True if *path* contains *keyword* as a standalone string value (case‑insensitive).
    Falls back to regex when JSON is malformed.
    """
    norm_kw = keyword.strip().lower()

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # malformed JSON – do a conservative regex scan
        pattern = re.compile(
            rf'"\s*{re.escape(norm_kw)}\s*"(?=[,\]\}}])', flags=re.IGNORECASE
        )
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return bool(pattern.search(f.read()))

    def contains(obj) -> bool:
        if isinstance(obj, str):
            return obj.strip().lower() == norm_kw
        if isinstance(obj, list):
            return any(contains(v) for v in obj)
        if isinstance(obj, dict):
            return any(contains(v) for v in obj.values())
        return False

    return contains(data)


def find_matching_pairs(root: Path, keyword: str):
    """Return a list of (json_path, image_path_or_None) pairs to delete."""
    pairs = []
    for jpath in walk_json_files(root):
        if json_contains_exact_keyword(jpath, keyword):
            stem = jpath.with_suffix("")  # strip .json
            img_path = next(
                (stem.with_suffix(ext) for ext in IMAGE_EXTS if stem.with_suffix(ext).exists()),
                None,
            )
            pairs.append((jpath, img_path))
    return pairs


def delete_files(pairs):
    for jpath, ipath in pairs:
        try:
            jpath.unlink()
            print(f"Deleted JSON : {jpath}")
        except Exception as e:
            print(f"ERROR deleting {jpath}: {e}")
        if ipath and ipath.exists():
            try:
                ipath.unlink()
                print(f"Deleted image: {ipath}")
            except Exception as e:
                print(f"ERROR deleting {ipath}: {e}")

# ---------- main -------------------------------------------------------------

def main():
    if not ROOT_DIR.is_dir():
        sys.exit(f"❌  ROOT_DIR does not exist: {ROOT_DIR}")

    keyword = input("Enter EXACT keyword to remove (case‑insensitive): ").strip()
    if not keyword:
        sys.exit("❌  Keyword cannot be empty.")

    print("\n🔍  Scanning, please wait…")
    pairs = find_matching_pairs(ROOT_DIR, keyword)

    if not pairs:
        print(f"✅  No matches for '{keyword}' found in {ROOT_DIR}. Nothing to delete.")
        return

    print("\nThe following files would be deleted:")
    for jpath, ipath in pairs:
        print(f"  - {jpath}")
        if ipath:
            print(f"    {ipath}")

    confirm = input(
        f"\n⚠️  Delete {len(pairs)} JSON file(s) "
        f"and their matched image(s)? Type 'YES' to confirm: "
    ).strip()

    if confirm == "YES":
        delete_files(pairs)
        print("\n✅  Deletion complete.")
    else:
        print("\nAborted. No files were removed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
