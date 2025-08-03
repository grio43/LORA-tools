#!/usr/bin/env python3
"""
append_r34.py – Batch‑rename image/JSON pairs and patch the “file_name”
field inside the JSON.

Default directory: K:\\Ready for captions without tags
Dry‑run by default; add --commit to make changes.

Improvements over v1:
 • safer two‑phase rename with rollback
 • atomic metadata rewrite via temp file
 • optional recursion
 • optional --suffix and --dir overrides
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# ────── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_SUFFIX = "_dab"
DEFAULT_DIR    = Path(r"K:\Ready for captions without tags")

IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif", ".tif",
                  ".tiff", ".bmp", ".webp"}
META_EXT       = ".json"          # Only one at the moment

# ────── Helpers ────────────────────────────────────────────────────────────────
def find_metadata(img: Path) -> Path | None:
    """Return the matching *.json* file if present."""
    meta = img.with_suffix(META_EXT)
    return meta if meta.exists() else None


def patch_json(meta: Path, new_image_name: str) -> None:
    """Update the file_name field, writing atomically."""
    with meta.open("r", encoding="utf‑8") as fh:
        data = json.load(fh)
    data["file_name"] = new_image_name

    # Write to temp file then replace – protects against mid‑write crashes.
    with tempfile.NamedTemporaryFile(
        "w", dir=meta.parent, delete=False, encoding="utf‑8"
    ) as tmp:
        json.dump(data, tmp, indent=4, ensure_ascii=False)
        tmp_path = Path(tmp.name)

    tmp_path.replace(meta)   # atomic on same filesystem


def safe_rename(src: Path, dst: Path, overwrite: bool) -> None:
    """Rename or replace *src* with *dst* atomically if allowed."""
    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)
        # Path.replace() overwrites atomically
        src.replace(dst)
    else:
        src.rename(dst)


def process(
    directory: Path, suffix: str, commit: bool, overwrite: bool, recursive: bool
) -> None:
    """Core routine."""
    iterable = directory.rglob("*") if recursive else directory.iterdir()

    for img in iterable:
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = img.stem
        if stem.endswith(suffix):      # Already done
            continue

        meta = find_metadata(img)
        if meta is None:
            print(f"⚠ No JSON for {img.name}; skip")
            continue

        new_img  = img.with_name(f"{stem}{suffix}{img.suffix}")
        new_meta = meta.with_name(f"{meta.stem}{suffix}{META_EXT}")

        print(f"{img.name}  →  {new_img.name}")
        print(f"{meta.name} →  {new_meta.name}")

        if not commit:
            continue           # dry‑run only

        # ── two‑phase rename with rollback ────────────────────────────────────
        try:
            safe_rename(meta, new_meta, overwrite)
            safe_rename(img,  new_img,  overwrite)
        except Exception as exc:
            # Attempt to roll back if first rename succeeded
            if new_meta.exists() and not meta.exists():
                new_meta.rename(meta)
            print(f"‼  Error: {exc}. Pair left untouched.", file=sys.stderr)
            continue

        # ── patch JSON ────────────────────────────────────────────────────────
        try:
            patch_json(new_meta, new_img.name)
        except Exception as exc:              # unlikely, but revert names
            new_img.rename(img)
            new_meta.rename(meta)
            print(f"‼  JSON write failed: {exc}. Rolled back.", file=sys.stderr)


# ────── CLI ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Append a suffix to image / JSON pairs and fix metadata."
    )
    ap.add_argument("--commit",    action="store_true",
                    help="Actually rename (omit for dry‑run).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite if target names already exist.")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into sub‑folders.")
    ap.add_argument("--suffix",    default=DEFAULT_SUFFIX,
                    help=f"Suffix to append (default '{DEFAULT_SUFFIX}').")
    ap.add_argument("--dir",       type=Path, default=DEFAULT_DIR,
                    help=f"Target directory (default '{DEFAULT_DIR}').")
    args = ap.parse_args()

    if not args.dir.is_dir():
        sys.exit(f"Directory not found: {args.dir}")

    process(args.dir, args.suffix, args.commit, args.overwrite, args.recursive)


if __name__ == "__main__":
    main()
