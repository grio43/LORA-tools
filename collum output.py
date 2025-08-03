#!/usr/bin/env python3
"""
list_parquet_columns.py

Print all column names in a Parquet file or in every *.parquet file
under a directory, without reading any rows.

Usage
-----
# Single file
python list_parquet_columns.py /path/to/data.parquet

# Entire directory (recursively finds *.parquet)
python list_parquet_columns.py /path/to/dir
"""
import sys
from pathlib import Path
import pyarrow.parquet as pq


def columns_in_file(file_path: Path) -> list[str]:
    """Return column names for one Parquet file (fast, metadata‑only)."""
    pf = pq.ParquetFile(file_path)
    return pf.schema.names


def main(pathstr: str) -> None:
    path = Path(pathstr)
    parquet_files = (
        [path] if path.is_file()
        else list(path.rglob("*.parquet"))
    )

    if not parquet_files:
        sys.exit(f"No Parquet files found at {path}")

    for f in parquet_files:
        cols = columns_in_file(f)
        print(f"📄 {f} ({len(cols)} columns)")
        for c in cols:
            print(f"  • {c}")
        print()  # blank line between files


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_parquet_columns.py <file_or_directory>")
        sys.exit(1)
    main(sys.argv[1])