
"""
Build tag_review.csv with four space‑separated columns
    file_name · added · removed · original_tags

• Any mixture of commas / whitespace in the source is accepted.  
• No field in the output contains a comma.
"""

from pathlib import Path
import re
import pandas as pd

# ── 0. EDIT THESE PATHS ────────────────────────────────────────────────────
DATASET_A_PATH = Path(r"J:\json backup from ready without tags\merged.csv")   # ground truth
DATASET_B_PATH = Path(r"D:\TrainingCandidates\train_dataset.csv")   # AI output
OUTPUT_PATH    = Path(r"D:\TrainingCandidates\tag_review3.csv")  # result

# ── 1. HELPERS ─────────────────────────────────────────────────────────────
def norm_fname(v: str) -> str:
    """Lower‑case basename only (images/sample.jpg → sample.jpg)."""
    return Path(v.strip()).name.lower() if isinstance(v, str) and v.strip() else ""

_SPLIT_RE = re.compile(r"[,\s]+")        # 1‑plus commas OR whitespace

def tokenise(raw: str) -> set[str]:
    """Return a set of lowercase Danbooru‑style tokens (underscores kept)."""
    if not isinstance(raw, str):
        return set()
    return {tok.lower() for tok in _SPLIT_RE.split(raw.strip()) if tok}

def collapse_tag_sets(series: pd.Series) -> set[str]:
    """Union of all tag‑sets in a Series of Python sets."""
    return set().union(*series)

# ── 2. LOAD CSVs (with existence check) ────────────────────────────────────
for path, label in [(DATASET_A_PATH, "Dataset A"), (DATASET_B_PATH, "Dataset B")]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found → {path}")

df_a = pd.read_csv(DATASET_A_PATH, dtype=str, keep_default_na=False)
df_b = pd.read_csv(DATASET_B_PATH, dtype=str, keep_default_na=False)

# ── 3. NORMALISE FILENAMES ─────────────────────────────────────────────────
df_a["file_norm"] = df_a["file_name"].apply(norm_fname)
df_b["file_norm"] = df_b["file_name"].apply(norm_fname)

# ── 4A. DATASET A  (deduplicate) ───────────────────────────────────────────
df_a["orig_tags_raw"] = df_a.get("tags", "").fillna("")
df_a["set_row"] = df_a["orig_tags_raw"].apply(tokenise)

df_a = (
    df_a.groupby("file_norm", sort=False)
        .agg(
            file_name=("file_name", "first"),             # keep first seen name
            set_A=("set_row", collapse_tag_sets),
            original_tags=("set_row",
                           lambda rows: " ".join(sorted(collapse_tag_sets(rows))))
        )
        .reset_index()
)

# ── 4B. DATASET B  (always union rows) ─────────────────────────────────────
tag_col_b = "tag" if "tag" in df_b.columns else "tags"
df_b["set_row"] = df_b[tag_col_b].fillna("").apply(tokenise)

set_B_df = (
    df_b.groupby("file_norm", sort=False)["set_row"]
        .agg(collapse_tag_sets)
        .rename("set_B")
        .reset_index()
)

# ── 5. MERGE (outer; now guaranteed one‑to‑one) ────────────────────────────
merged = (
    df_a[["file_norm", "file_name", "original_tags", "set_A"]]
        .merge(set_B_df, how="outer", on="file_norm", validate="one_to_one")
)

# Convert any NaNs into empty sets
for col in ("set_A", "set_B"):
    merged[col] = merged[col].apply(lambda x: x if isinstance(x, set) else set())

# ── 6. DELTA SETS ----------------------------------------------------------
merged["added"]   = merged.apply(lambda r: " ".join(sorted(r["set_B"] - r["set_A"])), axis=1)
merged["removed"] = merged.apply(lambda r: " ".join(sorted(r["set_A"] - r["set_B"])), axis=1)

# ── 7. CLEAN ORIGINAL_TAGS (remove commas) ---------------------------------
merged["original_tags_clean"] = (
    merged["original_tags"]
        .str.replace(",", " ", regex=False)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
)

# ── 8. FINAL REVIEW SHEET ---------------------------------------------------
review = (
    merged
        .assign(file_name=lambda d: d["file_name"].where(d["file_name"].str.strip() != "", d["file_norm"]))
        [["file_name", "added", "removed", "original_tags_clean"]]
        .rename(columns={"original_tags_clean": "original_tags"})
        .sort_values("file_name")
        .reset_index(drop=True)
)

# ── 9. WRITE CSV -----------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
review.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"✓  Wrote {len(review):,} rows → {OUTPUT_PATH.resolve()}")
