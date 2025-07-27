#!/usr/bin/env python3
"""
A script to filter the Danbooru dataset based on metadata criteria
and optionally download the matching images using the cheesechaser library.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
from huggingface_hub import HfFolder, login

try:
    # Requires cheesechaser >= 0.5.0
    from cheesechaser.datapool import DanbooruNewestDataPool as DataPool
except ModuleNotFoundError:  # graceful degradation
    DataPool = None
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger("cheesechaser_search")

# ---------------------------------------------------------------------------
# Configuration
#
# Easiest way to use: Edit the default values directly in this class.
# Any value set here can be overridden by a command-line argument.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Holds all runtime parameters (may be overridden from CLI)."""

    # ---- Paths ------------------------------------------------------------
    metadata_db_path: str = r"J:\New file\Danbooru2004\metadata.parquet"
    output_dir: str = r"J:\New file\Danbooru2004\Images"

    # ---- Hugging Face -----------------------------------------------------
    dataset_repo: str = "deepghs/danbooru2024"
    hf_auth_token: Optional[str] = os.getenv("Add token", None)  # Set your token here or use env var

    # ---- Column names (aligns with DeepGHS conventions) -------------------
    tags_col: str = "tags"                      # Changed from "tag_string"
    #character_tags_col: str = "tag_string_character"  # No change needed as filter is off
    #copyright_tags_col: str = "tag_string_copyright"  # No change needed as filter is off
    #artist_tags_col: str = "tag_string_artist"      # No change needed as filter is off
    score_col: str = "score"
    rating_col: str = "rating"
    width_col: str = "width"                    # Changed from "image_width"
    height_col: str = "height"                  # Changed from "image_height"
    file_path_col: str = "file_url"
    id_col: str = "id"

    # ---- Filtering Toggles (Set to False to disable a filter group) ------
    enable_include_tags: bool = True
    enable_exclude_tags: bool = True
    #enable_character_filtering: bool = False # <-- SET TO FALSE
    #enable_copyright_filtering: bool = False # <-- SET TO FALSE
    #enable_artist_filtering: bool = False # <-- SET TO FALSE
    enable_score_filtering: bool = True
    enable_rating_filtering: bool = False
    enable_dimension_filtering: bool = True 
    per_image_json: bool = True

    # ---- Filtering Criteria (with placeholders) ---------------------------
    # General tags (e.g., appearance, actions, or objects)
    # "absurdly" will latch onto an exact string match.
    # Use "absurdly*" for prefix matching.
    include_tags: List[str] = field(default_factory=lambda: ["*eye"]) # <--
    exclude_tags: List[str] = field(default_factory=lambda: [  
        # --- Image Quality & Artifacts ---
        "lowres", "blurry", "pixelated", "jpeg artifacts", "compression artifacts",
        "low quality", "worst quality", "bad quality",
        "watermark", "signature", "artist name", "logo", "stamp",
        "text", "english_text", "speech bubble",

        # --- Anatomy & Proportions ---
        "bad anatomy", "bad hands", "bad proportions", "malformed limbs",
        "mutated hands", "extra limbs", "extra fingers", "fused fingers",
        "long neck", "deformed", "disfigured", "mutation", "poorly drawn face",

        # --- Unwanted Art Styles ---
        "3d", "cgi", "render", "vray",
        "comic",
        # --- People ---
        "2girls", "3girls", "2boys", "3boys",

        # --- Composition & Framing ---
        "grid", "collage", "multi-panel", "multiple views", "split screen",
        "border", "frame", "out of frame", "cropped",

        # --- Color & Tone ---
        "monochrome", "grayscale",

        # --- AI ---
        "ai generated", "ai art", "ai generated art", "ai generated image", "ai_generated", "ai_art",  "ai_artwork", "ai_image", "ai_artwork","ai artifact",
        # --- General Undesirables ---
        "ugly", "grotesque", "loli", "loli_(style)", "loli_(character)", "lolicon", "loli", "Lolicon", "Lolicons", "shotacon", "shotacon_(style)", "shotacon_(character)", "shotacon", "Shotacon", "Shotacons"
    ])

    # Character tags (add or remove character names)
    include_characters: List[str] = field(default_factory=lambda: ["hakurei_reimu", "kirisame_marisa"])
    exclude_characters: List[str] = field(default_factory=lambda: ["some_character_to_exclude"])

    # Copyright/Series tags (add or remove series, game, or anime titles)
    include_copyrights: List[str] = field(default_factory=lambda: ["touhou", "genshin_impact"])
    exclude_copyrights: List[str] = field(default_factory=lambda: ["some_series_to_exclude"])

    # Artist tags (add or remove specific artist names)
    include_artists: List[str] = field(default_factory=lambda: ["cutesexyrobutts*"])
    exclude_artists: List[str] = field(default_factory=lambda: ["bob"])

    # Other filters
    min_score: Optional[int] = 30 # <--
    ratings: List[str] = field(default_factory=lambda: ["safe", "general"])
    square_only: bool = False # <--
    min_square_size: int = 1024 # <--
    min_width: int = 1024 # <--
    min_height: int = 1024 # <--
    max_width: int = 90000 # <--
    max_height: int = 90000 # <--

    # ---- Behaviour flags --------------------------------------------------
    download_images: bool = True # <--
    save_filtered_metadata: bool = True # <--
    filtered_metadata_format: str = "json"  # json | txt # <--
    strip_json_details: bool = True # <--
    exclude_gifs: bool = True # <--
    dry_run: bool = False # <--

    # ---- Performance ------------------------------------------------------
    workers: int = 15 # <--



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_tag_list(value: str) -> List[str]:
    """Parses comma- or space-separated tag strings into a list."""
    return [t for t in re.split(r"[\s,]+", value.strip()) if t]

def build_cli() -> argparse.ArgumentParser:
    """Defines and configures the command-line argument parser."""
    p = argparse.ArgumentParser(
        prog="cheesechaser-search",
        description="Filter Danbooru metadata & optionally download matching images.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
            Example Usage:
            --------------
            # Download high-scoring images of '1girl' to a specific folder
            python search.py --metadata /data/danbooru.parquet --include "1girl solo" --min-score 100 --output ./My_Filtered_Images

            # Filter for square images that are at least 1024x1024, excluding certain tags
            python search.py --square --min-square-size 1024 --exclude "multiple_girls nsfw"

            # Perform a dry run to see match count without downloading or saving metadata
            python search.py --include cat_ears --dry-run
            """),
    )

    # Paths / IO
    p.add_argument("--metadata", type=str, help="Path to Danbooru parquet (default: %(default)s)")
    p.add_argument("--output", type=str, help="Destination directory for downloads & outputs")

    # Dataset / Auth
    p.add_argument("--repo", type=str, help="Hugging Face dataset repo id (default: %(default)s)")
    p.add_argument("--token", type=str, help="HF token string (overrides env / keyring)")

    # Tag filtering
    p.add_argument("--include", "-i", type=_parse_tag_list, help="Tags to include (comma or space separated)")
    p.add_argument("--exclude", "-x", type=_parse_tag_list, help="Tags to exclude")

    # Other filters
    p.add_argument("--min-score", type=int, help="Minimum score")
    p.add_argument("--ratings", nargs="*", help="Allowed ratings e.g. safe questionable")
    p.add_argument("--square", action="store_true", help="Require square images")
    p.add_argument("--min-square-size", type=int, help="Min dimension for square images")
    p.add_argument("--min-width", type=int)
    p.add_argument("--min-height", type=int)
    p.add_argument("--max-width", type=int)
    p.add_argument("--max-height", type=int)
    p.add_argument("--per-image-json", action="store_true",
               help="Write one JSON side-car per downloaded image")

    # Behaviour flags
    p.add_argument("--no-download", dest="download", action="store_false", help="Skip image downloads")
    p.add_argument("--no-save-metadata", dest="save_meta", action="store_false", help="Do not save filtered metadata")
    p.add_argument("--dry-run", action="store_true", help="Exit after printing stats (implies --no-download)")
    p.add_argument("--exclude-gifs", action="store_true", help="Exclude .gif files from the download list")

    # Perf
    p.add_argument("--workers", type=int, help="Number of download workers")

    return p

def apply_cli_overrides(args: argparse.Namespace, cfg: Config) -> None:
    """Mutates cfg in-place based on CLI arguments."""
    if args.metadata: cfg.metadata_db_path = args.metadata
    if args.output: cfg.output_dir = args.output
    if args.repo: cfg.dataset_repo = args.repo
    if args.token: cfg.hf_auth_token = args.token

    if args.include is not None: cfg.include_tags = args.include
    if args.exclude is not None: cfg.exclude_tags = args.exclude
    if args.min_score is not None: cfg.min_score = args.min_score
    if args.ratings is not None: cfg.ratings = args.ratings
    if args.square: cfg.square_only = True
    if args.min_square_size is not None: cfg.min_square_size = args.min_square_size
    if args.min_width is not None: cfg.min_width = args.min_width
    if args.min_height is not None: cfg.min_height = args.min_height
    if args.max_width is not None: cfg.max_width = args.max_width
    if args.max_height is not None: cfg.max_height = args.max_height

    if args.workers is not None: cfg.workers = args.workers
    # Final check: can't use `is not None` because action="store_false" makes `download` False
    if hasattr(args, 'download') and not args.download:
        cfg.download_images = False
    if hasattr(args, 'save_meta') and not args.save_meta:
        cfg.save_filtered_metadata = False

    if args.dry_run:
        cfg.dry_run = True
        cfg.download_images = False

    if args.exclude_gifs: 
        cfg.exclude_gifs = True 

# ---------------------------------------------------------------------------
# Metadata loading & filtering
# ---------------------------------------------------------------------------
def load_metadata(path: Path, cfg: Config) -> pd.DataFrame:
    """Loads parquet, selecting only projected columns for efficiency."""
    if not path.exists():
        log.error(f"❌ Metadata path not found: {path}")
        log.error("Please update `metadata_db_path` in the script or use the --metadata argument.")
        sys.exit(1)

    # First, let's inspect the file to see what columns it actually has.
    try:
        import pyarrow.parquet as pq
        parquet_schema = pq.read_schema(path)
        available_cols = [field.name for field in parquet_schema]
    except Exception as e:
        log.error(f"❌ Failed to read the schema from the metadata file: {e}")
        log.error(f"   Please ensure the file at '{path}' is a valid Parquet file.")
        sys.exit(1)

    # Now, check if the required 'id' column exists before we do anything else.
    if cfg.download_images and cfg.id_col not in available_cols:
        log.error(f"❌ CRITICAL ERROR: The ID column '{cfg.id_col}' is required for downloading, but it was not found in your metadata file.")
        log.error("This can happen if your metadata file uses a different name for the post ID column (e.g., 'post_id').")
        log.error("\nHere is a list of all available columns in your file:")
        log.error("-------------------------------------------")
        # Print columns in a clean, readable format
        for col in available_cols:
            log.error(f"  - {col}")
        log.error("-------------------------------------------")
        log.error("\n>>> ACTION REQUIRED <<<")
        log.error("1. Find the correct column name for post IDs in the list above.")
        log.error(f"2. Open this script and change the line `id_col: str = \"{cfg.id_col}\"` in the `Config` class to use the correct name.")
        log.error("   For example, if the correct column is 'post_id', change it to: `id_col: str = \"post_id\"`")
        sys.exit(1)

    # Dynamically build the list of columns to load based on config
    cols_to_load = set()
    if (cfg.enable_include_tags or cfg.enable_exclude_tags or
         (cfg.save_filtered_metadata and cfg.per_image_json)):
        cols_to_load.add(cfg.tags_col)
    if cfg.enable_character_filtering or (cfg.save_filtered_metadata and cfg.per_image_json):
        cols_to_load.add(cfg.character_tags_col)
    if cfg.enable_copyright_filtering: cols_to_load.add(cfg.copyright_tags_col)
    if cfg.enable_artist_filtering or (cfg.save_filtered_metadata and cfg.per_image_json):
        cols_to_load.add(cfg.artist_tags_col)
    if cfg.enable_score_filtering: cols_to_load.add(cfg.score_col)
    if cfg.enable_rating_filtering: cols_to_load.add(cfg.rating_col)
    if cfg.enable_dimension_filtering: cols_to_load.update([cfg.width_col, cfg.height_col])

    # Add columns required for I/O and downloading
    if cfg.download_images:
        cols_to_load.add(cfg.id_col)
        cols_to_load.add(cfg.file_path_col) 

    if cfg.save_filtered_metadata:
        cols_to_load.add(cfg.file_path_col)



    # We only load columns that are both requested and available.
    final_cols = list(cols_to_load.intersection(available_cols))

    log.info(f"📖 Loading metadata from: {path}")
    log.info(f"   Requesting columns: {final_cols}")
    try:
        df = pd.read_parquet(path, columns=final_cols)
    except Exception as e:
        log.error(f"❌ Failed to load metadata with the specified columns: {e}")
        sys.exit(1)

    if (
        cfg.download_images
        and cfg.id_col not in df.columns
        and (df.index.name == cfg.id_col or cfg.id_col in df.index.names) # check both name and names for multi-index
    ):
        log.info(
            "Detected '%s' stored as index – converting it to a column.", cfg.id_col
        )
        df.reset_index(inplace=True)
    # --- END: ROBUST FIX ---

    return df

def build_filter_mask(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Return boolean mask for rows matching cfg filters."""
    mask = pd.Series(True, index=df.index)
    log.info

    # --- File Type Filtering -----------------------------------------
    if cfg.file_path_col in df.columns:
        excluded_extensions = ('.zip', '.mp4', '.webm', '.swf')
        log.info(f"    Excluding files with extensions: {', '.join(excluded_extensions)}")
        mask &= ~df[cfg.file_path_col].str.lower().str.endswith(excluded_extensions, na=False)

    if cfg.exclude_gifs and cfg.file_path_col in df.columns:
        log.info("    Excluding .gif files from download list.")
        mask &= ~df[cfg.file_path_col].str.lower().str.endswith('.gif', na=False)

    # --- Tag-based Filtering -----------------------------------------
# Helper function to avoid code repetition
    def apply_tag_filters(series: pd.Series, include_list: List[str], exclude_list: List[str], log_name: str):
        nonlocal mask

        def _create_pattern(tag: str) -> str:
            """Generates a regex pattern based on wildcard usage in the tag."""
            starts_with_star = tag.startswith('*')
            ends_with_star = tag.endswith('*')
            clean_tag = tag.strip('*')
            escaped_tag = re.escape(clean_tag)

            # Case 1: Substring match (e.g., "*dog*")
            if starts_with_star and ends_with_star:
                return escaped_tag
            
            # Case 2: Prefix match (e.g., "dog*")
            elif ends_with_star:
                # Matches "dog_boy", "dogs", but not "good_dog"
                return r"\b" + escaped_tag

            # Case 3: Suffix match (e.g., "*dog")
            elif starts_with_star:
                # Matches "good_dog", but not "dogs" or "dog_boy"
                return escaped_tag + r"\b"

            # Case 4: Exact whole-word match (e.g., "dog")
            else:
                # This is the most efficient pattern for exact matches.
                return r"\b" + escaped_tag + r"\b"

        # Inclusion (AND logic): image must have ALL specified tags.
        if include_list:
            log.info(f"    Including {log_name} (ALL required): {include_list}")
            for tag in include_list:
                if not tag: continue # Skip empty strings
                pattern = _create_pattern(tag)
                mask &= series.str.contains(pattern, regex=True, na=False, flags=re.IGNORECASE)

        # Exclusion (OR logic): image must not have ANY of these tags.
        if exclude_list:
            log.info(f"    Excluding {log_name} (ANY will be excluded): {exclude_list}")
            
            # Combine all non-empty exclusion patterns into a single regex with '|'.
            # This is more efficient than applying .str.contains in a loop.
            patterns = [_create_pattern(tag) for tag in exclude_list if tag]
            
            if patterns:
                final_pattern = "|".join(patterns)
                mask &= ~series.str.contains(final_pattern, regex=True, na=False, flags=re.IGNORECASE)

    # General Tags
    if cfg.enable_include_tags or cfg.enable_exclude_tags:
        if cfg.tags_col in df.columns:
            tag_series = df[cfg.tags_col].str.lower().fillna("")
            
            # Conditionally create the lists to pass to the helper function
            include_list = cfg.include_tags if cfg.enable_include_tags else []
            exclude_list = cfg.exclude_tags if cfg.enable_exclude_tags else []
            
            # Only call the function if there's actually something to do
            if include_list or exclude_list:
                apply_tag_filters(tag_series, include_list, exclude_list, "general tags")
        else:
            log.warning(f"'{cfg.tags_col}' not found. Skipping general tag filtering.")

    # Copyright Tags
    if cfg.enable_copyright_filtering:
        if cfg.copyright_tags_col in df.columns:
            copy_series = df[cfg.copyright_tags_col].str.lower().fillna("")
            apply_tag_filters(copy_series, cfg.include_copyrights, cfg.exclude_copyrights, "copyrights")
        else:
            log.warning(f"'{cfg.copyright_tags_col}' not found. Skipping copyright filtering.")

    # Artist Tags
    if cfg.enable_artist_filtering:
        if cfg.artist_tags_col in df.columns:
            artist_series = df[cfg.artist_tags_col].str.lower().fillna("")
            apply_tag_filters(artist_series, cfg.include_artists, cfg.exclude_artists, "artists")
        else:
            log.warning(f"'{cfg.artist_tags_col}' not found. Skipping artist filtering.")

    # --- Metadata Filtering ------------------------------------------
    # Score
    if cfg.enable_score_filtering and cfg.min_score is not None:
        log.info(f"    Filtering for score >= {cfg.min_score}")
        mask &= df[cfg.score_col].fillna(0) >= cfg.min_score

    # Ratings
    if cfg.enable_rating_filtering and cfg.ratings:
        log.info(f"    Filtering for ratings: {cfg.ratings}")
        mask &= df[cfg.rating_col].fillna("").isin(cfg.ratings)

    # Dimensions
    if cfg.enable_dimension_filtering:
        w, h = df[cfg.width_col], df[cfg.height_col]
        if cfg.square_only:
            log.info(f"    Filtering for square images >= {cfg.min_square_size}px.")
            mask &= (w == h) & (w >= cfg.min_square_size)
        else:
            log.info(f"    Filtering for dimensions: {cfg.min_width}x{cfg.min_height} to {cfg.max_width}x{cfg.max_height}")
            if cfg.min_width > 0: mask &= w >= cfg.min_width
            if cfg.min_height > 0: mask &= h >= cfg.min_height
            if cfg.max_width > 0: mask &= w <= cfg.max_width
            if cfg.max_height > 0: mask &= h <= cfg.max_height
            
    return mask

# ---------------------------------------------------------------------------
# Output and Download Helpers
# ---------------------------------------------------------------------------
def save_filtered_metadata(df: pd.DataFrame, cfg: Config, dest_dir: Path) -> None:
    """Save metadata either as one big file or as one JSON per image."""
    keys_to_strip = [cfg.score_col, cfg.width_col, cfg.height_col]

    try:
        # ── Per-image side-cars ──────────────────────────────────────
        if cfg.filtered_metadata_format == "json" and cfg.per_image_json:
            for _, row in df.iterrows():
                image_id = row[cfg.id_col]
                path = dest_dir / f"{image_id}.json"

                # Convert row to a dictionary, dropping any NaN values
                record = row.dropna().to_dict()

                # --- START: MODIFIED LOGIC ---
                # Remove the file path column as it's not needed in the side-car
                if cfg.file_path_col in record:
                    del record[cfg.file_path_col]

                # If the strip flag is on, remove the specified keys
                if cfg.strip_json_details:
                    for key in keys_to_strip:
                        if key in record:
                            del record[key]
                # --- END: MODIFIED LOGIC ---

                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, ensure_ascii=False, indent=2)

            log.info(f"💾 Wrote {len(df):,} side-car JSON files to {dest_dir}")

        # ── One master file ─────────────────────────────────────────
        else:
            # This logic handles the case for a single master file output.
            df_to_save = df.copy() # Create a copy to avoid modifying the original df

            if cfg.strip_json_details:
                # Drop the columns from the DataFrame before saving
                columns_to_drop = [col for col in keys_to_strip if col in df_to_save.columns]
                if columns_to_drop:
                    df_to_save.drop(columns=columns_to_drop, inplace=True)
                    log.info(f"Stripping columns from master file: {columns_to_drop}")


            outfile = dest_dir / f"filtered_metadata.{cfg.filtered_metadata_format}"

            if cfg.filtered_metadata_format == "json":
                df_to_save.to_json(outfile, orient="records", lines=True, force_ascii=False)
            elif cfg.filtered_metadata_format == "txt":
                # The .txt output only contains file paths, so no changes needed here.
                df[cfg.file_path_col].to_csv(outfile, index=False, header=False)

            log.info(f"💾 Filtered metadata saved → {outfile}")

    except Exception as e:
        log.error(f"❌ Could not save filtered metadata: {e}")


def download_with_datapool(df: pd.DataFrame, cfg: Config, dst: Path) -> None:
    """Download images via CheeseChaser DataPool (deduplicated, resumable)."""
    if DataPool is None:
        log.error("CheeseChaser is not installed. Install with 'pip install cheesechaser'.")
        log.error("Alternatively, use --no-download to skip this step.")
        return

    pool = DataPool()
    ids = df[cfg.id_col].tolist()
    log.info(f"Starting DataPool download of {len(ids)} images to {dst}...")
    pool.batch_download_to_directory(
        resource_ids=ids,
        dst_dir=dst,
        max_workers=cfg.workers
    )
    log.info("🎉 Download complete!")


# Add this function before the main() function, around line 310

def verify_hf_auth(cfg: Config) -> None:
    """Checks if the Hugging Face token is valid before proceeding."""
    log.info("🔐 Verifying Hugging Face authentication...")
    token = cfg.hf_auth_token or HfFolder.get_token()

    if not token:
        log.error("❌ No Hugging Face token found.")
        log.error("Please set the 'hf_auth_token' in the script or run 'huggingface-cli login'.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
        user = HfApi().whoami(token=token)
        log.info(f"✅ Successfully authenticated as Hugging Face user: {user['name']}")
    except Exception as e:
        log.error("❌ Hugging Face authentication failed!")
        log.error(f"The token is likely invalid or expired. Original error: {e}")
        log.error("Please generate a new token with 'read' access from https://huggingface.co/settings/tokens")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    """Main function to orchestrate the filtering and downloading process."""
    # --- Configuration and Argument Parsing ---
    cfg = Config()
    parser = build_cli()
    args = parser.parse_args()
    apply_cli_overrides(args, cfg)

    # --- Setup Paths and Authentication ---
    meta_path = Path(cfg.metadata_db_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.download_images:
        verify_hf_auth(cfg)

    # --- Load and Filter Data ---
    df = load_metadata(meta_path, cfg)
    log.info(f"Loaded {len(df):,} records.")

    mask = build_filter_mask(df, cfg)
    df_sub = df[mask].reset_index(drop=True)

    match_count = len(df_sub)
    log.info(f"✅ Found {match_count:,} records matching your criteria.")

    # --- START: NEW LOGIC TO AVOID REDUNDANT WORK ---
    # Check for existing files to avoid re-downloading or re-generating JSONs.
    # This applies if we are downloading or saving per-image JSONs.
    if not cfg.dry_run and (cfg.download_images or (cfg.save_filtered_metadata and cfg.per_image_json)):
        log.info(f"Checking for existing files in {out_dir} to avoid duplicates...")
        try:
            # Create a set of IDs from filenames like '12345.jpg' or '12345.json'.
            # p.stem extracts the '12345' part.
            existing_ids = {int(p.stem) for p in out_dir.iterdir() if p.stem.isdigit()}

            if existing_ids:
                original_count = len(df_sub)
                # Filter the dataframe, keeping only rows whose ID is NOT in the existing set.
                df_sub = df_sub[~df_sub[cfg.id_col].isin(existing_ids)]
                new_count = len(df_sub)
                skipped_count = original_count - new_count
                log.info(f"Skipped {skipped_count:,} items that already exist. {new_count:,} new items remain.")
            else:
                log.info("No existing items found. Proceeding with all matches.")
        except Exception as e:
            log.warning(f"Could not check for existing files due to an error: {e}. Proceeding without this check.")
    # --- END: NEW LOGIC ---

    if cfg.dry_run:
        log.info(f"Dry run: Would attempt to process {len(df_sub):,} new items.")
        return

    if len(df_sub) == 0:
        log.info("No new matching records to process. Exiting.")
        return

    # --- Execution Logic: Download First, then Conditionally Save Metadata ---

    # 1. First, attempt to download images if requested.
    if cfg.download_images:
        download_with_datapool(df_sub, cfg, out_dir)

        # 2. If metadata saving is also enabled, verify which downloads succeeded.
        if cfg.save_filtered_metadata and cfg.per_image_json:
            log.info("Verifying downloaded files before writing JSON metadata...")
            
            # Create a set of integer IDs from files that now exist in the output directory.
            # Using p.stem correctly extracts the filename without the extension (e.g., "12345" from "12345.jpg").
            existing_ids = {int(p.stem) for p in out_dir.iterdir() if p.stem.isdigit()}
            
            if not existing_ids:
                log.warning("No images were successfully downloaded. Skipping metadata generation.")
                return

            # Filter the dataframe to only include rows for images that were actually downloaded.
            df_to_save = df_sub[df_sub[cfg.id_col].isin(existing_ids)]
            
            log.info(f"Found {len(df_to_save):,} downloaded images. Now writing their corresponding metadata files...")
            save_filtered_metadata(df_to_save, cfg, out_dir)
        
        elif cfg.save_filtered_metadata:
             # This handles the case for a single master metadata file (not per-image).
             log.info("Image download complete. Saving master metadata file.")
             save_filtered_metadata(df_sub, cfg, out_dir)

        else:
             log.info("ℹ️ Image download complete. Metadata saving is disabled.")

    # 3. Handle the case where we only save metadata without downloading.
    elif cfg.save_filtered_metadata:
        log.info("Image download is disabled. Saving metadata file(s) now.")
        save_filtered_metadata(df_sub, cfg, out_dir)
    
    else:
        log.info("ℹ️ Image download and metadata saving are both disabled. Script finished.")

if __name__ == "__main__":
    main()