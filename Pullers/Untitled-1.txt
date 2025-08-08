#!/usr/bin/env python3
"""
A script to filter the R34 dataset based on metadata criteria
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
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict
import shutil

import polars as pl
from huggingface_hub import HfFolder, login

try:
    # Requires cheesechaser >= 0.5.0
    from cheesechaser.datapool import Rule34DataPool as DataPool
except ModuleNotFoundError:  # graceful degradation
    DataPool = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger("cheesechaser_search")

# ---------------------------------------------------------------------------
# Directory Sharding System (Improvement #9 - Single-level O(1) lookup)
# ---------------------------------------------------------------------------
class DirectorySharding:
    """Manages single-level directory sharding for O(1) lookups and shorter paths."""
    
    def __init__(self, base_dir: Path, files_per_dir: int = 5000):
        self.base_dir = base_dir
        self.files_per_dir = files_per_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_shard_path(self, image_id: int) -> Path:
        """Get the shard directory path for a given image ID (O(1) lookup)."""
        shard_index = image_id // self.files_per_dir
        shard_name = f"shard_{shard_index:05d}"
        shard_path = self.base_dir / shard_name
        shard_path.mkdir(parents=True, exist_ok=True)
        return shard_path
    
    def _file_exists(self, image_id: int) -> bool:
        """Check if file exists in the specific shard directory only (Improvement #2)."""
        shard_path = self.get_shard_path(image_id)
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        for ext in extensions:
            if (shard_path / f"{image_id}{ext}").exists():
                return True
        return False
    
    def find_existing_files(self) -> Set[int]:
        """Scan shard directories for existing files."""
        existing_ids = set()
        
        log.info("🔍 Scanning shard directories for existing files...")
        start_time = time.time()
        
        # Scan all shard directories
        for shard_dir in self.base_dir.iterdir():
            if shard_dir.is_dir() and shard_dir.name.startswith("shard_"):
                # Check for JSON files (faster indicator)
                for json_file in shard_dir.glob("*.json"):
                    if json_file.stem.isdigit():
                        existing_ids.add(int(json_file.stem))
        
        scan_time = time.time() - start_time
        log.info(f"📁 Found {len(existing_ids)} existing files in {scan_time:.2f}s")
        
        return existing_ids

# ---------------------------------------------------------------------------
# Fixed Batched JSON Writer (Improvement #7 - Single timer thread)
# ---------------------------------------------------------------------------
class BatchedJSONWriter:
    """Buffers JSON writes and flushes them in batches with exactly one timer thread."""
    
    def __init__(self, flush_interval: float = 5.0, batch_size: int = 100):
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.buffer: List[tuple[Path, Dict[str, Any]]] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self._flush_timer = None
        self._closed = False
        self._start_flush_timer()
    
    def add_write(self, path: Path, data: Dict[str, Any]):
        """Add a JSON write to the buffer."""
        with self.lock:
            if self._closed:
                return
            self.buffer.append((path, data))
            
            # Flush if buffer is full
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
    
    def _start_flush_timer(self):
        """Start the periodic flush timer (exactly one timer)."""
        def flush_periodically():
            with self.lock:
                if not self._closed and self.buffer and time.time() - self.last_flush >= self.flush_interval:
                    self._flush_buffer()
            
            # Schedule next flush if not closed
            if not self._closed:
                self._flush_timer = threading.Timer(self.flush_interval, flush_periodically)
                self._flush_timer.daemon = True
                self._flush_timer.start()
        
        if not self._closed:
            self._flush_timer = threading.Timer(self.flush_interval, flush_periodically)
            self._flush_timer.daemon = True
            self._flush_timer.start()
    
    def _flush_buffer(self):
        """Flush all buffered writes to disk."""
        if not self.buffer:
            return
        
        log.debug(f"💾 Flushing {len(self.buffer)} JSON writes to disk...")
        
        # Sort by directory to improve sequential access
        self.buffer.sort(key=lambda x: str(x[0].parent))
        
        # Write all files
        for path, data in self.buffer:
            self._atomic_write_json(path, data)
        
        self.buffer.clear()
        self.last_flush = time.time()
    
    def _atomic_write_json(self, path: Path, data: Dict[str, Any]):
        """Write JSON file atomically using temporary file."""
        tmp_path = path.with_suffix('.tmp')
        try:
            # Write to temporary file first
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Atomic rename
            tmp_path.rename(path)
        except Exception as e:
            log.error(f"❌ Failed to write {path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    def close(self):
        """Close writer and flush all remaining data."""
        with self.lock:
            self._closed = True
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None
            self._flush_buffer()

# ---------------------------------------------------------------------------
# Progress Tracking (Improvement #1 - Enhanced persistence)
# ---------------------------------------------------------------------------
class ProgressTracker:
    """Tracks download progress with atomic updates and persistent records."""
    
    def __init__(self, progress_file: Path, update_interval: int = 100):
        self.progress_file = progress_file
        self.update_interval = update_interval
        self.completed_ids: Set[int] = set()
        self.update_counter = 0
        self.lock = threading.Lock()
        self._load_progress()
    
    def _load_progress(self):
        """Load existing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_ids = set(data.get('completed_ids', []))
                log.info(f"📈 Loaded progress: {len(self.completed_ids)} completed downloads")
            except Exception as e:
                log.warning(f"⚠️  Failed to load progress file: {e}")
    
    def mark_completed(self, image_id: int):
        """Mark an image as completed and update progress file atomically."""
        with self.lock:
            self.completed_ids.add(image_id)
            self.update_counter += 1
            
            if self.update_counter >= self.update_interval:
                self._save_progress()
                self.update_counter = 0
    
    def _save_progress(self):
        """Save progress to file atomically."""
        data = {
            'completed_ids': sorted(list(self.completed_ids)),
            'last_updated': time.time(),
            'total_completed': len(self.completed_ids)
        }
        
        tmp_path = self.progress_file.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            tmp_path.rename(self.progress_file)
            log.debug(f"📈 Updated progress: {len(self.completed_ids)} completed")
        except Exception as e:
            log.error(f"❌ Failed to save progress: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    def is_completed(self, image_id: int) -> bool:
        """Check if an image has been completed."""
        return image_id in self.completed_ids
    
    def save_final(self):
        """Force save progress file."""
        with self.lock:
            self._save_progress()

# ---------------------------------------------------------------------------
# Two-stage soft stop handler (Improvement #4)
# ---------------------------------------------------------------------------
class SoftStopHandler:
    """Two-stage SIGINT/SIGTERM handler: first graceful stop, second force exit."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.original_sigint = None
        self.original_sigterm = None
        self._signal_count = 0
        
    def __enter__(self):
        self.original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self.original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_sigint)
        signal.signal(signal.SIGTERM, self.original_sigterm)
        
    def _signal_handler(self, signum, frame):
        self._signal_count += 1
        
        if self._signal_count == 1:
            log.warning("\n🛑 Graceful stop requested. Finishing current batch...")
            log.warning("Press Ctrl+C again to force exit.")
            self.stop_event.set()
        else:
            log.warning("\n⚠️ Force exit requested. Stopping immediately...")
            # Second signal forces immediate exit
            os._exit(1)
        
    def should_stop(self):
        return self.stop_event.is_set()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Holds all runtime parameters."""

    # ---- Paths ------------------------------------------------------------
    metadata_db_paths: List[str] = field(default_factory=lambda: [
        r"/media/andrewk/qnap-public/New file/rule34_full/combined_full.parquet",
    ])
    output_dir: str = r"/mnt/raid0/Test"

    # ---- Hugging Face -----------------------------------------------------
    dataset_repo: str = "deepghs/rule34_full"
    hf_auth_token: Optional[str] = os.getenv("Add token", None)

    # ---- Column names -----------------------------------------------------
    tags_col: str = "tags"
    score_col: str = "score"
    rating_col: str = "rating"
    width_col: str = "width"
    height_col: str = "height"
    filename_col: str = "filename"
    file_path_col: str = "src_filename"
    id_col: str = "id"

    # ---- Filtering Toggles ------
    enable_include_tags: bool = False
    enable_exclude_tags: bool = False
    enable_include_Any_tags: bool = False
    enable_character_filtering: bool = False
    enable_copyright_filtering: bool = False
    enable_artist_filtering: bool = False
    enable_score_filtering: bool = False
    enable_rating_filtering: bool = False
    enable_dimension_filtering: bool = False 
    per_image_json: bool = True

    # ---- Filtering Criteria -----------------------------------------------
    include_tags: List[str] = field(default_factory=lambda: ["*cow*"])
    exclude_tags: List[str] = field(default_factory=lambda: [  
        "lowres", "blurry", "pixelated", "jpeg artifacts", "compression artifacts",
        "low quality", "worst quality", "bad quality",
        "watermark", "signature", "artist name", "logo", "stamp",
        "text", "english_text", "speech bubble",
        "bad anatomy", "bad hands", "bad proportions", "malformed limbs",
        "mutated hands", "extra limbs", "extra fingers", "fused fingers",
        "long neck", "deformed", "disfigured", "mutation", "poorly drawn face",
        "3d", "cgi", "render", "vray", "comic",
        "2girls", "3girls", "2boys", "3boys",
        "grid", "collage", "multi-panel", "multiple views", "split screen",
        "border", "frame", "out of frame", "cropped",
        "monochrome", "grayscale",
        "ai generated", "ai art", "ai generated art", "ai generated image", 
        "ai_generated", "ai_art", "ai_artwork", "ai_image", "ai_artwork", 
        "ai artifact", "ai*",
    ])

    include_characters: List[str] = field(default_factory=lambda: ["hakurei_reimu", "kirisame_marisa"])
    exclude_characters: List[str] = field(default_factory=lambda: ["some_character_to_exclude"])
    include_copyrights: List[str] = field(default_factory=lambda: ["touhou", "genshin_impact"])
    exclude_copyrights: List[str] = field(default_factory=lambda: ["some_series_to_exclude"])
    include_artists: List[str] = field(default_factory=lambda: ["cutesexyrobutts*"])
    exclude_artists: List[str] = field(default_factory=lambda: ["bob"])

    min_score: Optional[int] = 140
    ratings: List[str] = field(default_factory=lambda: ["safe", "general"])
    square_only: bool = False
    min_square_size: int = 1024
    min_width: int = 1024
    min_height: int = 1024
    max_width: int = 90000
    max_height: int = 90000

    # ---- Behaviour flags --------------------------------------------------
    download_images: bool = True
    save_filtered_metadata: bool = True
    filtered_metadata_format: str = "jsonl"  # Changed to jsonl
    strip_json_details: bool = True
    exclude_gifs: bool = True
    dry_run: bool = False

    # ---- Performance ------------------------------------------------------
    workers: int = 15
    batch_size: int = 1000
    
    # ---- New performance settings ----------------------------------------
    files_per_shard: int = 5000
    json_batch_size: int = 100
    json_flush_interval: float = 5.0
    progress_update_interval: int = 100
    max_outstanding_multiplier: int = 10  # Bounded futures

# ---------------------------------------------------------------------------
# Rating normalization (Improvement #9)
# ---------------------------------------------------------------------------
def normalize_rating(rating: str) -> str:
    """Normalize rating values to standard form."""
    rating_lower = rating.lower().strip()
    
    if rating_lower in {"s", "safe", "g", "general"}:
        return "safe"
    elif rating_lower in {"q", "questionable"}:
        return "questionable"
    elif rating_lower in {"e", "explicit", "nsfw"}:
        return "explicit"
    else:
        return rating_lower

# ---------------------------------------------------------------------------
# Unified filter builder (Improvement #6)
# ---------------------------------------------------------------------------
def build_polars_filter_expr(cfg: Config) -> pl.Expr:
    """Build a unified Polars filter expression combining all criteria."""
    filters = []
    
    # File type filtering
    if cfg.file_path_col:
        exclude_extensions = ['.zip', '.mp4', '.webm', '.swf']
        if cfg.exclude_gifs:
            exclude_extensions.append('.gif')
        
        if exclude_extensions:
            pattern = '|'.join(re.escape(ext) + '$' for ext in exclude_extensions)
            filters.append(~pl.col(cfg.file_path_col).str.contains(pattern))

    # Tag filtering
    if cfg.tags_col:
        tag_filters = build_tag_filter_expr(cfg)
        if tag_filters is not None:
            filters.append(tag_filters)

    # Score filtering
    if cfg.enable_score_filtering and cfg.min_score is not None:
        filters.append(pl.col(cfg.score_col) >= cfg.min_score)

    # Rating filtering with normalization
    if cfg.enable_rating_filtering and cfg.ratings:
        normalized_ratings = [normalize_rating(r) for r in cfg.ratings]
        # Apply normalization before filtering
        normalized_col = pl.col(cfg.rating_col).map_elements(
            normalize_rating, return_dtype=pl.String
        )
        filters.append(normalized_col.is_in(normalized_ratings))

    # Dimension filtering
    dimension_filter = build_dimension_filter_expr(cfg)
    if dimension_filter is not None:
        filters.append(dimension_filter)

    # Combine all filters using AND logic
    if not filters:
        return pl.lit(True)
    elif len(filters) == 1:
        return filters[0]
    else:
        return pl.all_horizontal(filters)

def build_tag_filter_expr(cfg: Config) -> Optional[pl.Expr]:
    """Build tag filtering expression."""
    def create_pattern(tag: str) -> str:
        """Generate regex pattern for tag matching."""
        starts_with_star = tag.startswith('*')
        ends_with_star = tag.endswith('*')
        clean_tag = tag.strip('*')
        escaped_tag = re.escape(clean_tag)
        
        if starts_with_star and ends_with_star:
            return escaped_tag
        elif ends_with_star:
            return r"(?:^|\s)" + escaped_tag
        elif starts_with_star:
            return escaped_tag + r"(?:$|\s)"
        else:
            return r"(?:^|\s)" + escaped_tag + r"(?:$|\s)"

    tag_filters = []
    
    # Include tags
    if cfg.enable_include_tags and cfg.include_tags:
        patterns = [create_pattern(tag) for tag in cfg.include_tags]
        if cfg.enable_include_Any_tags:
            # ANY logic - match any of the patterns
            pattern = "|".join(patterns)
            tag_filters.append(pl.col(cfg.tags_col).str.contains(pattern))
        else:
            # ALL logic - match all patterns
            for pattern in patterns:
                tag_filters.append(pl.col(cfg.tags_col).str.contains(pattern))

    # Exclude tags
    if cfg.enable_exclude_tags and cfg.exclude_tags:
        patterns = [create_pattern(tag) for tag in cfg.exclude_tags]
        pattern = "|".join(patterns)
        tag_filters.append(~pl.col(cfg.tags_col).str.contains(pattern))

    if not tag_filters:
        return None
    elif len(tag_filters) == 1:
        return tag_filters[0]
    else:
        return pl.all_horizontal(tag_filters)

def build_dimension_filter_expr(cfg: Config) -> Optional[pl.Expr]:
    """Build dimension filtering expression."""
    if not cfg.enable_dimension_filtering:
        return None
        
    w = pl.col(cfg.width_col)
    h = pl.col(cfg.height_col)
    
    if cfg.square_only:
        return (w == h) & (w >= cfg.min_square_size)
    else:
        filters = []
        if cfg.min_width > 0:
            filters.append(w >= cfg.min_width)
        if cfg.min_height > 0:
            filters.append(h >= cfg.min_height)
        if cfg.max_width > 0:
            filters.append(w <= cfg.max_width)
        if cfg.max_height > 0:
            filters.append(h <= cfg.max_height)
        
        if filters:
            return pl.all_horizontal(filters)
    
    return None

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
        description="Filter Rule34 metadata & optionally download matching images.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
            Example Usage:
            --------------
            # Download high-scoring images with specific tags
            python search.py --metadata /data/rule34.parquet --include "1girl solo" --min-score 100 --output ./filtered_images

            # Filter for square images that are at least 1024x1024
            python search.py --square --min-square-size 1024 --exclude "multiple_girls"

            # Perform a dry run to see match count without downloading
            python search.py --include "cat_ears" --dry-run
            """),
    )

    # Paths / IO
    p.add_argument("--metadata", nargs='+', type=str, help="One or more paths to parquet files")
    p.add_argument("--output", type=str, help="Destination directory for downloads & outputs")
    p.add_argument("--repo", type=str, help="Hugging Face dataset repo id")
    p.add_argument("--token", type=str, help="HF token string")

    # Tag filtering
    p.add_argument("--include", "-i", type=_parse_tag_list, help="Tags to include")
    p.add_argument("--exclude", "-x", type=_parse_tag_list, help="Tags to exclude")

    # Other filters
    p.add_argument("--min-score", type=int, help="Minimum score")
    p.add_argument("--ratings", nargs="*", help="Allowed ratings")
    p.add_argument("--square", action="store_true", help="Require square images")
    p.add_argument("--min-square-size", type=int, help="Min dimension for square images")
    p.add_argument("--min-width", type=int)
    p.add_argument("--min-height", type=int)
    p.add_argument("--max-width", type=int)
    p.add_argument("--max-height", type=int)
    p.add_argument("--per-image-json", action="store_true", help="Write one JSON per image")
    p.add_argument("--batch-size", type=int, help="Batch size for processing")

    # Performance settings
    p.add_argument("--files-per-shard", type=int, help="Files per directory shard (default: 5000)")
    p.add_argument("--json-batch-size", type=int, help="JSON writes per batch (default: 100)")
    p.add_argument("--json-flush-interval", type=float, help="Seconds between JSON flushes (default: 5.0)")

    # Behaviour flags
    p.add_argument("--no-download", dest="download", action="store_false", help="Skip downloads")
    p.add_argument("--no-save-metadata", dest="save_meta", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="Exit after printing stats")
    p.add_argument("--exclude-gifs", action="store_true", help="Exclude .gif files")
    p.add_argument("--workers", type=int, help="Number of download workers")

    return p

def apply_cli_overrides(args: argparse.Namespace, cfg: Config) -> None:
    """Mutates cfg in-place based on CLI arguments."""
    if args.metadata: cfg.metadata_db_paths = args.metadata
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
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.workers is not None: cfg.workers = args.workers
    if args.per_image_json: cfg.per_image_json = True
    if args.files_per_shard is not None: cfg.files_per_shard = args.files_per_shard
    if args.json_batch_size is not None: cfg.json_batch_size = args.json_batch_size
    if args.json_flush_interval is not None: cfg.json_flush_interval = args.json_flush_interval
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
# HF Authentication
# ---------------------------------------------------------------------------
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
# Constant-memory metadata streaming (Improvement #5)
# ---------------------------------------------------------------------------
def stream_filtered_metadata(paths: List[Path], cfg: Config, stop_handler: SoftStopHandler) -> Iterator[Dict[str, Any]]:
    """
    Streams filtered metadata using collect(streaming=True).iter_slices() for constant memory.
    Yields individual rows as dictionaries.
    """
    # Build the lazy frame with unified filters
    lf = load_and_filter_metadata(paths, cfg)
    
    try:
        log.info(f"🔄 Starting streaming collection with batch size {cfg.batch_size}")
        
        # Use streaming collection with iter_slices for constant memory usage
        collected = lf.collect(streaming=True)
        batch_count = 0
        
        for batch_df in collected.iter_slices(cfg.batch_size):
            if stop_handler.should_stop():
                break
                
            batch_count += 1
            if batch_count % 10 == 0:
                log.info(f"📊 Processing batch {batch_count}...")
            
            # Yield each row in the batch
            for row in batch_df.iter_rows(named=True):
                if stop_handler.should_stop():
                    break
                yield row
                    
    except Exception as e:
        log.error(f"❌ Error during streaming: {e}")
        raise

def load_and_filter_metadata(paths: List[Path], cfg: Config) -> pl.LazyFrame:
    """
    Loads metadata and applies unified filters, returning a LazyFrame.
    """
    # Load required columns
    cols_to_load = {cfg.id_col, cfg.filename_col, cfg.rating_col, cfg.tags_col, cfg.file_path_col}
    
    # Add conditional columns based on enabled filters
    if cfg.enable_score_filtering: 
        cols_to_load.add(cfg.score_col)
    if cfg.enable_dimension_filtering: 
        cols_to_load.update([cfg.width_col, cfg.height_col])

    log.info("📖 Setting up lazy metadata scan with unified filters...")
    valid_paths = [p for p in paths if p.exists()]
    if not valid_paths:
        log.error("❌ No valid metadata files found.")
        sys.exit(1)

    # Lazily scan all parquet files
    try:
        lf = pl.concat(
            [pl.scan_parquet(p) for p in valid_paths],
            how="vertical_relaxed"
        )
        if cols_to_load:
            lf = lf.select(list(cols_to_load))
    except Exception as e:
        log.error(f"❌ Failed to scan Parquet files: {e}")
        sys.exit(1)

    # Apply transformations
    lf = apply_transformations(lf, cfg)
    
    # Apply unified filters
    unified_filter = build_polars_filter_expr(cfg)
    lf = lf.filter(unified_filter)
    
    return lf

def apply_transformations(lf: pl.LazyFrame, cfg: Config) -> pl.LazyFrame:
    """Apply data type transformations."""
    transformations = []
    
    # Numeric conversions
    for col_name in (cfg.width_col, cfg.height_col, cfg.score_col):
        if col_name in lf.columns:
            transformations.append(
                pl.col(col_name)
                .cast(pl.Float64, strict=False)
                .fill_null(0)
                .cast(pl.Int64)
                .alias(col_name)
            )

    # String normalization
    if cfg.tags_col in lf.columns:
        transformations.append(
            pl.col(cfg.tags_col)
            .cast(pl.String)
            .str.to_lowercase()
            .fill_null("")
            .alias(cfg.tags_col)
        )
    
    # Rating normalization
    if cfg.rating_col in lf.columns:
        transformations.append(
            pl.col(cfg.rating_col)
            .cast(pl.String)
            .str.to_lowercase()
            .fill_null("")
            .alias(cfg.rating_col)
        )

    if transformations:
        lf = lf.with_columns(transformations)
        
    return lf

# ---------------------------------------------------------------------------
# Download with verified completion (Improvement #10)
# ---------------------------------------------------------------------------
def download_with_json(row: Dict[str, Any], cfg: Config, sharding: DirectorySharding, 
                      json_writer: BatchedJSONWriter, progress_tracker: ProgressTracker,
                      pool: DataPool, stop_handler: SoftStopHandler) -> bool:
    """
    Downloads image and writes JSON only after successful download (verified completion).
    Returns True if successful, False otherwise.
    """
    if stop_handler.should_stop():
        return False
        
    image_id = row[cfg.id_col]
    
    # Check progress tracker first (fast memory lookup)
    if progress_tracker.is_completed(image_id):
        log.debug(f"⏭️  Skipping {image_id}: already completed")
        return True
    
    # Use O(1) existence check (Improvement #2)
    if sharding._file_exists(image_id):
        progress_tracker.mark_completed(image_id)
        log.debug(f"⏭️  Skipping {image_id}: file exists")
        return True
    
    try:
        # Get the shard directory for this image
        shard_dir = sharding.get_shard_path(image_id)
        
        # Download the image
        pool.batch_download_to_directory(
            resource_ids=[image_id],
            dst_dir=shard_dir,
            max_workers=1
        )
        
        # Only write JSON AFTER successful download (Improvement #10)
        if cfg.per_image_json:
            json_path = shard_dir / f"{image_id}.json"
            json_data = prepare_json_data(row, cfg)
            json_writer.add_write(json_path, json_data)
        
        # Mark as completed only after successful download
        progress_tracker.mark_completed(image_id)
        
        return True
        
    except Exception as e:
        log.error(f"❌ Failed to download {image_id}: {e}")
        return False

def prepare_json_data(row: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Prepare JSON data for writing - filename, rating, and tags only."""
    json_data = {}
    
    # Add filename
    if cfg.filename_col in row:
        json_data["filename"] = row[cfg.filename_col]
    
    # Add normalized rating
    if cfg.rating_col in row:
        json_data["rating"] = normalize_rating(str(row[cfg.rating_col]))
        
    # Add tags
    if cfg.tags_col in row:
        json_data["tags"] = row[cfg.tags_col]
    
    return json_data

def process_downloads(metadata_stream: Iterator[Dict[str, Any]], cfg: Config, dest_dir: Path, 
                     pool: DataPool, stop_handler: SoftStopHandler):
    """
    Process downloads with bounded futures queue (Improvement #3).
    """
    # Initialize performance systems
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)
    json_writer = BatchedJSONWriter(cfg.json_flush_interval, cfg.json_batch_size)
    progress_tracker = ProgressTracker(dest_dir / "progress.json", cfg.progress_update_interval)
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    # Load existing progress
    log.info("🔍 Loading existing progress...")
    existing_count = len(progress_tracker.completed_ids)
    if existing_count > 0:
        log.info(f"📁 Found {existing_count} previously completed downloads")
    
    # Sync with filesystem
    filesystem_existing = sharding.find_existing_files()
    progress_existing = progress_tracker.completed_ids
    
    new_existing = filesystem_existing - progress_existing
    if new_existing:
        log.info(f"🔄 Syncing {len(new_existing)} files found on filesystem to progress tracker")
        for image_id in new_existing:
            progress_tracker.mark_completed(image_id)
    
    all_existing = progress_tracker.completed_ids
    
    try:
        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            futures = []
            max_outstanding = cfg.workers * cfg.max_outstanding_multiplier  # Bounded queue
            
            for row in metadata_stream:
                if stop_handler.should_stop():
                    log.info("🛑 Stopping download queue...")
                    break
                
                # Skip if already exists
                image_id = row[cfg.id_col]
                if image_id in all_existing:
                    skipped += 1
                    if skipped % 1000 == 0:
                        log.info(f"⏭️  Skipped {skipped} existing files...")
                    continue
                    
                # Wait if too many futures outstanding (bounded queue)
                while len(futures) >= max_outstanding and not stop_handler.should_stop():
                    completed, futures = process_completed_futures(futures, wait_one=True)
                    downloaded += sum(1 for r in completed if r)
                    failed += sum(1 for r in completed if not r)
                    
                    if (downloaded + failed) % 100 == 0:
                        log.info(f"📥 Progress: {downloaded} downloaded, {failed} failed, {skipped} skipped")
                
                # Submit download task
                future = executor.submit(download_with_json, row, cfg, sharding, 
                                       json_writer, progress_tracker, pool, stop_handler)
                futures.append(future)
            
            # Process remaining futures
            if futures:
                log.info("⏳ Waiting for remaining downloads to complete...")
                completed, _ = process_completed_futures(futures, wait_all=True)
                downloaded += sum(1 for r in completed if r)
                failed += sum(1 for r in completed if not r)
    
    finally:
        # Clean up
        log.info("🧹 Cleaning up...")
        json_writer.close()
        progress_tracker.save_final()
    
    log.info(f"✅ Download complete: {downloaded} successful, {failed} failed, {skipped} skipped")

def process_completed_futures(futures, wait_one=False, wait_all=False):
    """Process completed futures and return results."""
    if wait_all:
        completed_futures = list(as_completed(futures))
        remaining = []
    elif wait_one and futures:
        completed_futures = [next(as_completed(futures))]
        remaining = [f for f in futures if f != completed_futures[0]]
    else:
        completed_futures = [f for f in futures if f.done()]
        remaining = [f for f in futures if not f.done()]
    
    results = []
    
    for future in completed_futures:
        try:
            results.append(future.result())
        except Exception as e:
            log.error(f"Future failed: {e}")
            results.append(False)
    
    return results, remaining

def _atomic_write_jsonl(output_file: Path, data_list: List[Dict[str, Any]]):
    """Write JSONL file atomically (Improvement #8)."""
    tmp_file = output_file.with_suffix('.tmp')
    
    try:
        with open(tmp_file, 'w', encoding='utf-8') as f:
            for item in data_list:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        # Atomic rename
        tmp_file.rename(output_file)
        log.info(f"💾 Atomically saved metadata to {output_file}")
        
    except Exception as e:
        log.error(f"❌ Failed to save metadata: {e}")
        if tmp_file.exists():
            tmp_file.unlink()
        raise

def save_filtered_metadata_with_sharding(metadata_list: List[Dict[str, Any]], cfg: Config, dest_dir: Path):
    """Save filtered metadata with atomic writes."""
    if cfg.per_image_json:
        # Use sharding and batched writing for individual JSON files
        sharding = DirectorySharding(dest_dir, cfg.files_per_shard)
        json_writer = BatchedJSONWriter(cfg.json_flush_interval, cfg.json_batch_size)
        
        try:
            for item in metadata_list:
                image_id = item[cfg.id_col]
                shard_dir = sharding.get_shard_path(image_id)
                json_path = shard_dir / f"{image_id}.json"
                
                # Skip if already exists
                if json_path.exists():
                    continue
                
                json_data = prepare_json_data(item, cfg)
                json_writer.add_write(json_path, json_data)
            
            log.info(f"💾 Wrote JSON sidecar files with directory sharding")
        
        finally:
            json_writer.close()
    else:
        # Save as single JSONL file with atomic write
        output_file = dest_dir / "filtered_metadata.jsonl"
        
        # Filter each item to only include essential data
        processed_list = []
        for item in metadata_list:
            json_data = prepare_json_data(item, cfg)
            processed_list.append(json_data)
        
        _atomic_write_jsonl(output_file, processed_list)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Main entry point."""
    # Parse arguments
    parser = build_cli()
    args = parser.parse_args()
    
    # Initialize config
    cfg = Config()
    apply_cli_overrides(args, cfg)
    
    # Validate paths
    metadata_paths = [Path(p) for p in cfg.metadata_db_paths]
    dest_dir = Path(cfg.output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup two-stage soft stop handler
    with SoftStopHandler() as stop_handler:
        
        # Handle HF authentication if needed
        if cfg.download_images:
            verify_hf_auth(cfg)
        
        # Initialize data pool if downloading
        pool = None
        if cfg.download_images:
            if DataPool is None:
                log.error("❌ cheesechaser library not found. Cannot download images.")
                sys.exit(1)
            pool = DataPool()
        
        # Stream filtered metadata with constant memory usage
        log.info("🔍 Starting metadata filtering with streaming...")
        metadata_stream = stream_filtered_metadata(metadata_paths, cfg, stop_handler)
        
        if cfg.dry_run:
            # Count matches for dry run
            count = 0
            for _ in metadata_stream:
                count += 1
                if count % 10000 == 0:
                    log.info(f"Counted {count} matches so far...")
                if stop_handler.should_stop():
                    break
            log.info(f"🎯 Dry run complete: {count} images match your criteria")
            
        elif cfg.download_images and pool:
            # Process downloads with all optimizations
            log.info("📥 Starting downloads with all optimizations...")
            process_downloads(metadata_stream, cfg, dest_dir, pool, stop_handler)
            
        elif cfg.save_filtered_metadata:
            # Save metadata only
            log.info("💾 Saving filtered metadata...")
            all_metadata = []
            
            progress_tracker = ProgressTracker(dest_dir / "progress.json", cfg.progress_update_interval)
            existing_count = 0
            
            for item in metadata_stream:
                if stop_handler.should_stop():
                    break
                    
                # Skip if already processed
                if cfg.per_image_json:
                    image_id = item[cfg.id_col]
                    if progress_tracker.is_completed(image_id):
                        existing_count += 1
                        continue
                    
                all_metadata.append(item)
                if len(all_metadata) % 10000 == 0:
                    log.info(f"Collected {len(all_metadata)} metadata entries...")
            
            if all_metadata:
                save_filtered_metadata_with_sharding(all_metadata, cfg, dest_dir)
                log.info(f"✅ Saved {len(all_metadata)} new entries ({existing_count} already existed)")
            else:
                log.info("ℹ️  No new metadata to save")
        
        else:
            log.info("ℹ️  No action specified. Use --download or --save-metadata")

if __name__ == "__main__":
    main()