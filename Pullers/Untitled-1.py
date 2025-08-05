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
# Directory Sharding System (Requirement #1)
# ---------------------------------------------------------------------------
class DirectorySharding:
    """Manages directory sharding to prevent filesystem performance issues."""
    
    def __init__(self, base_dir: Path, files_per_dir: int = 5000):
        self.base_dir = base_dir
        self.files_per_dir = files_per_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_shard_path(self, image_id: int) -> Path:
        """Get the shard directory path for a given image ID."""
        # Calculate which shard this ID belongs to
        shard_index = image_id // self.files_per_dir
        
        # Create directory structure like /base/0000/0000-4999/
        group_dir = f"{shard_index:04d}"
        start_id = shard_index * self.files_per_dir
        end_id = start_id + self.files_per_dir - 1
        range_dir = f"{start_id:04d}-{end_id:04d}"
        
        shard_path = self.base_dir / group_dir / range_dir
        shard_path.mkdir(parents=True, exist_ok=True)
        
        return shard_path
    
    def find_existing_files(self) -> Set[int]:
        """Recursively scan all shard directories for existing files (Requirement #4)."""
        existing_ids = set()
        
        log.info("🔍 Scanning all subdirectories for existing files...")
        start_time = time.time()
        
        # Scan all subdirectories recursively
        for json_file in self.base_dir.rglob("*.json"):
            if json_file.stem.isdigit():
                existing_ids.add(int(json_file.stem))
        
        scan_time = time.time() - start_time
        log.info(f"📁 Found {len(existing_ids)} existing files in {scan_time:.2f}s")
        
        return existing_ids

# ---------------------------------------------------------------------------
# Batched JSON Writer (Requirement #2)
# ---------------------------------------------------------------------------
class BatchedJSONWriter:
    """Buffers JSON writes and flushes them in batches to improve HDD performance."""
    
    def __init__(self, flush_interval: float = 5.0, batch_size: int = 100):
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.buffer: List[tuple[Path, Dict[str, Any]]] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self._flush_timer = None
        self._start_flush_timer()
    
    def add_write(self, path: Path, data: Dict[str, Any]):
        """Add a JSON write to the buffer."""
        with self.lock:
            self.buffer.append((path, data))
            
            # Flush if buffer is full
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
    
    def _start_flush_timer(self):
        """Start the periodic flush timer."""
        def flush_periodically():
            with self.lock:
                if self.buffer and time.time() - self.last_flush >= self.flush_interval:
                    self._flush_buffer()
            
            # Schedule next flush
            self._flush_timer = threading.Timer(self.flush_interval, flush_periodically)
            self._flush_timer.daemon = True
            self._flush_timer.start()
        
        self._flush_timer = threading.Timer(self.flush_interval, flush_periodically)
        self._flush_timer.daemon = True
        self._flush_timer.start()
    
    def _flush_buffer(self):
        """Flush all buffered writes to disk (Requirement #6 - Sequential access)."""
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
        """Write JSON file atomically using temporary file (Requirement #5)."""
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
    
    def flush_all(self):
        """Force flush all buffered writes."""
        with self.lock:
            self._flush_buffer()
    
    def __del__(self):
        """Cleanup timer on destruction."""
        if self._flush_timer:
            self._flush_timer.cancel()
        self.flush_all()

# ---------------------------------------------------------------------------
# Progress Tracking (Requirement #3)
# ---------------------------------------------------------------------------
class ProgressTracker:
    """Tracks download progress and maintains a progress index file."""
    
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
        """Mark an image as completed and update progress file if needed."""
        with self.lock:
            self.completed_ids.add(image_id)
            self.update_counter += 1
            
            if self.update_counter >= self.update_interval:
                self._save_progress()
                self.update_counter = 0
    
    def _save_progress(self):
        """Save progress to file atomically (Requirement #5)."""
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
# Global soft stop handler
# ---------------------------------------------------------------------------
class SoftStopHandler:
    """Handles graceful shutdown on SIGINT/SIGTERM"""
    def __init__(self):
        self.stop_event = threading.Event()
        self.original_sigint = None
        self.original_sigterm = None
        
    def __enter__(self):
        self.original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self.original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_sigint)
        signal.signal(signal.SIGTERM, self.original_sigterm)
        
    def _signal_handler(self, signum, frame):
        log.warning("\n🛑 Soft stop requested. Finishing current downloads...")
        log.warning("Press Ctrl+C again to force quit.")
        self.stop_event.set()
        # Reset to default handler for force quit
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        
    def should_stop(self):
        return self.stop_event.is_set()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Holds all runtime parameters (may be overridden from CLI)."""

    # ---- Paths ------------------------------------------------------------
    metadata_db_paths: List[str] = field(default_factory=lambda: [
        r"/media/andrewk/qnap-public/New file/rule34_full/combined_full.parquet",
    ])
    output_dir: str = r"/mnt/raid0/Test"

    # ---- Hugging Face -----------------------------------------------------
    dataset_repo: str = "deepghs/rule34_full"
    hf_auth_token: Optional[str] = os.getenv("Add token", None)  # Set your token here or use env var

    # ---- Column names (aligns with DeepGHS conventions) -------------------
    tags_col: str = "tags"
    score_col: str = "score"
    rating_col: str = "rating"
    width_col: str = "width"
    height_col: str = "height"
    filename_col: str = "filename"  # Fixed: Use filename column instead of src_filename
    file_path_col: str = "src_filename"  # Keep this for file filtering
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

    # ---- Filtering Criteria ---------------------------
    include_tags: List[str] = field(default_factory=lambda: ["*cow*"])
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
        "3d", "cgi", "render", "vray", "comic",
        # --- People ---
        "2girls", "3girls", "2boys", "3boys",
        # --- Composition & Framing ---
        "grid", "collage", "multi-panel", "multiple views", "split screen",
        "border", "frame", "out of frame", "cropped",
        # --- Color & Tone ---
        "monochrome", "grayscale",
        # --- AI ---
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
    filtered_metadata_format: str = "json"
    strip_json_details: bool = True
    exclude_gifs: bool = True
    dry_run: bool = False

    # ---- Performance ------------------------------------------------------
    workers: int = 15
    batch_size: int = 1000  # Process metadata in batches to reduce memory usage
    
    # ---- New performance settings ----------------------------------------
    files_per_shard: int = 5000  # Files per directory shard
    json_batch_size: int = 100   # JSON writes per batch
    json_flush_interval: float = 5.0  # Seconds between JSON flushes
    progress_update_interval: int = 100  # Downloads between progress updates

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
# Optimized metadata loading & filtering with streaming
# ---------------------------------------------------------------------------
def stream_filtered_metadata(paths: List[Path], cfg: Config, stop_handler: SoftStopHandler) -> Iterator[Dict[str, Any]]:
    """
    Streams filtered metadata in batches to reduce memory usage.
    Yields individual rows as dictionaries.
    """
    # Build the lazy frame with all filters
    lf = load_and_filter_metadata(paths, cfg)
    
    # Stream results in batches
    batch_size = cfg.batch_size
    offset = 0
    
    while not stop_handler.should_stop():
        # Fetch a batch
        batch_df = lf.slice(offset, batch_size).collect()
        
        if batch_df.height == 0:  # No more data
            break
            
        # Yield each row in the batch
        for row in batch_df.iter_rows(named=True):
            if stop_handler.should_stop():
                break
            yield row
            
        offset += batch_size
        
        # Log progress periodically
        if offset % (batch_size * 10) == 0:
            log.info(f"📊 Processed {offset} metadata entries...")

def load_and_filter_metadata(paths: List[Path], cfg: Config) -> pl.LazyFrame:
    """
    Loads metadata and applies all filters, returning a LazyFrame.
    """
    # Load columns needed
    cols_to_load = set()
    
    # Always include filename, rating, and tags for JSON output
    if cfg.per_image_json or cfg.save_filtered_metadata:
        cols_to_load.update([cfg.filename_col, cfg.rating_col, cfg.tags_col])
    
    if cfg.enable_include_tags or cfg.enable_exclude_tags:
        cols_to_load.add(cfg.tags_col)
    if cfg.enable_score_filtering: cols_to_load.add(cfg.score_col)
    if cfg.enable_rating_filtering: cols_to_load.add(cfg.rating_col)
    if cfg.enable_dimension_filtering: cols_to_load.update([cfg.width_col, cfg.height_col])
    if cfg.download_images or cfg.save_filtered_metadata:
        cols_to_load.update([cfg.id_col, cfg.file_path_col])

    log.info("📖 Setting up lazy metadata scan...")
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
    
    # Apply filters
    lf = apply_filters(lf, cfg)
    
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

    if transformations:
        lf = lf.with_columns(transformations)
        
    return lf

def apply_filters(lf: pl.LazyFrame, cfg: Config) -> pl.LazyFrame:
    """Apply all configured filters."""
    log.info("📝 Applying filters...")
    
    # File type filtering
    if cfg.file_path_col in lf.columns:
        log.info("    Excluding unwanted file types...")
        exclude_extensions = ['.zip', '.mp4', '.webm', '.swf']
        if cfg.exclude_gifs:
            exclude_extensions.append('.gif')
        
        pattern = '|'.join(re.escape(ext) + '$' for ext in exclude_extensions)
        lf = lf.filter(~pl.col(cfg.file_path_col).str.contains(pattern))

    # Tag filtering
    if cfg.tags_col in lf.columns and (cfg.enable_include_tags or cfg.enable_exclude_tags):
        lf = apply_tag_filters(lf, cfg)

    # Score filtering
    if cfg.enable_score_filtering and cfg.min_score is not None:
        log.info(f"    Filtering for score >= {cfg.min_score}")
        lf = lf.filter(pl.col(cfg.score_col) >= cfg.min_score)

    # Rating filtering
    if cfg.enable_rating_filtering and cfg.ratings:
        log.info(f"    Filtering for ratings: {cfg.ratings}")
        lf = lf.filter(pl.col(cfg.rating_col).is_in(cfg.ratings))

    # Dimension filtering
    if cfg.enable_dimension_filtering:
        lf = apply_dimension_filters(lf, cfg)

    return lf

def apply_tag_filters(lf: pl.LazyFrame, cfg: Config) -> pl.LazyFrame:
    """Apply tag inclusion/exclusion filters."""
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

    # Include tags
    if cfg.enable_include_tags and cfg.include_tags:
        if cfg.enable_include_Any_tags:
            log.info(f"    Including tags (ANY): {cfg.include_tags}")
            patterns = [create_pattern(tag) for tag in cfg.include_tags]
            pattern = "|".join(patterns)
            lf = lf.filter(pl.col(cfg.tags_col).str.contains(pattern))
        else:
            log.info(f"    Including tags (ALL): {cfg.include_tags}")
            for tag in cfg.include_tags:
                pattern = create_pattern(tag)
                lf = lf.filter(pl.col(cfg.tags_col).str.contains(pattern))

    # Exclude tags
    if cfg.enable_exclude_tags and cfg.exclude_tags:
        log.info(f"    Excluding tags: {cfg.exclude_tags}")
        patterns = [create_pattern(tag) for tag in cfg.exclude_tags]
        pattern = "|".join(patterns)
        lf = lf.filter(~pl.col(cfg.tags_col).str.contains(pattern))

    return lf

def apply_dimension_filters(lf: pl.LazyFrame, cfg: Config) -> pl.LazyFrame:
    """Apply dimension-based filters."""
    w = pl.col(cfg.width_col)
    h = pl.col(cfg.height_col)
    
    if cfg.square_only:
        log.info(f"    Filtering for square images >= {cfg.min_square_size}px")
        lf = lf.filter((w == h) & (w >= cfg.min_square_size))
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
            log.info(f"    Filtering dimensions: {cfg.min_width}x{cfg.min_height} to {cfg.max_width}x{cfg.max_height}")
            lf = lf.filter(pl.all_horizontal(filters))
    
    return lf

# ---------------------------------------------------------------------------
# Download with integrated JSON generation and all performance improvements
# ---------------------------------------------------------------------------
def download_with_json(row: Dict[str, Any], cfg: Config, sharding: DirectorySharding, 
                      json_writer: BatchedJSONWriter, progress_tracker: ProgressTracker,
                      pool: DataPool, stop_handler: SoftStopHandler) -> bool:
    """
    Downloads a single image and generates its JSON sidecar file.
    Returns True if successful, False otherwise.
    """
    if stop_handler.should_stop():
        return False
        
    image_id = row[cfg.id_col]
    
    # Check progress tracker first (fast memory lookup)
    if progress_tracker.is_completed(image_id):
        log.debug(f"⏭️  Skipping {image_id}: already completed")
        return True
    
    # Get the shard directory for this image
    shard_dir = sharding.get_shard_path(image_id)
    json_path = shard_dir / f"{image_id}.json"
    
    # Double-check if JSON already exists
    if json_path.exists():
        progress_tracker.mark_completed(image_id)
        log.debug(f"⏭️  Skipping {image_id}: already exists")
        return True
    
    try:
        # Download the image using batch_download with single item
        pool.batch_download_to_directory(
            resource_ids=[image_id],
            dst_dir=shard_dir,
            max_workers=1
        )
        
        # Generate JSON sidecar using batched writer
        if cfg.per_image_json:
            json_data = prepare_json_data(row, cfg)
            json_writer.add_write(json_path, json_data)
        
        # Mark as completed in progress tracker
        progress_tracker.mark_completed(image_id)
        
        return True
        
    except Exception as e:
        log.error(f"❌ Failed to download {image_id}: {e}")
        return False

def prepare_json_data(row: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Prepare JSON data for writing - only filename, rating, and tags."""
    json_data = {}
    
    # Add filename using the correct filename column
    if cfg.filename_col in row:
        json_data["filename"] = row[cfg.filename_col]
    
    # Add rating 
    if cfg.rating_col in row:
        json_data["rating"] = row[cfg.rating_col]
        
    # Add tags
    if cfg.tags_col in row:
        json_data["tags"] = row[cfg.tags_col]
    
    return json_data

def process_downloads(metadata_stream: Iterator[Dict[str, Any]], cfg: Config, dest_dir: Path, 
                     pool: DataPool, stop_handler: SoftStopHandler):
    """
    Process downloads using all performance improvements.
    """
    # Initialize performance systems
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)
    json_writer = BatchedJSONWriter(cfg.json_flush_interval, cfg.json_batch_size)
    progress_tracker = ProgressTracker(dest_dir / "progress.json", cfg.progress_update_interval)
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    # Use fast progress tracker lookup instead of filesystem scan
    log.info("🔍 Loading existing progress...")
    existing_count = len(progress_tracker.completed_ids)
    if existing_count > 0:
        log.info(f"📁 Found {existing_count} previously completed downloads")
    
    # Also check filesystem for any files not in progress tracker
    filesystem_existing = sharding.find_existing_files()
    progress_existing = progress_tracker.completed_ids
    
    # Sync progress tracker with filesystem
    new_existing = filesystem_existing - progress_existing
    if new_existing:
        log.info(f"🔄 Syncing {len(new_existing)} files found on filesystem to progress tracker")
        for image_id in new_existing:
            progress_tracker.mark_completed(image_id)
    
    all_existing = progress_tracker.completed_ids
    
    try:
        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            futures = []
            
            for row in metadata_stream:
                if stop_handler.should_stop():
                    log.info("🛑 Stopping download queue...")
                    break
                
                # Skip if already exists (fast memory lookup)
                image_id = row[cfg.id_col]
                if image_id in all_existing:
                    skipped += 1
                    if skipped % 1000 == 0:
                        log.info(f"⏭️  Skipped {skipped} existing files...")
                    continue
                    
                # Submit download task
                future = executor.submit(download_with_json, row, cfg, sharding, 
                                       json_writer, progress_tracker, pool, stop_handler)
                futures.append(future)
                
                # Process completed futures periodically
                if len(futures) >= cfg.workers * 2:
                    completed, futures = process_completed_futures(futures, False)
                    downloaded += sum(1 for r in completed if r)
                    failed += sum(1 for r in completed if not r)
                    
                    if (downloaded + failed) % 100 == 0:
                        log.info(f"📥 Progress: {downloaded} downloaded, {failed} failed, {skipped} skipped")
            
            # Process remaining futures
            if futures:
                log.info("⏳ Waiting for remaining downloads to complete...")
                completed, _ = process_completed_futures(futures, True)
                downloaded += sum(1 for r in completed if r)
                failed += sum(1 for r in completed if not r)
    
    finally:
        # Clean up performance systems
        log.info("🧹 Cleaning up...")
        json_writer.flush_all()
        progress_tracker.save_final()
    
    log.info(f"✅ Download complete: {downloaded} successful, {failed} failed, {skipped} skipped")

def process_completed_futures(futures, wait_all=False):
    """Process completed futures and return results."""
    if wait_all:
        completed_futures = as_completed(futures)
    else:
        completed_futures = [f for f in futures if f.done()]
    
    results = []
    remaining = []
    
    for future in futures:
        if future.done():
            try:
                results.append(future.result())
            except Exception as e:
                log.error(f"Future failed: {e}")
                results.append(False)
        else:
            remaining.append(future)
    
    return results, remaining

def save_filtered_metadata_with_sharding(metadata_list: List[Dict[str, Any]], cfg: Config, dest_dir: Path):
    """Save filtered metadata to file(s) with directory sharding and atomic writes."""
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
            
            # Flush all remaining writes
            json_writer.flush_all()
            log.info(f"💾 Wrote JSON sidecar files with directory sharding")
        
        finally:
            json_writer.flush_all()
    else:
        # Save as single file with atomic write - filter data to only include filename, rating, tags
        output_file = dest_dir / f"filtered_metadata.{cfg.filtered_metadata_format}"
        tmp_file = output_file.with_suffix('.tmp')
        
        try:
            # Filter each item to only include filename, rating, and tags
            processed_list = []
            for item in metadata_list:
                json_data = prepare_json_data(item, cfg)
                processed_list.append(json_data)
            
            with open(tmp_file, 'w', encoding='utf-8') as f:
                for item in processed_list:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            
            # Atomic rename
            tmp_file.rename(output_file)
            log.info(f"💾 Saved metadata to {output_file}")
            
        except Exception as e:
            log.error(f"❌ Failed to save metadata: {e}")
            if tmp_file.exists():
                tmp_file.unlink()

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
    
    # Setup soft stop handler
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
        
        # Stream filtered metadata
        log.info("🔍 Starting metadata filtering...")
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
            # Process downloads with all performance improvements
            log.info("📥 Starting downloads with all performance optimizations...")
            process_downloads(metadata_stream, cfg, dest_dir, pool, stop_handler)
            
        elif cfg.save_filtered_metadata:
            # Just save metadata with sharding if not downloading
            log.info("💾 Saving filtered metadata with performance optimizations...")
            all_metadata = []
            
            # Use progress tracking for metadata-only mode too
            progress_tracker = ProgressTracker(dest_dir / "progress.json", cfg.progress_update_interval)
            existing_count = 0
            
            for item in metadata_stream:
                if stop_handler.should_stop():
                    break
                    
                # Skip if already processed (for metadata-only mode)
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