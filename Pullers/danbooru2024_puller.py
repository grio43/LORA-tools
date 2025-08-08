#!/usr/bin/env python3
"""
Enhanced Danbooru dataset filtering and downloading script with:
- Progress tracking and resumability
- Directory sharding for O(1) lookups
- Memory-efficient streaming
- Graceful interruption handling
- Batched JSON writing
- Bounded concurrency
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
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Set

import polars as pl
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
# Progress Tracking
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
                log.info(f"📈 Loaded progress: {len(self.completed_ids):,} completed downloads")
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
            tmp_path.replace(self.progress_file)
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
# Directory Sharding
# ---------------------------------------------------------------------------
class DirectorySharding:
    """Manages single-level directory sharding for O(1) lookups."""
    
    def __init__(self, base_dir: Path, files_per_dir: int = 5000):
        self.base_dir = base_dir
        self.files_per_dir = files_per_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_shard_path(self, image_id: int) -> Path:
        """Get the shard directory path for a given image ID."""
        shard_index = image_id // self.files_per_dir
        shard_name = f"shard_{shard_index:05d}"
        shard_path = self.base_dir / shard_name
        shard_path.mkdir(parents=True, exist_ok=True)
        return shard_path
    
    def file_exists(self, image_id: int) -> bool:
        """Check if file exists in the specific shard directory."""
        shard_path = self.get_shard_path(image_id)
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        for ext in extensions:
            if (shard_path / f"{image_id}{ext}").exists():
                return True
        return False

# ---------------------------------------------------------------------------
# Batched JSON Writer
# ---------------------------------------------------------------------------
class BatchedJSONWriter:
    """Buffers JSON writes and flushes them in batches."""
    
    def __init__(self, flush_interval: float = 5.0, batch_size: int = 100):
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.buffer: List[tuple[Path, Dict[str, Any]]] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self._closed = False
        self._flush_thread = None
        self._start_flush_thread()
    
    def _start_flush_thread(self):
        """Start background thread for periodic flushing."""
        def flush_periodically():
            while not self._closed:
                time.sleep(self.flush_interval)
                with self.lock:
                    if time.time() - self.last_flush >= self.flush_interval:
                        self._flush_buffer()
        
        self._flush_thread = threading.Thread(target=flush_periodically, daemon=True)
        self._flush_thread.start()
    
    def add_write(self, path: Path, data: Dict[str, Any]):
        """Add a JSON write to the buffer."""
        with self.lock:
            if self._closed:
                return
            self.buffer.append((path, data))
            
            # Flush if buffer is full
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush all buffered writes to disk."""
        if not self.buffer:
            return
        
        log.debug(f"💾 Flushing {len(self.buffer)} JSON writes...")
        
        for path, data in self.buffer:
            self._atomic_write_json(path, data)
        
        self.buffer.clear()
        self.last_flush = time.time()
    
    def _atomic_write_json(self, path: Path, data: Dict[str, Any]):
        """Write JSON file atomically."""
        tmp_path = path.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp_path.replace(path)
        except Exception as e:
            log.error(f"❌ Failed to write {path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    def close(self):
        """Close writer and flush remaining data."""
        with self.lock:
            self._closed = True
            self._flush_buffer()

# ---------------------------------------------------------------------------
# Signal Handler
# ---------------------------------------------------------------------------
class SoftStopHandler:
    """Two-stage SIGINT handler: first graceful, second force."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.original_sigint = None
        self._signal_count = 0
        
    def __enter__(self):
        self.original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_sigint)
        
    def _signal_handler(self, signum, frame):
        self._signal_count += 1
        
        if self._signal_count == 1:
            log.warning("\n🛑 Graceful stop requested. Finishing current batch...")
            log.warning("Press Ctrl+C again to force exit.")
            self.stop_event.set()
        else:
            log.warning("\n⚠️ Force exit requested.")
            os._exit(1)
    
    def should_stop(self):
        return self.stop_event.is_set()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Holds all runtime parameters (may be overridden from CLI)."""

    # ---- Paths ------------------------------------------------------------
    metadata_db_path: str = "/media/andrewk/qnap-public/New file/Danbooru2004/metadata.parquet"
    output_dir: str = "/media/andrewk/qnap-public/Training data/"

    # ---- Hugging Face -----------------------------------------------------
    dataset_repo: str = "deepghs/danbooru2024"
    hf_auth_token: Optional[str] = os.getenv("HF_TOKEN", None)

    # ---- Column names -----------------------------------------------------
    tags_col: str = "tag_string"
    character_tags_col: str = "tag_string_character"
    copyright_tags_col: str = "tag_string_copyright"
    artist_tags_col: str = "tag_string_artist"
    score_col: str = "score"
    rating_col: str = "rating"
    width_col: str = "image_width"
    height_col: str = "image_height"
    file_path_col: str = "file_url"
    id_col: str = "id"

    # ---- Filtering Toggles -----------------------------------------------
    enable_include_tags: bool = False
    enable_exclude_tags: bool = False
    enable_character_filtering: bool = False
    enable_copyright_filtering: bool = False
    enable_artist_filtering: bool = False
    enable_score_filtering: bool = False
    enable_rating_filtering: bool = False
    enable_dimension_filtering: bool = False
    per_image_json: bool = True

    # ---- Filtering Criteria -----------------------------------------------
    include_tags: List[str] = field(default_factory=lambda: ["*eye"])
    exclude_tags: List[str] = field(default_factory=lambda: [
        "lowres", "blurry", "pixelated", "jpeg artifacts", "compression artifacts",
        "low quality", "worst quality", "bad quality", "watermark", "signature",
        "artist name", "logo", "stamp", "text", "english_text", "speech bubble",
        "bad anatomy", "bad hands", "bad proportions", "malformed limbs",
        "mutated hands", "extra limbs", "extra fingers", "fused fingers",
        "long neck", "deformed", "disfigured", "mutation", "poorly drawn face",
        "3d", "cgi", "render", "vray", "comic", "2girls", "3girls", "2boys", "3boys",
        "grid", "collage", "multi-panel", "multiple views", "split screen",
        "border", "frame", "out of frame", "cropped", "monochrome", "grayscale",
        "ai generated", "ai art", "ai_generated", "ai_art", "ai_artwork", "ai_image"
    ])
    
    include_characters: List[str] = field(default_factory=lambda: ["hakurei_reimu", "kirisame_marisa"])
    exclude_characters: List[str] = field(default_factory=lambda: ["some_character_to_exclude"])
    include_copyrights: List[str] = field(default_factory=lambda: ["touhou", "genshin_impact"])
    exclude_copyrights: List[str] = field(default_factory=lambda: ["some_series_to_exclude"])
    include_artists: List[str] = field(default_factory=lambda: ["cutesexyrobutts*"])
    exclude_artists: List[str] = field(default_factory=lambda: ["bob"])

    # Other filters
    min_score: Optional[int] = 30
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
    files_per_shard: int = 5000  # New: directory sharding
    batch_size: int = 10000  # New: streaming batch size
    use_streaming: bool = True  # New: enable memory-efficient streaming

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
    p.add_argument("--metadata", type=str, help="Path to Danbooru parquet")
    p.add_argument("--output", type=str, help="Destination directory")
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

    # Behaviour flags
    p.add_argument("--no-download", dest="download", action="store_false", help="Skip downloads")
    p.add_argument("--no-save-metadata", dest="save_meta", action="store_false", help="No metadata")
    p.add_argument("--dry-run", action="store_true", help="Exit after stats")
    p.add_argument("--exclude-gifs", action="store_true", help="Exclude .gif files")
    
    # Performance (new)
    p.add_argument("--workers", type=int, help="Number of download workers")
    p.add_argument("--files-per-shard", type=int, help="Files per directory shard")
    p.add_argument("--batch-size", type=int, help="Streaming batch size")
    p.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming")

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
    if hasattr(args, 'files_per_shard') and args.files_per_shard:
        cfg.files_per_shard = args.files_per_shard
    if hasattr(args, 'batch_size') and args.batch_size:
        cfg.batch_size = args.batch_size
    if hasattr(args, 'streaming') and not args.streaming:
        cfg.use_streaming = False
    
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
# Streaming Metadata
# ---------------------------------------------------------------------------
def create_tag_pattern(tag: str) -> str:
    """Generates a regex pattern based on wildcard usage in the tag."""
    starts_with_star = tag.startswith('*')
    ends_with_star = tag.endswith('*')
    clean_tag = tag.strip('*')
    escaped_tag = re.escape(clean_tag)

    if starts_with_star and ends_with_star:
        return escaped_tag  # Substring match
    elif ends_with_star:
        return r"\b" + escaped_tag  # Prefix match
    elif starts_with_star:
        return escaped_tag + r"\b"  # Suffix match
    else:
        return r"\b" + escaped_tag + r"\b"  # Exact match

def build_polars_filter_expr(cfg: Config) -> pl.Expr:
    """Build unified Polars filter expression for lazy evaluation."""
    filters = []
    
    # File type filtering
    if cfg.exclude_gifs and cfg.file_path_col:
        filters.append(~pl.col(cfg.file_path_col).str.ends_with('.gif'))
    
    excluded_extensions = ('.zip', '.mp4', '.webm', '.swf')
    for ext in excluded_extensions:
        filters.append(~pl.col(cfg.file_path_col).str.ends_with(ext))
    
    # Tag filtering (using Polars expressions)
    if cfg.enable_include_tags and cfg.include_tags:
        for tag in cfg.include_tags:
            pattern = create_tag_pattern(tag)
            filters.append(pl.col(cfg.tags_col).str.contains(pattern))
    
    if cfg.enable_exclude_tags and cfg.exclude_tags:
        for tag in cfg.exclude_tags:
            pattern = create_tag_pattern(tag)
            filters.append(~pl.col(cfg.tags_col).str.contains(pattern))
    
    # Score filtering
    if cfg.enable_score_filtering and cfg.min_score:
        filters.append(pl.col(cfg.score_col) >= cfg.min_score)
    
    # Dimension filtering
    if cfg.enable_dimension_filtering:
        if cfg.square_only:
            filters.append(
                (pl.col(cfg.width_col) == pl.col(cfg.height_col)) &
                (pl.col(cfg.width_col) >= cfg.min_square_size)
            )
        else:
            if cfg.min_width > 0:
                filters.append(pl.col(cfg.width_col) >= cfg.min_width)
            if cfg.min_height > 0:
                filters.append(pl.col(cfg.height_col) >= cfg.min_height)
            if cfg.max_width > 0:
                filters.append(pl.col(cfg.width_col) <= cfg.max_width)
            if cfg.max_height > 0:
                filters.append(pl.col(cfg.height_col) <= cfg.max_height)
    
    # Combine all filters
    if not filters:
        return pl.lit(True)
    return pl.all_horizontal(filters)

def stream_filtered_metadata(path: Path, cfg: Config, stop_handler: Optional[SoftStopHandler] = None) -> Iterator[Dict[str, Any]]:
    """
    Streams filtered metadata using collect(streaming=True) for constant memory.
    Yields individual rows as dictionaries.
    """
    # Build lazy frame
    lf = pl.scan_parquet(str(path))
    
    # Apply columns selection
    cols_to_load: set[str] = set()
    if (cfg.enable_include_tags or cfg.enable_exclude_tags or
            (cfg.save_filtered_metadata and cfg.per_image_json)):
        cols_to_load.update([
            cfg.tags_col,
            cfg.character_tags_col,
            cfg.copyright_tags_col,
            cfg.artist_tags_col,
            "tag_string_meta"
        ])
    
    if cfg.enable_score_filtering:
        cols_to_load.add(cfg.score_col)
    if cfg.enable_rating_filtering or cfg.save_filtered_metadata:
        cols_to_load.add(cfg.rating_col)
    if cfg.enable_dimension_filtering:
        cols_to_load.update([cfg.width_col, cfg.height_col])
    if cfg.download_images:
        cols_to_load.update([cfg.id_col, cfg.file_path_col])
    
    # Get available columns
    available_cols = lf.columns
    final_cols = list(cols_to_load.intersection(available_cols))
    
    if final_cols:
        lf = lf.select(final_cols)
    
    # Apply transformations while still lazy
    numeric_cols = [c for c in (cfg.width_col, cfg.height_col, cfg.score_col) if c in final_cols]
    for col in numeric_cols:
        lf = lf.with_columns(
            pl.col(col)
              .cast(pl.Float64, strict=False)
              .fill_null(0)
              .cast(pl.Int64)
        )
    
    # Normalize tag columns
    if cfg.tags_col in final_cols:
        lf = lf.with_columns(
            pl.col(cfg.tags_col)
              .cast(pl.String)
              .str.to_lowercase()
              .fill_null("")
        )
    
    # Apply filters while still lazy
    filter_expr = build_polars_filter_expr(cfg)
    lf = lf.filter(filter_expr)
    
    try:
        log.info(f"🔄 Starting streaming collection with batch size {cfg.batch_size}")
        
        # Stream in batches for constant memory usage
        batch_count = 0
        total_yielded = 0
        
        for batch_df in lf.collect(streaming=True).iter_slices(cfg.batch_size):
            batch_count += 1
            if batch_count % 10 == 0:
                log.info(f"📊 Processing batch {batch_count} ({total_yielded:,} items so far)...")
            
            # Check for stop signal
            if stop_handler and stop_handler.should_stop():
                log.info("🛑 Stopping stream due to user interrupt...")
                break
            
            # Yield each row in the batch
            for row in batch_df.iter_rows(named=True):
                total_yielded += 1
                yield row
                
    except Exception as e:
        log.error(f"❌ Error during streaming: {e}")
        raise

# ---------------------------------------------------------------------------
# Download Functions
# ---------------------------------------------------------------------------
def normalize_danbooru_rating(rating: str) -> str:
    rating = rating.strip().lower()
    if rating in ["e", "explicit"]:
        return "explicit"
    elif rating in ["q", "questionable"]:
        return "questionable"
    elif rating in ["s", "safe"]:
        return "safe"
    elif rating in ["g", "general"]:
        return "general"
    return "unknown"

def prepare_json_data(row: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Prepare JSON metadata for a single image."""
    file_url = row.get(cfg.file_path_col, "")
    ext = os.path.splitext(file_url)[-1].lower() or ".jpg"
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        ext = ".jpg"
    
    image_id = row[cfg.id_col]
    filename = f"{image_id}{ext}"
    
    rating_raw = str(row.get(cfg.rating_col, "")).strip().lower()
    rating = normalize_danbooru_rating(rating_raw)
    
    tag_cols = [
        cfg.tags_col,
        cfg.character_tags_col,
        cfg.copyright_tags_col,
        cfg.artist_tags_col,
        "tag_string_meta"
    ]
    tags = [row.get(col, "") for col in tag_cols if col in row and isinstance(row.get(col), str)]
    tag_string = " ".join(tags).strip()
    
    return {
        "filename": filename,
        "rating": rating,
        "tags": tag_string
    }

def download_single_with_json(row: Dict[str, Any], cfg: Config, 
                             shard_dir: Path, json_writer: Optional[BatchedJSONWriter],
                             progress_tracker: ProgressTracker, 
                             pool: DataPool) -> bool:
    """Download single image with JSON metadata."""
    image_id = row[cfg.id_col]
    
    try:
        # Download image
        pool.batch_download_to_directory(
            resource_ids=[image_id],
            dst_dir=shard_dir,
            max_workers=1
        )
        
        # Only write JSON after successful download
        if cfg.per_image_json and json_writer:
            json_path = shard_dir / f"{image_id}.json"
            json_data = prepare_json_data(row, cfg)
            json_writer.add_write(json_path, json_data)
        
        # Mark completed
        progress_tracker.mark_completed(image_id)
        return True
        
    except Exception as e:
        log.error(f"❌ Failed {image_id}: {e}")
        return False

def process_downloads_optimized(metadata_stream: Iterator[Dict[str, Any]], 
                               cfg: Config, dest_dir: Path,
                               stop_handler: Optional[SoftStopHandler] = None) -> None:
    """
    Process downloads with all optimizations:
    - Progress tracking & resumability
    - Directory sharding
    - Batched JSON writing
    - Bounded concurrency
    - Graceful interruption
    """
    # Initialize all systems
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)
    json_writer = BatchedJSONWriter(flush_interval=5.0, batch_size=100) if cfg.per_image_json else None
    progress_tracker = ProgressTracker(dest_dir / "progress.json")
    
    if DataPool is None:
        log.error("CheeseChaser not installed. Install with 'pip install cheesechaser'.")
        return
    
    pool = DataPool()
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    try:
        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            futures = []
            max_outstanding = cfg.workers * 10  # Bounded queue
            
            for row in metadata_stream:
                # Check for stop signal
                if stop_handler and stop_handler.should_stop():
                    log.info("🛑 Stopping downloads due to user interrupt...")
                    break
                
                image_id = row[cfg.id_col]
                
                # Skip if already completed
                if progress_tracker.is_completed(image_id):
                    skipped += 1
                    continue
                
                # Check filesystem (O(1) with sharding)
                if sharding.file_exists(image_id):
                    progress_tracker.mark_completed(image_id)
                    skipped += 1
                    continue
                
                # Wait if too many futures outstanding
                while len(futures) >= max_outstanding:
                    # Process some completed futures
                    done_futures = []
                    for future in futures:
                        if future.done():
                            done_futures.append(future)
                            try:
                                if future.result():
                                    downloaded += 1
                                else:
                                    failed += 1
                            except:
                                failed += 1
                    
                    # Remove done futures
                    for f in done_futures:
                        futures.remove(f)
                    
                    if len(futures) >= max_outstanding:
                        time.sleep(0.1)
                
                # Submit download task
                shard_dir = sharding.get_shard_path(image_id)
                future = executor.submit(
                    download_single_with_json,
                    row, cfg, shard_dir, json_writer, progress_tracker, pool
                )
                futures.append(future)
                
                # Progress reporting
                if (downloaded + failed + skipped) % 100 == 0:
                    log.info(f"📥 Progress: {downloaded:,} downloaded, "
                           f"{failed:,} failed, {skipped:,} skipped")
            
            # Wait for remaining futures
            log.info("⏳ Waiting for remaining downloads to complete...")
            for future in as_completed(futures):
                try:
                    if future.result():
                        downloaded += 1
                    else:
                        failed += 1
                except:
                    failed += 1
    
    finally:
        # Cleanup
        if json_writer:
            json_writer.close()
        progress_tracker.save_final()
        log.info(f"✅ Complete: {downloaded:,} downloaded, "
               f"{failed:,} failed, {skipped:,} skipped")

# ---------------------------------------------------------------------------
# Authentication
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

    # --- Use signal handler for graceful interruption ---
    with SoftStopHandler() as stop_handler:
        
        # --- Stream metadata efficiently ---
        if cfg.use_streaming:
            log.info(f"📖 Using memory-efficient streaming from {meta_path}")
            metadata_stream = stream_filtered_metadata(meta_path, cfg, stop_handler)
            
            if cfg.dry_run:
                count = sum(1 for _ in metadata_stream)
                log.info(f"🎯 Dry run: {count:,} images match criteria")
                return
            
            # Process downloads with all optimizations
            if cfg.download_images:
                process_downloads_optimized(metadata_stream, cfg, out_dir, stop_handler)
            elif cfg.save_filtered_metadata:
                log.info("📝 Saving metadata only (download disabled)")
                # Save metadata without downloading
                json_writer = BatchedJSONWriter() if cfg.per_image_json else None
                try:
                    for row in metadata_stream:
                        if stop_handler.should_stop():
                            break
                        if json_writer:
                            json_path = out_dir / f"{row[cfg.id_col]}.json"
                            json_data = prepare_json_data(row, cfg)
                            json_writer.add_write(json_path, json_data)
                finally:
                    if json_writer:
                        json_writer.close()
        else:
            # Fallback to legacy non-streaming mode
            log.warning("⚠️  Using legacy non-streaming mode (not recommended for large datasets)")
            from load_metadata_legacy import load_metadata, build_filter_mask, download_with_datapool
            
            df = load_metadata(meta_path, cfg)
            log.info(f"Loaded {len(df):,} records.")
            
            mask = build_filter_mask(df, cfg)
            df_sub = df[mask].reset_index(drop=True)
            
            match_count = len(df_sub)
            log.info(f"✅ Found {match_count:,} records matching your criteria.")
            
            if cfg.dry_run:
                log.info(f"Dry run: Would process {match_count:,} items.")
                return
            
            if cfg.download_images:
                download_with_datapool(df_sub, cfg, out_dir)

    log.info("🎉 Script completed successfully!")

if __name__ == "__main__":
    main()