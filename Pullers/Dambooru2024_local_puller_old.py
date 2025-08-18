#!/usr/bin/env python3
"""
RAID-Optimized Danbooru dataset filtering and extraction script.
Optimizations for RAID 5 source and RAID 0 destination with high RAM.
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
import tarfile
import mmap
from collections import defaultdict, deque
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Set, Tuple, Callable
from functools import lru_cache
import io

import polars as pl
import psutil

# --- Lightweight zero-copy reader for bytes-like buffers or mmap ---
@dataclass
class Config:
    """Holds all runtime parameters optimized for RAID."""

    # ---- Paths ------------------------------------------------------------
    metadata_db_path: str = "/media/andrewk/qnap-public/danbooru2024/metadata.parquet"
    source_images_dir: str = "/media/andrewk/qnap-public/danbooru2024/images/"
    output_dir: str = "/mnt/raid0/Danbroo/"

    # ---- Column names (unchanged) -----------------------------------------
    tags_col: str = "tag_string"
    character_tags_col: str = "tag_string_character"
    copyright_tags_col: str = "tag_string_copyright"
    artist_tags_col: str = "tag_string_artist"
    score_col: str = "score"
    rating_col: str = "rating"
    width_col: str = "image_width"
    height_col: str = "image_height"
    file_path_col: str = "file_url"
    file_id_col: str = "media_asset.id"  # Not used - kept for compatibility
    id_col: str = "id"  # Primary ID for both finding files and metadata association

    # ---- Filtering (unchanged) --------------------------------------------
    enable_include_tags: bool = False
    enable_exclude_tags: bool = False
    enable_character_filtering: bool = False
    enable_copyright_filtering: bool = False
    enable_artist_filtering: bool = False
    enable_score_filtering: bool = False
    enable_rating_filtering: bool = False
    enable_dimension_filtering: bool = False
    per_image_json: bool = True

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

    min_score: Optional[int] = 30
    ratings: List[str] = field(default_factory=lambda: ["safe", "general"])
    square_only: bool = False
    min_square_size: int = 1024
    min_width: int = 1024
    min_height: int = 1024
    max_width: int = 90000
    max_height: int = 90000

    # ---- Behaviour flags --------------------------------------------------
    extract_images: bool = True
    save_filtered_metadata: bool = True
    filtered_metadata_format: str = "json"
    strip_json_details: bool = True
    exclude_gifs: bool = True
    dry_run: bool = False
    validate_on_start: bool = True
    full_scan: bool = False
    rebuild_tar_index: bool = False

    # ---- Performance settings ---------------------------------------------
    workers: int = 24  # CPU-bound extraction workers
    io_workers: int = 8  # I/O-bound JSON and file writers
    files_per_shard: int = 10000
    batch_size: int = 1000  # Number of metadata rows to read at once
    write_buffer_size_mb: int = 8  # Reasonable write buffer for high-throughput RAID 0
    progress_update_interval: int = 5000

    # Number of worker threads for the asynchronous JSON writer.  A value of zero
    # disables the async writer and falls back to BatchedJSONWriter.
    json_writer_workers: int = 4
    pre_create_shards: bool = True
    enable_fsync: bool = False  # Disable fsync for speed (filesystem will handle it)

    # JSON writer performance
    json_flush_interval: float = 5.0
    on_basename_collision: str = "prefer_top"
    
    # Bound memory usage of not-found sampling
    not_found_max_sample: int = 5000

    # Memory management for large tar files
    use_tar_streaming: bool = False  # Use 'r|' mode for very large tars to reduce memory

# ---------------------------------------------------------------------------
# Directory Sharding
# ---------------------------------------------------------------------------
class DirectorySharding:
    """Manages single-level directory sharding for O(1) lookups."""
    
    def __init__(self, base_dir: Path, files_per_dir: int = 5000):
        """Initialize the sharding system."""
        self.base_dir = base_dir
        self.files_per_dir = files_per_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_shard_path(self, image_id: int, create: bool = False) -> Path:
        """Get the shard directory path for a given image ID.
        If create=True, ensures the directory exists; otherwise does not mkdir."""
        shard_index = image_id // self.files_per_dir
        shard_name = f"shard_{shard_index:05d}"
        shard_path = self.base_dir / shard_name
        if create:
            try:
                shard_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logging.error(f"Failed to create shard directory {shard_path}: {e}")
                raise
        return shard_path
    
    def file_exists(self, image_id: int) -> Tuple[bool, Optional[str]]:
        """
        Check if file exists in the specific shard directory.
        Returns (exists, extension) tuple.
        """
        # Do not create directories during validation/existence checks
        shard_index = image_id // self.files_per_dir
        shard_path = self.base_dir / f"shard_{shard_index:05d}"
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        for ext in extensions:
            if (shard_path / f"{image_id}{ext}").exists():
                return True, ext
        return False, None

# ---------------------------------------------------------------------------
# Batched JSON Writer
# ---------------------------------------------------------------------------
class BatchedJSONWriter:
    """Buffers JSON writes and flushes them in batches."""
    
    def __init__(self, flush_interval: float = 5.0, batch_size: int = 1000, enable_fsync: bool = True):
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.buffer: List[tuple[Path, Dict[str, Any]]] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self.metadata_to_tar_id = {}  # Map metadata row IDs to actual tar IDs
        self.in_flight_writes = []  # Track writes in progress
        self.enable_fsync = enable_fsync
        self._closed = False
        self._flush_thread = None
        self._start_flush_thread()
    
    def _start_flush_thread(self):
        """Start background thread for periodic flushing."""
        def flush_periodically():
            while not self._closed:
                time.sleep(self.flush_interval)
                due = False
                with self.lock:
                    if time.time() - self.last_flush >= self.flush_interval and self.buffer:
                        due = True
                if due:
                    self._flush_buffer()
        
        self._flush_thread = threading.Thread(target=flush_periodically, daemon=True)
        self._flush_thread.start()
    
    def add_write(self, path: Path, data: Dict[str, Any]):
        """Add a JSON write to the buffer."""
        should_flush = False
        with self.lock:
            if self._closed:                    
                return
            self.buffer.append((path, data))
            if len(self.buffer) >= self.batch_size:
                should_flush = True
        if should_flush:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer to disk - fixed to actually write data."""
        to_write: List[tuple[Path, Dict[str, Any]]] = []
        with self.lock:
            if not self.buffer:
                return
            to_write = self.buffer.copy()
            self.buffer.clear()
            self.last_flush = time.time()
            self.in_flight_writes.append(to_write)
        
        logging.debug(f"💾 Flushing {len(to_write)} JSON writes...")
        
        # Do I/O without lock
        for path, data in to_write:
            self._atomic_write_json(path, data)
        
        # Remove from in-flight after completion
        with self.lock:
            if to_write in self.in_flight_writes:
                self.in_flight_writes.remove(to_write)
    
    def _atomic_write_json(self, path: Path, data: Dict[str, Any]):
        """Write JSON file atomically."""
        tmp_path = path.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                if self.enable_fsync:
                    os.fsync(f.fileno())  # Ensure data is on disk before rename                
            tmp_path.replace(path)
        except Exception as e:
            logging.error(f"❌ Failed to write {path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    def close(self):
        """Close writer and flush remaining data."""
        with self.lock:
            self._closed = True
        
        # Flush remaining buffer
        self._flush_buffer()
        
        # Wait for flush thread with timeout
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10.0)
            if self._flush_thread.is_alive():
                logging.warning("⚠️ JSON flush thread did not terminate, forcing flush...")
                # Force flush any remaining in-flight writes
                with self.lock:
                    for to_write in self.in_flight_writes:
                        for path, data in to_write:
                            try:
                                self._atomic_write_json(path, data)
                            except Exception as e:
                                logging.error(f"❌ Failed to force flush {path}: {e}")

# ---------------------------------------------------------------------------
# Asynchronous JSON Writer
# ---------------------------------------------------------------------------
class AsyncJSONWriter:
    """Threaded JSON writer using a queue for thread-safe writes."""

    def __init__(self, workers: int = 4, enable_fsync: bool = False):
        self.queue: queue.Queue[Optional[Tuple[Path, Dict[str, Any]]]] = queue.Queue()
        self.workers = max(1, int(workers))
        self.enable_fsync = enable_fsync
        self._threads: List[threading.Thread] = []
        self._closed = False

        # Start worker threads
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, name=f"json_writer_{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def add_write(self, path: Path, data: Dict[str, Any]):
        """Enqueue a JSON write operation."""
        if self._closed:
            # Ignore writes after closure
            return
        self.queue.put((path, data))

    def _worker(self):
        """Worker thread loop that processes queued JSON writes."""
        while True:
            task = self.queue.get()
            if task is None:
                # Sentinel received: exit thread
                self.queue.task_done()
                break
            path, data = task
            try:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                # Write directly to the target path
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    if self.enable_fsync:
                        os.fsync(f.fileno())
            except Exception as e:
                logging.error(f"❌ AsyncJSONWriter failed to write {path}: {e}")
            finally:
                self.queue.task_done()

    def close(self):
        """Signal all worker threads to stop and wait for them to finish."""
        if self._closed:
            return
        self._closed = True
        # Wait until all pending tasks have been processed
        self.queue.join()
        # Send sentinel tasks to stop workers
        for _ in range(self.workers):
            self.queue.put(None)
        # Ensure sentinel tasks are processed
        self.queue.join()
        for t in self._threads:
            t.join(timeout=5)

# ---------------------------------------------------------------------------
# Validating Progress Tracker
# ---------------------------------------------------------------------------
class ValidatingProgressTracker:
    """Tracks extraction progress with validation and persistent records."""
    
    def __init__(self, progress_file: Path, update_interval: int = 1000, enable_fsync: bool = False):
        self.progress_file = progress_file
        self.update_interval = update_interval
        self.completed_ids: Set[int] = set()
        self.missing_ids: Set[int] = set()
        self.update_counter = 0
        self.lock = threading.Lock()
        self._load_progress()
        self.enable_fsync = enable_fsync
    
    def _load_progress(self):
        """Load existing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_ids = set(data.get('completed_ids', []))
                    self.missing_ids = set(data.get('missing_ids', []))
                logging.info(f"📈 Loaded progress: {len(self.completed_ids):,} completed extractions")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load progress file: {e}")
    
    def mark_completed(self, image_id: int):
        """Mark an image as completed."""
        with self.lock:
            self.completed_ids.add(image_id)
            self.missing_ids.discard(image_id)
            self.update_counter += 1
            
            if self.update_counter >= self.update_interval:
                self._save_progress()
                self.update_counter = 0
    
    def _save_progress(self):
        """Save progress to file atomically."""
        data = {
            'completed_ids': sorted(list(self.completed_ids)),
            'missing_ids': sorted(list(self.missing_ids)),
            'last_updated': time.time(),
            'total_completed': len(self.completed_ids),
            'total_missing': len(self.missing_ids)
        }
        
        tmp_path = self.progress_file.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                # Only fsync if explicitly configured (respect enable_fsync from config)
                if getattr(self, 'enable_fsync', False):
                    os.fsync(f.fileno())       
            tmp_path.replace(self.progress_file)
        except Exception as e:
            logging.error(f"❌ Failed to save progress: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    def is_completed(self, image_id: int) -> bool:
        """Check if an image has been completed."""
        return image_id in self.completed_ids
    
    def save_final(self):
        """Force save progress file."""
        with self.lock:
            self._save_progress()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get current statistics."""
        return {
            'completed': len(self.completed_ids),
            'missing': len(self.missing_ids)
        }
    
    def validate_files(self, sharding: DirectorySharding, auto_clean: bool = True):
        """Validate that all completed files actually exist."""
        logging.info("🔍 Starting file validation...")
        invalid_ids = []
        for image_id in list(self.completed_ids):
            exists, _ = sharding.file_exists(image_id)
            if not exists:
                invalid_ids.append(image_id)
                if auto_clean:
                    self.completed_ids.discard(image_id)
        
        if invalid_ids:
            logging.warning(f"⚠️ Found {len(invalid_ids)} invalid completion records")
            if auto_clean:
                logging.info(f"✅ Cleaned up invalid records")
                self._save_progress()
    
    def full_filesystem_scan(self, sharding: DirectorySharding):
        """Perform full filesystem scan."""
        logging.info("🔍 Performing full filesystem scan...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        found_ids = set()
        for shard_dir in sharding.base_dir.iterdir():
            if shard_dir.is_dir() and shard_dir.name.startswith("shard_"):
                for file_path in shard_dir.iterdir():
                    if (file_path.is_file() and file_path.stem.isdigit() and 
                                            file_path.suffix.lower() in image_extensions):
                        found_ids.add(int(file_path.stem))
        
        logging.info(f"📊 Found {len(found_ids):,} files on filesystem")
        self.completed_ids = found_ids
        self._save_progress()

# ---------------------------------------------------------------------------
# Progress Reporter
# ---------------------------------------------------------------------------
class ProgressReporter:
    """Reports progress at regular intervals."""
    
    def __init__(self, interval: float = 10.0):
        self.interval = interval
        self.last_report = time.time()
        self.last_stats = {}
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def should_report(self) -> bool:
        """Check if it's time to report."""
        return time.time() - self.last_report >= self.interval
    
    def report(self, stats: Dict[str, int], force: bool = False):
        """Report progress."""
        if not force and not self.should_report():
            return
        
        with self.lock:
            elapsed = time.time() - self.start_time
            # Make a thread-safe copy of stats
            stats_copy = stats.copy()

            # Calculate rates
            if self.last_stats and elapsed > 0:
                time_delta = time.time() - self.last_report
                extracted_delta = stats_copy['extracted'] - self.last_stats.get('extracted', 0)
                rate = extracted_delta / time_delta if time_delta > 0 else 0
            else:
                rate = stats_copy['extracted'] / elapsed if elapsed > 0 else 0

            # Memory usage
            process = psutil.Process()
            mem_usage = process.memory_info().rss / 1024**3  # GB

            # Write throughput (only bytes, not mixed units)
            write_mb = stats_copy.get('total_bytes_written', 0) / (1024 * 1024)
            write_time = stats_copy.get('write_time', 0.1)  # Avoid divide by zero
            write_throughput = write_mb / write_time if write_time > 0 else 0

            # Use consistent units (images/s for rate, MB/s for write)
            logging.info(
                f"📊 Progress: {stats_copy['extracted']:,} extracted, "
                f"{stats_copy['failed']:,} failed, {stats_copy['skipped']:,} skipped, "
                f"{stats_copy['not_found']:,} not found | "
                f"Rate: {rate:.1f} img/s | Memory: {mem_usage:.1f}GB | "
                f"Write: {write_throughput:.1f} MB/s | "
                f"Total written: {write_mb:.1f}MB | "
                f"Time: {elapsed/60:.1f}min"
            )

            self.last_report = time.time()
            self.last_stats = stats_copy

# ---------------------------------------------------------------------------
# Signal Handler
# ---------------------------------------------------------------------------
class SoftStopHandler:
    """Two-stage SIGINT handler: first graceful, second force."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.original_sigint = None
        self._signal_count = 0
        self._flush_hooks: List[Callable[[], None]] = []
        self._force_exit_event = threading.Event()
        
    def __enter__(self):
        self.original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_sigint)
        
    def add_flush_hook(self, fn: Callable[[], None]) -> None:
        """Register a hook to flush/flush+fsync before hard-exit."""
        self._flush_hooks.append(fn)
        
    def _signal_handler(self, signum, frame):
        self._signal_count += 1
        
        if self._signal_count == 1:
            logging.warning("\n🛑 Graceful stop requested. Finishing current batch...")
            logging.warning("Press Ctrl+C again to force exit.")
            self.stop_event.set()
        else:
            logging.warning("\n⚠️ Force exit requested.")
            self._force_exit_event.set()
            # Don't block in signal handler - let main thread handle cleanup
            # Set a timer to force exit if main thread doesn't respond
            threading.Timer(2.0, lambda: os._exit(1)).start()
    
    def should_stop(self):
        return self.stop_event.is_set() or self._force_exit_event.is_set()

    def should_force_exit(self):
        return self._force_exit_event.is_set()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def prepare_json_data(row: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Prepare JSON metadata for a single image."""
    file_url = row.get(cfg.file_path_col, "")
    ext = os.path.splitext(file_url)[-1].lower() or ".jpg"
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        ext = ".jpg"
    
    image_id = row[cfg.id_col]  # Use id column to match actual file names
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

# ---------------------------------------------------------------------------
# Tag pattern and filtering functions
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
    if cfg.file_path_col:
        # Handle nulls properly
        file_col = pl.col(cfg.file_path_col).fill_null("")
        
        if cfg.exclude_gifs:
            filters.append(~file_col.str.ends_with('.gif'))
    
        excluded_extensions = ('.zip', '.mp4', '.webm', '.swf')
        for ext in excluded_extensions:
            filters.append(~file_col.str.ends_with(ext))
    
    # Tag filtering (using Polars expressions)
    if cfg.enable_include_tags and cfg.include_tags:
        for tag in cfg.include_tags:
            pattern = create_tag_pattern(tag)
            filters.append(pl.col(cfg.tags_col).str.contains(pattern))
    
    if cfg.enable_exclude_tags and cfg.exclude_tags:
        for tag in cfg.exclude_tags:
            pattern = create_tag_pattern(tag)
            filters.append(~pl.col(cfg.tags_col).str.contains(pattern))
    
    # Character filtering
    if cfg.enable_character_filtering:
        if cfg.include_characters:
            char_filters = []
            for char in cfg.include_characters:
                pattern = create_tag_pattern(char)
                char_filters.append(pl.col(cfg.character_tags_col).str.contains(pattern))
            if char_filters:
                filters.append(pl.any_horizontal(char_filters))
        
        if cfg.exclude_characters:
            for char in cfg.exclude_characters:
                pattern = create_tag_pattern(char)
                filters.append(~pl.col(cfg.character_tags_col).str.contains(pattern))
    
    # Copyright filtering
    if cfg.enable_copyright_filtering:
        if cfg.include_copyrights:
            copy_filters = []
            for copy in cfg.include_copyrights:
                pattern = create_tag_pattern(copy)
                copy_filters.append(pl.col(cfg.copyright_tags_col).str.contains(pattern))
            if copy_filters:
                filters.append(pl.any_horizontal(copy_filters))
        
        if cfg.exclude_copyrights:
            for copy in cfg.exclude_copyrights:
                pattern = create_tag_pattern(copy)
                filters.append(~pl.col(cfg.copyright_tags_col).str.contains(pattern))
    
    # Artist filtering
    if cfg.enable_artist_filtering:
        if cfg.include_artists:
            artist_filters = []
            for artist in cfg.include_artists:
                pattern = create_tag_pattern(artist)
                artist_filters.append(pl.col(cfg.artist_tags_col).str.contains(pattern))
            if artist_filters:
                filters.append(pl.any_horizontal(artist_filters))
        
        if cfg.exclude_artists:
            for artist in cfg.exclude_artists:
                pattern = create_tag_pattern(artist)
                filters.append(~pl.col(cfg.artist_tags_col).str.contains(pattern))
    
    # Score filtering
    if cfg.enable_score_filtering and cfg.min_score is not None:
        filters.append(pl.col(cfg.score_col) >= cfg.min_score)
    
    # Rating filtering
    if cfg.enable_rating_filtering and cfg.ratings:
        rating_filters: list[pl.Expr] = []
        # Work on a normalized, string-typed view to avoid runtime errors (E007)
        rate_col = (
            pl.col(cfg.rating_col)
              .cast(pl.String)
              .str.to_lowercase()
        )
        for rating in cfg.ratings:
            rating_lower = rating.lower()
            if rating_lower in ["safe", "s"]:
                rating_filters.append(rate_col.is_in(["safe", "s"]))
            elif rating_lower in ["general", "g"]:
                rating_filters.append(rate_col.is_in(["general", "g"]))
            elif rating_lower in ["questionable", "q"]:
                rating_filters.append(rate_col.is_in(["questionable", "q"]))
            elif rating_lower in ["explicit", "e"]:
                rating_filters.append(rate_col.is_in(["explicit", "e"]))
        if rating_filters:
            filters.append(pl.any_horizontal(rating_filters))
    
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
    
    # Always load ID and file path for extraction
    cols_to_load.update([cfg.id_col, cfg.file_path_col])
    
    # Load columns needed for filtering
    if cfg.enable_include_tags or cfg.enable_exclude_tags:
        cols_to_load.add(cfg.tags_col)
    
    if cfg.enable_character_filtering:
        cols_to_load.add(cfg.character_tags_col)
    
    if cfg.enable_copyright_filtering:
        cols_to_load.add(cfg.copyright_tags_col)
    
    if cfg.enable_artist_filtering:
        cols_to_load.add(cfg.artist_tags_col)   
    
    if cfg.enable_score_filtering:
        cols_to_load.add(cfg.score_col)
    
    if cfg.enable_rating_filtering or cfg.save_filtered_metadata:
        cols_to_load.add(cfg.rating_col)
    
    if cfg.enable_dimension_filtering:
        cols_to_load.update([cfg.width_col, cfg.height_col])
    
    # Add metadata columns for JSON export
    if cfg.per_image_json:
        cols_to_load.update([
            cfg.tags_col,
            cfg.character_tags_col,
            cfg.copyright_tags_col,
            cfg.artist_tags_col,
            "tag_string_meta"
        ])
    
    # Get available columns
    available_cols = set(lf.schema.keys())
    final_cols = list(cols_to_load.intersection(available_cols))
    
    if final_cols:
        lf = lf.select(final_cols)
    
    # Apply transformations while still lazy
    numeric_cols = [c for c in (cfg.width_col, cfg.height_col, cfg.score_col) if c in final_cols]
    for col in numeric_cols:
    # Handle numeric columns more carefully
        lf = lf.with_columns(
            pl.col(col)
              .cast(pl.Int64, strict=False)  # Try direct int cast first
              .fill_null(-1 if col == cfg.score_col else 0)  # Use -1 for score to distinguish from actual 0
        )

    # Normalize rating column early so downstream .str ops never hit non-strings (E007)
    if cfg.rating_col in final_cols:
        lf = lf.with_columns(
            pl.col(cfg.rating_col)
              .cast(pl.String)
              .fill_null("")
              .str.to_lowercase()
        )
    
    # Normalize tag columns - MUST normalize all columns used for filtering
    tag_cols_to_normalize = []

    # Always normalize columns used for filtering
    if cfg.enable_include_tags or cfg.enable_exclude_tags:
        tag_cols_to_normalize.append(cfg.tags_col)
    if cfg.enable_character_filtering:
        tag_cols_to_normalize.append(cfg.character_tags_col)
    if cfg.enable_copyright_filtering:
        tag_cols_to_normalize.append(cfg.copyright_tags_col)
    if cfg.enable_artist_filtering:
        tag_cols_to_normalize.append(cfg.artist_tags_col)
    
    # Also normalize for JSON output if needed
    if cfg.per_image_json:
        for col in [cfg.tags_col, cfg.character_tags_col, cfg.copyright_tags_col, cfg.artist_tags_col]:
            if col not in tag_cols_to_normalize and col in final_cols:
                tag_cols_to_normalize.append(col)
    
    # Apply normalization
    for col in tag_cols_to_normalize:
        if col in final_cols:
            lf = lf.with_columns(
                pl.col(col)
                .cast(pl.String)
                .str.to_lowercase()
                .fill_null("")
            )
        
    # Apply filters while still lazy
    filter_expr = build_polars_filter_expr(cfg)
    lf = lf.filter(filter_expr)
    
    try:
        logging.info(f"🔥 Starting streaming collection with batch size {cfg.batch_size}")
        
        # Stream in batches for constant memory usage
        batch_count = 0
        total_yielded = 0
        
        try:
            collected = lf.collect(engine="streaming")  # new API
        except TypeError:
            collected = lf.collect(streaming=True)      # fallback for older Polars
        # Handle potential API differences in Polars versions
        if hasattr(collected, "iter_slices"):
            iterator = collected.iter_slices(cfg.batch_size)
        else:
            # Fallback for versions without iter_slices
            total_rows = len(collected)
            iterator = (collected[i:min(i+cfg.batch_size, total_rows)] 
                       for i in range(0, total_rows, cfg.batch_size))
        
        for batch_df in iterator:
            batch_count += 1
            if batch_count % 10 == 0:
                logging.info(f"📊 Processing batch {batch_count} ({total_yielded:,} items so far)...")
            
            # Check for stop signal
            if stop_handler and stop_handler.should_stop():
                logging.info("🛑 Stopping stream due to user interrupt...")
                break
            
            # Yield each row in the batch
            for row in batch_df.iter_rows(named=True):
                total_yielded += 1
                yield row
                
    except Exception as e:
        logging.error(f"❌ Error during streaming: {e}")
        raise

# ---------------------------------------------------------------------------
# Tar Index with memory-optimized tar handling
# ---------------------------------------------------------------------------
class TarIndex:
    """Index for finding images in tar files."""
    
    def __init__(self, source_dir: Path, rebuild: bool = False):
        self.source_dir = source_dir
        self.index = {}  # {file_id: tar_name}
        self.index_paths = {}  # {file_id: internal path within tar (if known)}
        self.lock = threading.Lock()
        
        # Use the existing cache that's already built
        cache_file = source_dir / ".tar_index_cache.json"
        
        if cache_file.exists() and not rebuild:
            self._load_existing_cache(cache_file)
        else:
            logging.info("⌛ No existing cache found. Building new index...")
            self._build_from_json_files()
    
    def _load_existing_cache(self, cache_file: Path):
        """Load the existing tar_index_cache.json."""
        try:
            logging.info("📚 Loading existing tar index cache...")
            with open(cache_file, 'r') as f:
                data = json.load(f)
            # The cache has 'image_to_tar' mapping
            if 'image_to_tar' in data:
                image_to_tar = data['image_to_tar']
                
                # Process the mappings
                for key, tar_name in image_to_tar.items():
                    # Key might be like "8349000.png" or "8349000"
                    # Extract just the numeric ID
                    if '.' in key:
                        # Remove extension
                        file_id_str = key.split('.')[0]
                    else:
                        file_id_str = key
                    
                    try:
                        file_id = int(file_id_str)
                        self.index[file_id] = tar_name
                        if '/' in key:
                            self.index_paths[file_id] = key
                    except ValueError:
                        continue
                
                logging.info(f"✅ Loaded {len(self.index):,} file ID mappings from existing cache")
                
                # Show sample mappings
                samples = list(self.index.items())[:5]
                for fid, tar in samples:
                    logging.info(f"  File ID {fid} -> {tar}")

                # Log total unique tar files
                unique_tars = len(set(self.index.values()))
                logging.info(f"  Unique tar files referenced: {unique_tars}")

        except Exception as e:
            logging.error(f"Failed to load cache: {e}")
            self._build_from_json_files()
    
    def _build_from_json_files(self):
        """Build index from the JSON files next to tar files."""
        logging.info("📚 Building index from JSON files...")
        
        tar_files = sorted(self.source_dir.glob("*.tar"))
        if not tar_files:
            logging.warning(f"⚠️ No tar files found in {self.source_dir}")
            return
        
        # Process each tar's JSON file
        for tar_path in tar_files:  # Process all tar files, not just first 100
            tar_name = tar_path.name
            json_path = tar_path.with_suffix('.json')
            
            if not json_path.exists():
                logging.warning(f"No JSON for {tar_name}")
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if 'files' in data and isinstance(data['files'], dict):
                    # Process the dictionary entries
                    for key in data['files'].keys():
                        # Extract numeric ID from key
                        base_key = key.rsplit('/', 1)[-1]
                        file_id_str = base_key.split('.')[0] if '.' in base_key else base_key
                        try:
                            file_id = int(file_id_str)
                            self.index[file_id] = tar_name
                            if '/' in key:
                                self.index_paths[file_id] = key
                        except ValueError:
                            continue
                                       
            except Exception as e:
                logging.error(f"Failed to process {json_path}: {e}")
        
        logging.info(f"✅ Built index with {len(self.index):,} file ID mappings")
    
    def find_image(self, image_id: int, file_url: str = "") -> Optional[Tuple[str, str]]:
        """
         Find which tar contains an image based on image ID.
        Returns (tar_name, filename_or_fullpath) or None.
        """
        with self.lock:
            tar_name = self.index.get(image_id)
            if not tar_name:
                return None
            internal_path = self.index_paths.get(image_id)

        actual_ext = None
        if internal_path:
            actual_ext = os.path.splitext(internal_path)[1]
        else:
            # Try to get extension/path from the tar's sidecar JSON if available
            json_path = self.source_dir / tar_name.replace('.tar', '.json')
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        files = data.get('files', {})
                        if isinstance(files, dict):
                            for k in files.keys():
                                if not isinstance(k, str):
                                    continue
                                if k.startswith(f"{image_id}.") or k.rsplit('/', 1)[-1].startswith(f"{image_id}."):
                                    actual_ext = os.path.splitext(k)[1]
                                    internal_path = k  # Prefer discovered full path
                                    with self.lock:
                                        self.index_paths[image_id] = internal_path
                                    break
                except Exception:
                    pass

        # Fall back to URL extension or default
        ext = actual_ext or '.jpg'
        if file_url and not actual_ext:
            ext_match = re.search(r'\.(jpg|jpeg|png|gif|webp)$', file_url, re.IGNORECASE)
            if ext_match:
                ext = ext_match.group(0)

        filename = internal_path if internal_path else f"{image_id}{ext}"
        return (tar_name, filename)

    def cache_discovered_path(self, image_id: int, tar_name: str, internal_path: str) -> None:
        """Cache a discovered full internal path for future lookups."""
        with self.lock:
            self.index[image_id] = tar_name
            self.index_paths[image_id] = internal_path

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics for debugging."""
        stats = {
            'total_images': len(self.index),
            'sample_mappings': dict(list(self.index.items())[:10]) if self.index else {}
        }
        return stats

# ---------------------------------------------------------------------------
# CLI Functions
# ---------------------------------------------------------------------------
def _parse_tag_list(value: str) -> List[str]:
    """Parses comma- or space-separated tag strings into a list."""
    return [t for t in re.split(r"[\s,]+", value.strip()) if t]

def build_cli() -> argparse.ArgumentParser:
    """Defines and configures the command-line argument parser."""
    p = argparse.ArgumentParser(
        prog="danbooru-local-puller",
        description="Filter and extract Danbooru images from local tar archives with RAID optimizations.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
            Example Usage:
            --------------
            """),
    )

    # Paths
    p.add_argument("--metadata", type=str, help="Path to Danbooru parquet")
    p.add_argument("--source", type=str, help="Source directory with tar files")
    p.add_argument("--output", type=str, help="Destination directory")

    # Tag filtering
    p.add_argument("--include", "-i", type=_parse_tag_list, help="Tags to include (enables filter)")
    p.add_argument("--exclude", "-x", type=_parse_tag_list, help="Tags to exclude (enables filter)")

    # Character/Copyright/Artist filtering
    p.add_argument("--include-characters", type=_parse_tag_list, help="Characters to include")
    p.add_argument("--exclude-characters", type=_parse_tag_list, help="Characters to exclude")
    p.add_argument("--include-copyrights", type=_parse_tag_list, help="Copyrights to include")
    p.add_argument("--exclude-copyrights", type=_parse_tag_list, help="Copyrights to exclude")
    p.add_argument("--include-artists", type=_parse_tag_list, help="Artists to include")
    p.add_argument("--exclude-artists", type=_parse_tag_list, help="Artists to exclude")

    # Other filters
    p.add_argument("--min-score", type=int, help="Minimum score (enables filter)")
    p.add_argument("--ratings", nargs="*", help="Allowed ratings (enables filter)")
    p.add_argument("--square", action="store_true", help="Require square images (enables filter)")
    p.add_argument("--min-square-size", type=int, help="Min dimension for square images")
    p.add_argument("--min-width", type=int, help="Minimum width (enables filter)")
    p.add_argument("--min-height", type=int, help="Minimum height (enables filter)")
    p.add_argument("--max-width", type=int, help="Maximum width")
    p.add_argument("--max-height", type=int, help="Maximum height")

    # Behaviour flags
    p.add_argument("--no-extract", dest="extract", action="store_false", help="Skip extraction")
    p.add_argument("--no-json", dest="json", action="store_false", help="Don't write JSON metadata")
    p.add_argument("--dry-run", action="store_true", help="Show matches without extracting")
    p.add_argument("--include-gifs", action="store_true", help="Include .gif files (excluded by default)")
    p.add_argument("--no-validate", dest="validate", action="store_false", help="Skip validation")
    p.add_argument("--full-scan", action="store_true", help="Perform full filesystem scan")
    p.add_argument("--rebuild-index", action="store_true", help="Rebuild tar index")

    # Performance
    p.add_argument("--workers", type=int, help="Number of extraction workers")
    p.add_argument("--io-workers", type=int, help="Number of I/O workers")
    p.add_argument("--batch-size", type=int, help="Streaming batch size")
    p.add_argument("--use-tar-streaming", action="store_true", help="Use streaming mode for large tars to reduce memory")

    # Additional tuning options
    p.add_argument(
        "--progress-update-interval",
        type=int,
        help="Number of completed images between progress file updates (default: based on Config)"
    )
    p.add_argument(
        "--json-writer-workers",
        type=int,
        help="Number of threads for the asynchronous JSON writer; 0 to disable and use the batched writer"
    )
    p.add_argument(
        "--pre-create-shards",
        action="store_true",
        help="Pre-create shard directories ahead of extraction to reduce metadata overhead"
    )

    return p

def apply_cli_overrides(args: argparse.Namespace, cfg: Config) -> None:
    """Mutates cfg in-place based on CLI arguments."""
    if args.metadata:
        cfg.metadata_db_path = args.metadata
    if args.source:
        cfg.source_images_dir = args.source
    if args.output:
        cfg.output_dir = args.output

    # Tag filtering - enable filter if tags provided
    if args.include is not None:
        cfg.include_tags = args.include
        cfg.enable_include_tags = True
    if args.exclude is not None:
        cfg.exclude_tags = args.exclude
        cfg.enable_exclude_tags = True

    # Character filtering - enable if provided
    if args.include_characters is not None:
        cfg.include_characters = args.include_characters
        cfg.enable_character_filtering = True
    if args.exclude_characters is not None:
        cfg.exclude_characters = args.exclude_characters
        cfg.enable_character_filtering = True

    # Copyright filtering - enable if provided
    if args.include_copyrights is not None:
        cfg.include_copyrights = args.include_copyrights
        cfg.enable_copyright_filtering = True
    if args.exclude_copyrights is not None:
        cfg.exclude_copyrights = args.exclude_copyrights
        cfg.enable_copyright_filtering = True

    # Artist filtering - enable if provided
    if args.include_artists is not None:
        cfg.include_artists = args.include_artists
        cfg.enable_artist_filtering = True
    if args.exclude_artists is not None:
        cfg.exclude_artists = args.exclude_artists
        cfg.enable_artist_filtering = True

    # Score filtering - enable if provided
    if args.min_score is not None:
        cfg.min_score = args.min_score
        cfg.enable_score_filtering = True

    # Rating filtering - enable if provided
    if args.ratings is not None:
        cfg.ratings = args.ratings
        cfg.enable_rating_filtering = True

    # Dimension filtering - enable if constraints provided
    if args.square:
        cfg.square_only = True
        cfg.enable_dimension_filtering = True
    if args.min_square_size is not None:
        cfg.min_square_size = args.min_square_size
        cfg.enable_dimension_filtering = True
    if args.min_width is not None:
        cfg.min_width = args.min_width
        cfg.enable_dimension_filtering = True
    if args.min_height is not None:
        cfg.min_height = args.min_height
        cfg.enable_dimension_filtering = True
    if args.max_width is not None:
        cfg.max_width = args.max_width
        cfg.enable_dimension_filtering = True
    if args.max_height is not None:
        cfg.max_height = args.max_height
        cfg.enable_dimension_filtering = True

    # Behaviour flags
    if hasattr(args, 'extract') and not args.extract:
        cfg.extract_images = False
    if hasattr(args, 'json') and not args.json:
        cfg.per_image_json = False
    if args.dry_run:
        cfg.dry_run = True
        cfg.extract_images = False
    if args.include_gifs:
        cfg.exclude_gifs = False  # Include GIFs when flag is set
    if hasattr(args, 'validate') and not args.validate:
        cfg.validate_on_start = False
    if args.full_scan:
        cfg.full_scan = True
    if args.rebuild_index:
        cfg.rebuild_tar_index = True

    # Performance
    if args.workers is not None:
        cfg.workers = args.workers
    if args.io_workers is not None:
        cfg.io_workers = args.io_workers
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if hasattr(args, 'use_tar_streaming') and args.use_tar_streaming:
        cfg.use_tar_streaming = True

    # New configurable fields
    if hasattr(args, 'progress_update_interval') and args.progress_update_interval is not None:
        cfg.progress_update_interval = args.progress_update_interval
    if hasattr(args, 'json_writer_workers') and args.json_writer_workers is not None:
        cfg.json_writer_workers = args.json_writer_workers
    if hasattr(args, 'pre_create_shards') and args.pre_create_shards:
        cfg.pre_create_shards = True

# ---------------------------------------------------------------------------
# Main extraction functions with memory-optimized tar handling
# ---------------------------------------------------------------------------
def process_extractions_simple(metadata_stream: Iterator[Dict[str, Any]],
                              cfg: Config, dest_dir: Path,
                              tar_index,
                              stop_handler: Optional[Any] = None) -> None:
    """Process extractions with simplified RAID optimizations and memory-aware tar handling."""

    # Initialize components
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)

    # Initialize JSON writer: use asynchronous writer if configured
    if cfg.per_image_json:
        if cfg.json_writer_workers and cfg.json_writer_workers > 0:
            json_writer: Optional[object] = AsyncJSONWriter(workers=cfg.json_writer_workers, enable_fsync=cfg.enable_fsync)
        else:
            json_writer = BatchedJSONWriter(flush_interval=cfg.json_flush_interval, batch_size=1000, enable_fsync=cfg.enable_fsync)
    else:
        json_writer = None

    # Progress tracker
    progress_tracker = ValidatingProgressTracker(dest_dir / "progress.json", update_interval=cfg.progress_update_interval, enable_fsync=cfg.enable_fsync)

    # Validation
    if cfg.validate_on_start:
        if cfg.full_scan:
            progress_tracker.full_filesystem_scan(sharding)
        else:
            progress_tracker.validate_files(sharding, auto_clean=True)

    # Statistics
    stats = {
        'extracted': 0,
        'failed': 0,
        'skipped': 0,
        'not_found': 0,
        'not_found_ids': deque(maxlen=cfg.not_found_max_sample),
        'total_bytes_written': 0,
        'write_time': 0.0
    }
    stats_lock = threading.Lock()

    progress_reporter = ProgressReporter(interval=10.0)

    # Register flush hooks for interruption
    if stop_handler:
        if json_writer:
            stop_handler.add_flush_hook(lambda: json_writer.close())
        stop_handler.add_flush_hook(lambda: progress_tracker.save_final())

    # Pre-create shard directories if configured
    if cfg.pre_create_shards:
        logging.info("📁 Pre-creating shard directories...")
        # Estimate the range of IDs we might encounter
        # This is a simple implementation - could be improved with metadata scanning
        max_estimated_id = 10000000  # Reasonable upper bound for Danbooru
        max_shard = max_estimated_id // cfg.files_per_shard
        for shard_idx in range(max_shard + 1):
            shard_dir = sharding.base_dir / f"shard_{shard_idx:05d}"
            if not shard_dir.exists():
                shard_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"✅ Pre-created up to shard_{max_shard:05d}")


    # Show starting stats
    initial_stats = progress_tracker.get_statistics()
    logging.info(f"📊 Starting with {initial_stats['completed']:,} completed, {initial_stats['missing']:,} missing")
    logging.info(f"🚀 Using {cfg.workers} extraction workers, {cfg.io_workers} I/O workers")
    logging.info(f"💽 RAID optimized: {cfg.write_buffer_size_mb}MB write buffers")
    if cfg.use_tar_streaming:
        logging.info("🔄 Using streaming tar mode for reduced memory usage")

    # Group images by tar file
    logging.info("📦 Grouping images by tar file...")
    tar_groups: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = defaultdict(list)
    total_collected = 0

    try:
        # Collect all images grouped by tar
        for row in metadata_stream:
            if stop_handler and stop_handler.should_stop():
                logging.info("🛑 Stopping collection due to user interrupt...")
                break
            image_id = row[cfg.id_col]

            # Skip if already completed
            if progress_tracker.is_completed(image_id):
                with stats_lock:
                    stats['skipped'] += 1
                continue
            # Skip if file already exists
            file_exists, _ = sharding.file_exists(image_id)
            if file_exists:
                progress_tracker.mark_completed(image_id)
                with stats_lock:
                    stats['skipped'] += 1
                continue
            # Find tar info
            file_url = row.get(cfg.file_path_col, "")
            tar_info = tar_index.find_image(image_id, file_url)
            if not tar_info:
                with stats_lock:
                    stats['not_found'] += 1
                    stats['not_found_ids'].append(image_id)
                continue
            tar_name, filename = tar_info
            tar_groups[tar_name].append((image_id, filename, row))
            total_collected += 1
            if total_collected % 10000 == 0:
                logging.info(f"📊 Collected {total_collected:,} images across {len(tar_groups)} tar files")

        logging.info(f"📦 Collected {total_collected:,} images from {len(tar_groups)} tar files")

        # Process each tar sequentially
        for tar_name, images in tar_groups.items():
            if stop_handler and stop_handler.should_stop():
                logging.info("🛑 Stopping extraction due to user interrupt...")
                break
            # Check for force exit
            if stop_handler and stop_handler.should_force_exit():
                logging.warning('⚠️ Force exit detected, saving progress...')
                break
            process_single_tar(cfg, tar_name, images, sharding, progress_tracker, json_writer, tar_index, stats, stats_lock)
            # Report progress
            if progress_reporter.should_report():
                with stats_lock:
                    progress_reporter.report(stats)

        # Final report
        with stats_lock:
            progress_reporter.report(stats, force=True)

        # Persist not-found sample
        try:
            not_found_sample = list(stats.get('not_found_ids', []))
            if not_found_sample:
                with open(dest_dir / "not_found_ids.sample.json", "w", encoding="utf-8") as f:
                    json.dump({"count": stats.get('not_found', 0),
                               "sample_size": len(not_found_sample),
                               "ids": not_found_sample}, f, indent=2)
        except Exception as e:
            logging.warning(f"⚠️ Failed to persist not-found sample: {e}")

        logging.info(
            f"✅ Extraction complete: {stats['extracted']:,} extracted, "
            f"{stats['failed']:,} failed, {stats['skipped']:,} skipped, "
            f"{stats['not_found']:,} not found"
        )
    finally:
        # Cleanup
        if json_writer:
            json_writer.close()
        progress_tracker.save_final()

def process_single_tar(cfg: Config, tar_name: str, images: List[Tuple[int, str, Dict[str, Any]]],
                       sharding: DirectorySharding, progress_tracker: ValidatingProgressTracker,
                       json_writer: Optional[object], tar_index, stats: Dict[str, Any], stats_lock: threading.Lock) -> None:
    """Process a single tar file with memory-optimized handling."""
    tar_path = Path(cfg.source_images_dir) / tar_name
    logging.info(f"📂 Processing {tar_name} with {len(images)} images")
    
    # Monitor memory usage before opening tar
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**3  # GB
    
    try:
        # Choose tar mode based on configuration
        tar_mode = 'r|' if cfg.use_tar_streaming else 'r'
        
        if cfg.use_tar_streaming:
            # Streaming mode: Lower memory but cannot random-access
            process_tar_streaming(cfg, tar_path, tar_mode, images, sharding, progress_tracker, json_writer, tar_index, stats, stats_lock)
        else:
            # Random access mode: Higher memory but faster for multiple files
            with tarfile.open(tar_path, tar_mode) as tar:
                # Monitor memory after opening
                mem_after = process.memory_info().rss / 1024**3
                memory_increase = mem_after - mem_before
                
                if memory_increase > 2.0:  # If memory increased by more than 2GB
                    logging.warning(f"⚠️ High memory usage detected ({memory_increase:.1f}GB increase). Consider using --use-tar-streaming for large tars.")
                
                members = {m.name: m for m in tar}
                # Use a thread pool only for writing to disk
                with ThreadPoolExecutor(max_workers=cfg.io_workers) as executor:
                    futures = []
                    for image_id, filename, row in images:
                        # Locate member
                        member = members.get(filename)
                        if member is None:
                            # Fallback to basename match
                            basename = filename.rsplit('/', 1)[-1]
                            for m_name, m in members.items():
                                if m_name.rsplit('/', 1)[-1] == basename:
                                    member = m
                                    break
                        if member is None:
                            with stats_lock:
                                stats['not_found'] += 1
                            continue
                        # Extract file data
                        try:
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                data = file_obj.read()
                                file_obj.close()
                                # Schedule saving to disk
                                future = executor.submit(
                                    save_file, cfg, image_id, data, row, member, sharding,
                                    progress_tracker, json_writer, tar_index, tar_name, stats, stats_lock
                                )
                                futures.append(future)
                        except Exception as e:
                            logging.error(f"Failed to extract {filename}: {e}")
                            with stats_lock:
                                stats['failed'] += 1
                    # Wait for all writes
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logging.error(f"Write task failed: {e}")
                            with stats_lock:
                                stats['failed'] += 1
    except Exception as e:
        logging.error(f"Failed to process tar {tar_name}: {e}")
        with stats_lock:
            stats['failed'] += len(images)

def process_tar_streaming(cfg: Config, tar_path: Path, tar_mode: str, images: List[Tuple[int, str, Dict[str, Any]]],
                         sharding: DirectorySharding, progress_tracker: ValidatingProgressTracker,
                         json_writer: Optional[object], tar_index, stats: Dict[str, Any], stats_lock: threading.Lock) -> None:
    """Process tar in streaming mode for lower memory usage."""
    # Create lookup for images we need
    needed_files = {}
    for image_id, filename, row in images:
        basename = filename.rsplit('/', 1)[-1]
        needed_files[filename] = (image_id, row)
        needed_files[basename] = (image_id, row)  # Also try basename
    
    try:
        with tarfile.open(tar_path, tar_mode) as tar:
            with ThreadPoolExecutor(max_workers=cfg.io_workers) as executor:
                futures = []
                for member in tar:
                    if member.isfile():
                        # Check if this is a file we need
                        match = needed_files.get(member.name)
                        if not match:
                            # Try basename match
                            basename = member.name.rsplit('/', 1)[-1]
                            match = needed_files.get(basename)
                        
                        if match:
                            image_id, row = match
                            try:
                                file_obj = tar.extractfile(member)
                                if file_obj:
                                    data = file_obj.read()
                                    file_obj.close()
                                    # Schedule saving to disk
                                    future = executor.submit(
                                        save_file, cfg, image_id, data, row, member, sharding,
                                        progress_tracker, json_writer, tar_index, tar_path.name, stats, stats_lock
                                    )
                                    futures.append(future)
                            except Exception as e:
                                logging.error(f"Failed to extract {member.name}: {e}")
                                with stats_lock:
                                    stats['failed'] += 1
                
                # Wait for all writes
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Write task failed: {e}")
                        with stats_lock:
                            stats['failed'] += 1
    except Exception as e:
        logging.error(f"Failed to process streaming tar {tar_path}: {e}")
        with stats_lock:
            stats['failed'] += len(images)

def save_file(cfg: Config, image_id: int, data: bytes, row: Dict[str, Any], member: tarfile.TarInfo,
              sharding: DirectorySharding, progress_tracker: ValidatingProgressTracker,
              json_writer: Optional[object], tar_index, tar_name: str, stats: Dict[str, Any], stats_lock: threading.Lock) -> None:
    """Save a single file to disk and update statistics and progress."""
    try:
        start_write = time.time()
        # Determine extension and paths
        ext = os.path.splitext(member.name)[1] or '.jpg'
        shard_path = sharding.get_shard_path(image_id, create=True)
        final_path = shard_path / f"{image_id}{ext}"
        temp_path = final_path.with_suffix(final_path.suffix + '.tmp')
        # Write file
        buffer_mb = min(max(1, cfg.write_buffer_size_mb), 64)
        buffer_size_bytes = buffer_mb * 1024 * 1024
        with open(temp_path, 'wb', buffering=buffer_size_bytes) as f:
            f.write(data)
            if cfg.enable_fsync:
                f.flush()
                os.fsync(f.fileno())
        temp_path.replace(final_path)
        # Update stats
        with stats_lock:
            stats['total_bytes_written'] += len(data)
            stats['write_time'] += (time.time() - start_write)
            stats['extracted'] += 1
        # Write JSON metadata if configured
        if cfg.per_image_json and json_writer:
            json_path = shard_path / f"{image_id}.json"
            json_data = prepare_json_data(row, cfg)
            json_writer.add_write(json_path, json_data)
        # Cache discovered path
        if hasattr(tar_index, 'cache_discovered_path'):
            tar_index.cache_discovered_path(image_id, tar_name, member.name)
        # Mark completed
        progress_tracker.mark_completed(image_id)
    except Exception as e:
        logging.error(f"Failed to save file {image_id}: {e}")
        with stats_lock:
            stats['failed'] += 1

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    """Main function to orchestrate the filtering and extraction process."""
    
    # Configuration and argument parsing
    cfg = Config()
    parser = build_cli()
    args = parser.parse_args()
    apply_cli_overrides(args, cfg)
    
    # Setup logging
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    
    # Log active filters if none specified via CLI (using defaults)
    if not any([args.include, args.exclude, args.include_characters, args.exclude_characters,
                args.include_copyrights, args.exclude_copyrights, args.include_artists, 
                args.exclude_artists, args.min_score, args.ratings, args.min_width, args.min_height]):
        logging.info("📋 No filters specified via CLI, using default configuration:")
        if cfg.enable_include_tags:
            logging.info(f"  Include tags: {cfg.include_tags}")
        if cfg.enable_exclude_tags:
            logging.info(f"  Exclude tags: {cfg.exclude_tags[:10]}..." if len(cfg.exclude_tags) > 10 else f"  Exclude tags: {cfg.exclude_tags}")
        if cfg.enable_score_filtering:
            logging.info(f"  Min score: {cfg.min_score}")
        if cfg.enable_rating_filtering:
            logging.info(f"  Ratings: {cfg.ratings}")
        if cfg.enable_dimension_filtering:
            logging.info(f"  Min dimensions: {cfg.min_width}x{cfg.min_height}")
    
    # Setup paths
    meta_path = Path(cfg.metadata_db_path)
    source_dir = Path(cfg.source_images_dir)
    out_dir = Path(cfg.output_dir)
    
    if not meta_path.exists():
        logging.error(f"❌ Metadata file not found: {meta_path}")
        sys.exit(1)
    
    if not source_dir.exists():
        logging.error(f"❌ Source directory not found: {source_dir}")
        sys.exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build tar index
    logging.info("🗂️ Initializing tar index...")
    rebuild_index = cfg.rebuild_tar_index
    tar_index = TarIndex(source_dir, rebuild=rebuild_index)

    # Debug: Show index statistics
    index_stats = tar_index.get_statistics()
    logging.info(f"📊 Index statistics:")
    logging.info(f"   Total indexed file IDs: {index_stats['total_images']:,}")
    if index_stats['sample_mappings']:
        logging.info(f"   Sample mappings: {index_stats['sample_mappings']}")
    
    # Important note about the ID column
    logging.info(" NOTE: Using 'id' column as primary identifier")
    logging.info("   This column contains the actual image IDs that match tar files")
    logging.info("   Each metadata row describes the image with that row's ID")
     
    # Use signal handler for graceful interruption
    with SoftStopHandler() as stop_handler:
        
        # Stream filtered metadata
        logging.info(f"📖 Streaming filtered metadata from {meta_path}")
        metadata_stream = stream_filtered_metadata(meta_path, cfg, stop_handler)
        
        if cfg.dry_run:
            count = sum(1 for _ in metadata_stream)
            logging.info(f"🎯 Dry run: {count:,} images match criteria")
            return
        
        # Process extractions
        if cfg.extract_images:
            process_extractions_simple(metadata_stream, cfg, out_dir, tar_index, stop_handler)
        else:
            logging.info("📄 Filtering metadata only (extraction disabled)")
            count = sum(1 for _ in metadata_stream)
            logging.info(f"✅ Found {count:,} images matching criteria")
    
    logging.info("🎉 Script completed successfully!")

if __name__ == "__main__":
    main()
