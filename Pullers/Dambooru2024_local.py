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
from typing import Union
from collections import defaultdict, deque
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Set, Tuple, Callable
from functools import lru_cache
import io

# Additional utility for robust file URL parsing
from urllib.parse import urlparse

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
    file_path_col: Optional[str] = None  # Auto-detect from available columns
    id_col: str = "id"  # Primary ID for both finding files and metadata association
    md5_col: str = "md5"

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
    io_workers: int = 12  # I/O-bound JSON and file writers
    files_per_shard: int = 10000
    batch_size: int = 1000  # Number of metadata rows to read at once
    write_buffer_size_mb: int = 12 # Reasonable write buffer for high-throughput RAID 0
    progress_update_interval: int = 5000

    # Number of worker threads for the asynchronous JSON writer.  A value of zero
    # disables the async writer and falls back to BatchedJSONWriter.
    json_writer_workers: int = 4
    pre_create_shards: bool = True
    enable_fsync: bool = False  # Disable fsync for speed (filesystem will handle it)

    # JSON writer performance
    json_flush_interval: float = 5.0
    
    # Bound memory usage of not-found sampling
    not_found_max_sample: int = 20000

    # Memory management for large tar files
    use_tar_streaming: bool = False  # Use 'r|' mode for very large tars to reduce memory
    # Process mode: accumulate per-tar to avoid thrash
    tar_major: bool = False  # If True, batch by tar instead of switching constantly

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
            # Don't flush if we're closing; return early to allow graceful shutdown
            if self._closed:
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
                # Create a copy to avoid modification during iteration
                in_flight_copy: List[List[tuple[Path, Dict[str, Any]]]] = []
                with self.lock:
                    in_flight_copy = self.in_flight_writes.copy()
                    self.in_flight_writes.clear()

                # Iterate outside the lock to avoid deadlocks
                for to_write in in_flight_copy:
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
                if self.enable_fsync:
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
        found_ids = set()
        for shard_dir in sharding.base_dir.iterdir():
            if shard_dir.is_dir() and shard_dir.name.startswith("shard_"):
                for file_path in shard_dir.iterdir():
                    if file_path.is_file() and file_path.stem.isdigit():
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
            
            # Calculate rates
            if self.last_stats and elapsed > 0:
                time_delta = time.time() - self.last_report
                extracted_delta = stats['extracted'] - self.last_stats.get('extracted', 0)
                rate = extracted_delta / time_delta if time_delta > 0 else 0
            else:
                rate = stats['extracted'] / elapsed if elapsed > 0 else 0
            
            # Memory usage
            process = psutil.Process()
            mem_usage = process.memory_info().rss / 1024**3  # GB

            # Write throughput (only bytes, not mixed units)
            write_mb = stats.get('total_bytes_written', 0) / (1024 * 1024)
            write_time = stats.get('write_time', 0.1)  # Avoid divide by zero
            write_throughput = write_mb / write_time if write_time > 0 else 0            
            
            # Use consistent units (images/s for rate, MB/s for write)
            logging.info(
                f"📊 Progress: {stats['extracted']:,} extracted, "
                f"{stats['failed']:,} failed, {stats['skipped']:,} skipped, "
                f"{stats['not_found']:,} not found | "
                f"Rate: {rate:.1f} img/s | Memory: {mem_usage:.1f}GB | "
                f"Write: {write_throughput:.1f} MB/s | "
                f"Total written: {write_mb:.1f}MB | "
                f"Time: {elapsed/60:.1f}min"
            )
            
            self.last_report = time.time()
            self.last_stats = stats.copy()

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
        # Event signaled when a force exit (second Ctrl+C) is requested
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
            logging.warning("\n⚠️ Force exit requested. Attempting quick durability flush...")
            # Mark force exit event so loops can respond
            self._force_exit_event.set()
            # Best-effort short flush of pending writes/progress
            try:
                for fn in self._flush_hooks:
                    try:
                        fn()
                    except Exception:
                        pass
                time.sleep(0.5)
            finally:
                # Exit the process to abort any remaining work
                os._exit(1)
    
    def should_stop(self):
        """Return True if a graceful or force exit has been requested."""
        return self.stop_event.is_set() or self._force_exit_event.is_set()

    def should_force_exit(self):
        """Return True if a force exit (second Ctrl+C) has been requested."""
        return self._force_exit_event.is_set()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def prepare_json_data(row: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Prepare JSON metadata for a single image."""
    file_url = row.get(cfg.file_path_col, "") if cfg.file_path_col else ""
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

def detect_metadata_structure(path: Path, cfg: Config) -> None:
    """Auto-detect metadata structure and update config accordingly."""
    logging.info("🔍 Detecting metadata structure...")
    try:
        import pandas as pd
        pdf = pd.read_parquet(str(path), engine='pyarrow').head(5)
        # Check if ID is in the index
        if pdf.index.name == 'id' or (getattr(pdf.index, 'dtype', None) in ['int64', 'int32'] and 'id' not in pdf.columns):
            logging.info("📊 ID column is in the DataFrame index, will handle accordingly")
            cfg.id_in_index = True
        else:
            cfg.id_in_index = False
        # Detect file URL column
        file_cols = [col for col in pdf.columns if any(x in col.lower() for x in ['file', 'url', 'path', 'media'])]
        if file_cols:
            cfg.file_path_col = file_cols[0]
            logging.info(f"📁 Using file column: {cfg.file_path_col}")
        else:
            cfg.file_path_col = None
            logging.warning("⚠️ No file URL column found, will rely on ID-based lookup only")
    except Exception as e:
        logging.warning(f"⚠️ Could not detect structure with pandas: {e}")
        cfg.id_in_index = False
    # Show sample data for debugging
    try:
        lf = pl.scan_parquet(str(path))
        try:
            sample = lf.head(2).collect()
        except Exception as e:
            logging.warning(f"⚠️ Could not collect sample: {e}")
            # Try without streaming
            sample = pl.read_parquet(str(path), n_rows=2)
        logging.info(f"📋 Available columns: {sample.columns[:10]}...")
        logging.info(f"📋 Sample row:\n{sample.head(1)}")
    except Exception as e:
        logging.debug(f"Failed to collect sample metadata: {e}")

def stream_filtered_metadata(path: Path, cfg: Config, stop_handler: Optional[SoftStopHandler] = None) -> Iterator[Dict[str, Any]]:
    """
    Streams filtered metadata using collect(streaming=True) for constant memory.
    Yields individual rows as dictionaries.
    """
    # Build lazy frame
    lf = pl.scan_parquet(str(path))

    # Handle case where ID is in index (pandas-style parquet)
    if getattr(cfg, 'id_in_index', False):
        logging.info("🔄 Converting index-based ID to column...")
        import pandas as pd
        pdf = pd.read_parquet(str(path))
        # Reset index if needed so that ID becomes a column
        if pdf.index.name == 'id' or 'id' not in pdf.columns:
            pdf = pdf.reset_index()
            if 'index' in pdf.columns and 'id' not in pdf.columns:
                pdf = pdf.rename(columns={'index': 'id'})
        df_full = pl.from_pandas(pdf)
        lf = df_full.lazy()
        logging.info(f"📊 Loaded {len(df_full):,} total rows from pandas conversion")
    
    # Apply columns selection
    cols_to_load: set[str] = set()
    
    # Always load ID and file path for extraction
    cols_to_load.add(cfg.id_col)
    if cfg.file_path_col:
        cols_to_load.add(cfg.file_path_col)
    
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
    try:
        # Prefer using collect_schema for robust schema retrieval
        available_cols = set(lf.collect_schema().keys())
    except Exception:
        # Fallback: collect small sample to get schema
        available_cols = set(lf.head(1).collect().columns)
    final_cols = list(cols_to_load.intersection(available_cols))

    # md5 is optional
    if cfg.md5_col and cfg.md5_col in available_cols:
        final_cols.append(cfg.md5_col)
        logging.info(f"✅ MD5 column '{cfg.md5_col}' found and will be used")

    if final_cols:
        lf = lf.select(final_cols)
    
    # Apply transformations while still lazy
    numeric_cols = [c for c in (cfg.width_col, cfg.height_col, cfg.score_col) if c in final_cols]
    for col in numeric_cols:
        # Cast numeric columns to Int64 and fill nulls. Use -1 for score to distinguish from actual 0.
        lf = lf.with_columns(
            pl.col(col)
                .cast(pl.Int64, strict=False)
                .fill_null(-1 if col == cfg.score_col else 0)
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
        # Try different collection strategies
        df = None
        try:
            # First try: streaming collection
            df = lf.collect(streaming=True)
            logging.info("✅ Using streaming collection")
        except Exception as e1:
            logging.warning(f"⚠️ Streaming collection failed: {e1}")
            try:
                # Second try: regular collection
                logging.info("🔄 Trying regular collection...")
                df = lf.collect()
                logging.info("✅ Using regular collection")
            except Exception as e2:
                logging.warning(f"⚠️ Regular collection failed: {e2}")
                # Final fallback: read directly with chunking
                logging.info("🔄 Using direct parquet read with chunking...")
                df = pl.read_parquet(str(path))
                # Apply same column selection and filters
                if final_cols:
                    df = df.select(final_cols)
                filter_expr = build_polars_filter_expr(cfg)
                df = df.filter(filter_expr)
                logging.info("✅ Using direct read")

        total_rows = len(df)
        logging.info(f"📊 Filtered metadata contains {total_rows:,} matching images")
        if total_rows == 0:
            logging.warning("⚠️ No images match the filter criteria!")
            return

        batch_count = 0
        total_yielded = 0
        # Use iter_slices for memory efficiency
        slice_size = min(cfg.batch_size, 10000)
        for batch_df in df.iter_slices(slice_size):
            batch_count += 1
            # More frequent progress updates for large datasets
            if batch_count % 5 == 0 or total_yielded == 0:
                progress_pct = (total_yielded / total_rows * 100) if total_rows > 0 else 0
                logging.info(f"📊 Processing batch {batch_count} ({total_yielded:,}/{total_rows:,} items - {progress_pct:.1f}%)...")

            # Check for stop signal
            if stop_handler and stop_handler.should_stop():
                logging.info("🛑 Stopping stream due to user interrupt...")
                break

            for row in batch_df.iter_rows(named=True):
                total_yielded += 1
                yield row

        if total_yielded > 0:
            logging.info(f"✅ Streamed {total_yielded:,} filtered items")

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
        # Primary numeric-id index
        self.index: Dict[int, str] = {}       # {file_id: tar_name}
        self.index_paths: Dict[int, str] = {}  # {file_id: internal path within tar}
        self.lock = threading.Lock()
        
        # Use the existing cache that's already built
        cache_file = source_dir / ".tar_index_cache.json"
        
        if cache_file.exists() and not rebuild:
            self._load_existing_cache(cache_file)
        else:
            # Use info-level logging and a book emoji to indicate we are building a new index
            logging.info("📚 No existing cache found. Building new index...")
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

    def _extract_id_from_key(self, key: str) -> Optional[int]:
        """Extract numeric file ID from a manifest key."""
        if not isinstance(key, str):
            return None
        base = key.rsplit('/', 1)[-1]
        stem = base.split('.', 1)[0]
        try:
            return int(stem)
        except ValueError:
            return None

    def _parse_manifest(self, data: Any) -> List[str]:
        """Parse manifest data and return list of file paths."""
        paths: List[str] = []
        if isinstance(data, dict):
            # Most common: dict with numeric keys or 'files' key
            if 'files' in data:
                if isinstance(data['files'], dict):
                    paths.extend(data['files'].keys())
                elif isinstance(data['files'], list):
                    paths.extend(data['files'])
            else:
                # Try direct numeric keys (common in Danbooru dumps)
                for k in data.keys():
                    if k.isdigit() or '.' in k:
                        paths.append(k)
        elif isinstance(data, list):
            # Simple list of filenames
            paths.extend([str(x) for x in data if x])
        return paths
    
    def _build_from_json_files(self):
        """Build index from the JSON files next to tar files."""
        logging.info("📚 Building index from JSON files...")
        tar_files = sorted(self.source_dir.glob("*.tar"))
        if not tar_files:
            logging.warning(f"⚠️ No tar files found in {self.source_dir}")
            return

        total_mapped = 0
        sample_shown = False

        for tar_path in tar_files:
            tar_name = tar_path.name
            json_path = tar_path.with_suffix('.json')
            if not json_path.exists():
                logging.debug(f"No JSON for {tar_name}")
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Show structure of first JSON for debugging
                if not sample_shown:
                    logging.info(f"🔍 Sample JSON structure from {json_path.name}:")
                    if isinstance(data, dict):
                        logging.info(f"   Type: dict with {len(data)} keys")
                        sample_keys = list(data.keys())[:5]
                        logging.info(f"   Sample keys: {sample_keys}")
                    elif isinstance(data, list):
                        logging.info(f"   Type: list with {len(data)} items")
                        logging.info(f"   Sample items: {data[:5]}")
                    sample_shown = True

                # Parse paths from manifest
                paths = self._parse_manifest(data)

                for path in paths:
                    fid = self._extract_id_from_key(path)
                    if fid is not None:
                        self.index[fid] = tar_name
                        self.index_paths[fid] = path
                        total_mapped += 1
            except Exception as e:
                logging.error(f"Failed to process {json_path}: {e}")

        logging.info(f"✅ Built index with {len(self.index):,} file ID mappings from {len(tar_files)} tar files")

        # Show sample mappings for debugging
        if self.index:
            sample = list(self.index.items())[:5]
            logging.info("📋 Sample ID mappings:")
            for fid, tar in sample:
                path = self.index_paths.get(fid, f"{fid}.jpg")
                logging.info(f"   ID {fid} -> {tar} (path: {path})")
    
    def find_image(self, image_id: int, file_url: str = "", md5: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Find image by ID in tar files."""
        with self.lock:
            if image_id in self.index:
                tar_name = self.index[image_id]
                internal_path = self.index_paths.get(image_id, f"{image_id}.jpg")
                return (tar_name, internal_path)

        # Log miss for debugging
        if len(self.index) > 0:
            sample_ids = list(self.index.keys())[:3]
            logging.debug(f"❌ ID {image_id} not in index. Sample IDs: {sample_ids}")

        return None

    # cache_discovered_path removed; paths are stored directly in save_file

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics for debugging."""
        stats = {
            'total_images': len(self.index),
            'unique_tars': len(set(self.index.values())) if self.index else 0,
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
    p.add_argument("--tar-major", action="store_true", help="Batch by tar (accumulate and process per-tar to reduce open/close thrash)")

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

    # Toggle tar-major batching mode
    if hasattr(args, 'tar_major') and args.tar_major:
        cfg.tar_major = True

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
def process_pending_extractions(tar_handle: tarfile.TarFile,
                               members: Dict[str, tarfile.TarInfo],
                               pending: List[Tuple[int, str, Dict[str, Any]]],
                               cfg: Config,
                               tar_name: str,
                               sharding: DirectorySharding,
                               progress_tracker: ValidatingProgressTracker,
                               json_writer: Optional[Union[BatchedJSONWriter, AsyncJSONWriter]],
                               tar_index,
                               stats: Dict[str, Any],
                               stats_lock: threading.Lock) -> None:
    """Process a batch of pending extractions from the current open tar."""
    if not pending:
        return

    if not tar_handle:
        logging.error(f"❌ Cannot process {len(pending)} files - tar handle is None")
        return

    logging.debug(f"📦 Processing {len(pending)} files from {tar_name}")

    # Use thread pool for I/O operations
    with ThreadPoolExecutor(max_workers=cfg.io_workers) as executor:
        # List to keep track of pending futures
        futures = []

        for image_id, filename, row in pending:
            # Locate member
            member = members.get(filename)
            if member is None:
                # Fallback to basename match
                basename = filename.rsplit('/', 1)[-1]

                # If members dict is empty (streaming mode), try to find the member directly
                if not members and hasattr(tar_handle, 'getmember'):
                    try:
                        member = tar_handle.getmember(filename)
                    except KeyError:
                        try:
                            member = tar_handle.getmember(basename)
                        except KeyError:
                            pass

                # Otherwise search through members dict
                if member is None and members:
                    for m_name, m in members.items():
                        if m_name.rsplit('/', 1)[-1] == basename:
                            member = m
                            break

            if member is None:
                with stats_lock:
                    stats['not_found'] += 1
                    stats['not_found_ids'].append(image_id)
                continue

            # Extract file data
            try:
                file_obj = tar_handle.extractfile(member)
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

        # Wait for all writes to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Write task failed: {e}")
                with stats_lock:
                    stats['failed'] += 1


def process_extractions_streaming(metadata_stream: Iterator[Dict[str, Any]],
                                 cfg: Config, dest_dir: Path,
                                 tar_index,
                                 stop_handler: Optional[Any] = None) -> None:
    """Process extractions with true streaming - no full collection phase."""

    # Initialize components
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)

    # Initialize JSON writer
    json_writer: Optional[Union[BatchedJSONWriter, AsyncJSONWriter]] = None
    if cfg.per_image_json:
        if cfg.json_writer_workers and cfg.json_writer_workers > 0:
            json_writer = AsyncJSONWriter(workers=cfg.json_writer_workers, enable_fsync=cfg.enable_fsync)
        else:
            json_writer = BatchedJSONWriter(flush_interval=cfg.json_flush_interval, batch_size=1000, enable_fsync=cfg.enable_fsync)

    # Progress tracker
    progress_tracker = ValidatingProgressTracker(dest_dir / "progress.json", update_interval=cfg.progress_update_interval, enable_fsync=cfg.enable_fsync)

    # Validation
    if cfg.validate_on_start:
        if cfg.full_scan:
            progress_tracker.full_filesystem_scan(sharding)
        else:
            progress_tracker.validate_files(sharding, auto_clean=True)

    # Statistics
    stats: Dict[str, Any] = {
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

    # Show starting stats
    initial_stats = progress_tracker.get_statistics()
    logging.info(f"📊 Starting with {initial_stats['completed']:,} completed, {initial_stats['missing']:,} missing")
    logging.info(f"🚀 Using streaming extraction with {cfg.workers} workers, {cfg.io_workers} I/O workers")
    logging.info(f"💾 RAID optimized: {cfg.write_buffer_size_mb}MB write buffers")

    # Streaming state
    current_tar: Optional[str] = None
    current_tar_handle: Optional[tarfile.TarFile] = None
    current_members: Dict[str, tarfile.TarInfo] = {}
    pending_extractions: List[Tuple[int, str, Dict[str, Any]]] = []
    total_processed = 0
    BATCH_SIZE = 100  # Target batch size, but will process smaller batches when needed

    try:
        logging.info("🔄 Starting streaming extraction...")

        for row in metadata_stream:
            if stop_handler and stop_handler.should_stop():
                logging.info("🛑 Stopping extraction due to user interrupt...")
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
            file_url = row.get(cfg.file_path_col, "") if cfg.file_path_col else ""
            tar_info = tar_index.find_image(image_id, file_url, row.get(cfg.md5_col))
            if not tar_info:
                with stats_lock:
                    stats['not_found'] += 1
                    stats['not_found_ids'].append(image_id)
                continue

            tar_name, filename = tar_info

            # Check if we need to switch tar files
            if tar_name != current_tar:
                # ALWAYS process any pending extractions from previous tar
                if pending_extractions:
                    logging.debug(f"🔄 Processing {len(pending_extractions)} pending files before switching tar")
                    if current_tar:
                        if cfg.use_tar_streaming:
                            # Stream the previous tar before switching
                            tar_path_prev = Path(cfg.source_images_dir) / current_tar
                            logging.info(f"📂 Opening {current_tar}...")
                            process_tar_streaming(
                                cfg, tar_path_prev, 'r|', pending_extractions,
                                sharding, progress_tracker, json_writer,
                                tar_index, stats, stats_lock
                            )
                            # Log memory usage after closing stream
                            mem_usage = psutil.Process().memory_info().rss / 1024**3
                            logging.info(f"📁 Closed {current_tar} | Memory: {mem_usage:.1f}GB")
                        else:
                            # Random-access: use existing handle if available
                            if current_tar_handle:
                                process_pending_extractions(
                                    current_tar_handle, current_members, pending_extractions,
                                    cfg, current_tar, sharding, progress_tracker, json_writer,
                                    tar_index, stats, stats_lock
                                )
                    pending_extractions = []

                # Close previous tar handle if open
                if current_tar_handle:
                    current_tar_handle.close()
                    mem_usage = psutil.Process().memory_info().rss / 1024**3
                    logging.info(f"📁 Closed {current_tar} | Memory: {mem_usage:.1f}GB")

                # Switch to new tar
                current_tar = tar_name
                current_members = {}
                current_tar_handle = None
                if not cfg.use_tar_streaming:
                    # Only open tar in random-access mode
                    tar_path_new = Path(cfg.source_images_dir) / tar_name
                    logging.info(f"📂 Opening {tar_name}...")
                    try:
                        current_tar_handle = tarfile.open(tar_path_new, 'r')
                        current_members = {m.name: m for m in current_tar_handle}
                    except Exception as e:
                        logging.error(f"Failed to open tar {tar_name}: {e}")
                        with stats_lock:
                            stats['failed'] += 1
                        current_tar = None
                        continue

            # Add to pending extractions
            pending_extractions.append((image_id, filename, row))
            total_processed += 1

            # Process batch if it's large enough, stop requested, or we're at the end of current tar's files
            # Note: We should process small batches (even 1-2 files) rather than leave them unprocessed
            should_process = (
                len(pending_extractions) >= BATCH_SIZE or
                (stop_handler and stop_handler.should_stop())
            )

            if should_process and current_tar:
                if cfg.use_tar_streaming:
                    # Use true sequential scan for streaming tars (no random access)
                    tar_path_cur = Path(cfg.source_images_dir) / current_tar
                    logging.info(f"📂 Opening {current_tar}...")
                    process_tar_streaming(
                        cfg, tar_path_cur, 'r|', pending_extractions,
                        sharding, progress_tracker, json_writer,
                        tar_index, stats, stats_lock
                    )
                    # Close & log memory after streaming pass
                    mem_usage = psutil.Process().memory_info().rss / 1024**3
                    logging.info(f"📁 Closed {current_tar} | Memory: {mem_usage:.1f}GB")
                    pending_extractions = []
                else:
                    # Random-access path (existing behavior)
                    if current_tar_handle:
                        process_pending_extractions(
                            current_tar_handle, current_members, pending_extractions,
                            cfg, current_tar, sharding, progress_tracker, json_writer,
                            tar_index, stats, stats_lock
                        )
                        pending_extractions = []

                # Report progress
                if progress_reporter.should_report():
                    progress_reporter.report(stats)

            # Periodic status update
            if total_processed % 10000 == 0:
                logging.info(f"📊 Processed {total_processed:,} metadata entries...")

        # CRITICAL: Process any remaining extractions, even if batch is small
        # This ensures we don't skip files that didn't meet the batch threshold
        if pending_extractions:
            logging.info(f"📦 Processing final batch of {len(pending_extractions)} files...")
            if current_tar:
                if cfg.use_tar_streaming:
                    # Streaming mode: open tar and process sequentially
                    tar_path_final = Path(cfg.source_images_dir) / current_tar
                    logging.info(f"📂 Opening {current_tar}...")
                    process_tar_streaming(
                        cfg, tar_path_final, 'r|', pending_extractions,
                        sharding, progress_tracker, json_writer,
                        tar_index, stats, stats_lock
                    )
                    mem_usage = psutil.Process().memory_info().rss / 1024**3
                    logging.info(f"📁 Closed {current_tar} | Memory: {mem_usage:.1f}GB")
                else:
                    # Random-access fallback
                    if not current_tar_handle and current_tar:
                        try:
                            tar_path_final = Path(cfg.source_images_dir) / current_tar
                            if tar_path_final.exists():
                                logging.warning(f"⚠️ Reopening {current_tar} for final batch processing")
                                current_tar_handle = tarfile.open(tar_path_final, 'r')
                                current_members = {m.name: m for m in current_tar_handle}
                        except Exception as e:
                            logging.error(f"❌ Failed to reopen tar for final batch: {e}")
                    if current_tar_handle:
                        process_pending_extractions(
                            current_tar_handle, current_members, pending_extractions,
                            cfg, current_tar, sharding, progress_tracker, json_writer,
                            tar_index, stats, stats_lock
                        )
                    else:
                        logging.error(f"❌ Unable to process final {len(pending_extractions)} files - tar handle unavailable")
                        with stats_lock:
                            stats['failed'] += len(pending_extractions)
            else:
                # No current tar available
                logging.error(f"❌ Unable to process final {len(pending_extractions)} files - no current tar")
                with stats_lock:
                    stats['failed'] += len(pending_extractions)

        # Close final tar
        if current_tar_handle:
            current_tar_handle.close()

        # Final report
        progress_reporter.report(stats, force=True)

        # Persist not-found sample
        try:
            not_found_sample = list(stats.get('not_found_ids', []))
            if not_found_sample:
                with open(dest_dir / "not_found_ids.sample.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "count": stats.get('not_found', 0),
                        "sample_size": len(not_found_sample),
                        "ids": not_found_sample
                    }, f, indent=2)
        except Exception as e:
            logging.warning(f"⚠️ Failed to persist not-found sample: {e}")

        logging.info(
            f"✅ Extraction complete: {stats['extracted']:,} extracted, "
            f"{stats['failed']:,} failed, {stats['skipped']:,} skipped, "
            f"{stats['not_found']:,} not found"
        )

    finally:
        # Cleanup
        if current_tar_handle:
            current_tar_handle.close()
        if json_writer:
            json_writer.close()
        progress_tracker.save_final()
        # Log if we had any unprocessed files
        if pending_extractions:
            logging.error(f"⚠️ WARNING: {len(pending_extractions)} files were not processed!")


def process_extractions_tar_major(metadata_stream: Iterator[Dict[str, Any]],
                                  cfg: Config, dest_dir: Path,
                                  tar_index,
                                  stop_handler: Optional[Any] = None) -> None:
    """Accumulate rows per tar and process each tar in chunky batches to avoid open/close thrash."""
    # Initialize sharding
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)

    # Initialize JSON writer if per-image JSON output is enabled
    json_writer: Optional[Union[BatchedJSONWriter, AsyncJSONWriter]] = None
    if cfg.per_image_json:
        if cfg.json_writer_workers and cfg.json_writer_workers > 0:
            json_writer = AsyncJSONWriter(workers=cfg.json_writer_workers, enable_fsync=cfg.enable_fsync)
        else:
            json_writer = BatchedJSONWriter(flush_interval=cfg.json_flush_interval, batch_size=1000, enable_fsync=cfg.enable_fsync)

    # Create progress tracker for per-image completion tracking
    progress_tracker = ValidatingProgressTracker(dest_dir / "progress.json",
                                                 update_interval=cfg.progress_update_interval,
                                                 enable_fsync=cfg.enable_fsync)

    # Optional validation at start, to clean up or scan existing files
    if cfg.validate_on_start:
        if cfg.full_scan:
            progress_tracker.full_filesystem_scan(sharding)
        else:
            progress_tracker.validate_files(sharding, auto_clean=True)

    # Ensure safe shutdown flush: register hooks on stop_handler
    if stop_handler:
        if json_writer:
            stop_handler.add_flush_hook(lambda: json_writer.close())
        stop_handler.add_flush_hook(lambda: progress_tracker.save_final())

    # Stats for reporting extraction outcomes
    stats: Dict[str, Any] = {
        'extracted': 0, 'failed': 0, 'skipped': 0, 'not_found': 0,
        'not_found_ids': deque(maxlen=cfg.not_found_max_sample),
        'total_bytes_written': 0, 'write_time': 0.0
    }
    stats_lock = threading.Lock()
    reporter = ProgressReporter(interval=10.0)

    logging.info("🔄 Starting TAR-major extraction…")

    # Accumulate rows per tar; each key is tar_name mapping to list of (image_id, filename, metadata)
    buckets: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = defaultdict(list)
    # Determine threshold for flushing a tar bucket; ensure at least 50 to reduce thrashing
    target_batch = max(50, cfg.batch_size)

    # Helper to process a single tar once enough rows accumulate
    def flush_tar(tar_name: str) -> None:
        items = buckets.get(tar_name, [])
        if not items:
            return
        # Process the tar via existing handler; this writes files and updates stats
        process_single_tar(cfg, tar_name, items, sharding, progress_tracker, json_writer,
                           tar_index, stats, stats_lock)
        # Clear the bucket for this tar
        buckets[tar_name].clear()
        # Periodically report progress
        if reporter.should_report():
            reporter.report(stats)

    # Iterate through metadata rows, bucketizing by tar
    for row in metadata_stream:
        # Allow for graceful shutdown via stop_handler
        if stop_handler and stop_handler.should_stop():
            logging.info("🛑 Stopping due to interrupt…")
            break

        image_id = row[cfg.id_col]

        # Skip if already completed or exists in destination
        if progress_tracker.is_completed(image_id):
            with stats_lock:
                stats['skipped'] += 1
            continue
        exists, _ = sharding.file_exists(image_id)
        if exists:
            progress_tracker.mark_completed(image_id)
            with stats_lock:
                stats['skipped'] += 1
            continue

        # Determine which tar and filename contain this image
        file_url = row.get(cfg.file_path_col, "") if cfg.file_path_col else ""
        tar_info = tar_index.find_image(image_id, file_url, row.get(cfg.md5_col))
        if not tar_info:
            with stats_lock:
                stats['not_found'] += 1
                stats['not_found_ids'].append(image_id)
            continue

        tar_name, filename = tar_info
        buckets[tar_name].append((image_id, filename, row))

        # If this tar bucket has reached the threshold, flush it
        if len(buckets[tar_name]) >= target_batch:
            flush_tar(tar_name)

    # Final flush for any remaining buckets once streaming is finished
    for tar_name, items in list(buckets.items()):
        if items:
            logging.info(f"📦 Finalizing {tar_name} with {len(items)} files…")
            flush_tar(tar_name)

    # Final reporting and cleanup
    reporter.report(stats, force=True)
    if json_writer:
        json_writer.close()
    progress_tracker.save_final()
def process_extractions_simple(metadata_stream: Iterator[Dict[str, Any]],
                              cfg: Config, dest_dir: Path,
                              tar_index,
                              stop_handler: Optional[Any] = None) -> None:
    """Process extractions - now uses streaming mode for better memory efficiency."""
    # Delegate to the streaming implementation for improved memory usage
    logging.info("🔄 Using streaming extraction mode for better memory efficiency...")
    process_extractions_streaming(metadata_stream, cfg, dest_dir, tar_index, stop_handler)

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
              json_writer: Optional[Union[BatchedJSONWriter, AsyncJSONWriter]], tar_index, tar_name: str, stats: Dict[str, Any], stats_lock: threading.Lock) -> None:
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
        with tar_index.lock:
            tar_index.index_paths[image_id] = member.name
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

    # Check Polars version and settings
    try:
        import polars as pl
        pl_version = pl.__version__
        logging.info(f"📦 Using Polars version: {pl_version}")
        # Set Polars to use all available cores
        import os
        os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())
    except Exception as e:
        logging.warning(f"⚠️ Could not check Polars version: {e}")

    # Auto-detect metadata structure
    detect_metadata_structure(meta_path, cfg)
    
    # Build tar index
    logging.info("🗂️ Initializing tar index...")
    rebuild_index = cfg.rebuild_tar_index
    tar_index = TarIndex(source_dir, rebuild=rebuild_index)

    # Debug: Show index statistics
    index_stats = tar_index.get_statistics()
    logging.info(f"📊 Index statistics:")
    logging.info(f"   Total indexed file IDs: {index_stats['total_images']:,}")
    logging.info(f"   Unique tar files: {index_stats.get('unique_tars', 0)}")
    if index_stats.get('sample_mappings'):
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
            # Use TAR-major mode when configured; otherwise fallback to simple streaming extraction
            if getattr(cfg, "tar_major", False):
                process_extractions_tar_major(metadata_stream, cfg, out_dir, tar_index, stop_handler)
            else:
                process_extractions_simple(metadata_stream, cfg, out_dir, tar_index, stop_handler)
        else:
            logging.info("📄 Filtering metadata only (extraction disabled)")
            count = sum(1 for _ in metadata_stream)
            logging.info(f"✅ Found {count:,} images matching criteria")
    
    logging.info("🎉 Script completed successfully!")

if __name__ == "__main__":
    main()
