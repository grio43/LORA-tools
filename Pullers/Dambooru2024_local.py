#!/usr/bin/env python3
"""
RAID-Optimized Danbooru dataset filtering and extraction script.
Fixed version addressing tar file ordering and memory efficiency issues.
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
from typing import Union
from collections import defaultdict, deque
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Set, Tuple, Callable

import polars as pl
import psutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Holds all runtime parameters optimized for RAID."""

    # ---- Paths ------------------------------------------------------------
    metadata_db_path: str = "/media/andrewk/qnap-public/danbooru2024/metadata.parquet"
    source_images_dir: str = "/media/andrewk/qnap-public/danbooru2024/images/"
    output_dir: str = "/mnt/raid0/Danbroo/"

    # ---- Column names -----------------------------------------------------
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

    # ---- Filtering --------------------------------------------------------
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
    workers: int = 12  # Reduced for better memory management
    io_workers: int = 8  # I/O-bound JSON and file writers
    files_per_shard: int = 10000
    batch_size: int = 1000  # Number of metadata rows to read at once
    write_buffer_size_mb: int = 8  # Reduced buffer size
    progress_update_interval: int = 5000

    json_writer_workers: int = 4
    pre_create_shards: bool = True
    enable_fsync: bool = False

    json_flush_interval: float = 5.0
    not_found_max_sample: int = 20000
    use_tar_streaming: bool = True  # Enable true streaming

# ---------------------------------------------------------------------------
# Directory Sharding
# ---------------------------------------------------------------------------
class DirectorySharding:
    """Manages single-level directory sharding for O(1) lookups."""
    
    def __init__(self, base_dir: Path, files_per_dir: int = 5000):
        self.base_dir = base_dir
        self.files_per_dir = files_per_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_shard_path(self, image_id: int, create: bool = False) -> Path:
        """Get the shard directory path for a given image ID."""
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
        """Check if file exists in the specific shard directory."""
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
        self.enable_fsync = enable_fsync
        self._closed = False
        self._flush_thread = None
        self._start_flush_thread()
    
    def _start_flush_thread(self):
        """Start background thread for periodic flushing."""
        def flush_periodically():
            while not self._closed:
                time.sleep(self.flush_interval)
                if time.time() - self.last_flush >= self.flush_interval:
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
        """Flush buffer to disk."""
        to_write: List[tuple[Path, Dict[str, Any]]] = []
        with self.lock:
            if not self.buffer or self._closed:
                return
            to_write = self.buffer.copy()
            self.buffer.clear()
            self.last_flush = time.time()
        
        logging.debug(f"💾 Flushing {len(to_write)} JSON writes...")
        
        for path, data in to_write:
            self._atomic_write_json(path, data)
    
    def _atomic_write_json(self, path: Path, data: Dict[str, Any]):
        """Write JSON file atomically."""
        tmp_path = path.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                if self.enable_fsync:
                    os.fsync(f.fileno())
            tmp_path.replace(path)
        except Exception as e:
            logging.error(f"❌ Failed to write {path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    def close(self):
        """Close writer and flush remaining data."""
        with self.lock:
            self._closed = True
        self._flush_buffer()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10.0)

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

        for i in range(self.workers):
            t = threading.Thread(target=self._worker, name=f"json_writer_{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def add_write(self, path: Path, data: Dict[str, Any]):
        """Enqueue a JSON write operation."""
        if self._closed:
            return
        self.queue.put((path, data))

    def _worker(self):
        """Worker thread loop that processes queued JSON writes."""
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break
            path, data = task
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
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
        self.queue.join()
        for _ in range(self.workers):
            self.queue.put(None)
        self.queue.join()
        for t in self._threads:
            t.join(timeout=5)

# ---------------------------------------------------------------------------
# Progress Tracker
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
        self.enable_fsync = enable_fsync
        self._load_progress()
    
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
            
            if self.last_stats and elapsed > 0:
                time_delta = time.time() - self.last_report
                extracted_delta = stats['extracted'] - self.last_stats.get('extracted', 0)
                rate = extracted_delta / time_delta if time_delta > 0 else 0
            else:
                rate = stats['extracted'] / elapsed if elapsed > 0 else 0
            
            process = psutil.Process()
            mem_usage = process.memory_info().rss / 1024**3

            write_mb = stats.get('total_bytes_written', 0) / (1024 * 1024)
            write_time = stats.get('write_time', 0.1)
            write_throughput = write_mb / write_time if write_time > 0 else 0
            
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
        self._force_exit_event = threading.Event()
        
    def __enter__(self):
        self.original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_sigint)
        
    def add_flush_hook(self, fn: Callable[[], None]) -> None:
        """Register a hook to flush before hard-exit."""
        self._flush_hooks.append(fn)
        
    def _signal_handler(self, signum, frame):
        self._signal_count += 1
        
        if self._signal_count == 1:
            logging.warning("\n🛑 Graceful stop requested. Finishing current batch...")
            logging.warning("Press Ctrl+C again to force exit.")
            self.stop_event.set()
        else:
            logging.warning("\n⚠️ Force exit requested. Attempting quick durability flush...")
            self._force_exit_event.set()
            try:
                for fn in self._flush_hooks:
                    try:
                        fn()
                    except Exception:
                        pass
                time.sleep(0.5)
            finally:
                os._exit(1)
    
    def should_stop(self):
        """Return True if a graceful or force exit has been requested."""
        return self.stop_event.is_set() or self._force_exit_event.is_set()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def prepare_json_data(row: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Prepare JSON metadata for a single image."""
    file_url = row.get(cfg.file_path_col, "") if cfg.file_path_col else ""
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
        return escaped_tag
    elif ends_with_star:
        return r"\b" + escaped_tag
    elif starts_with_star:
        return escaped_tag + r"\b"
    else:
        return r"\b" + escaped_tag + r"\b"

def build_polars_filter_expr(cfg: Config) -> pl.Expr:
    """Build unified Polars filter expression for lazy evaluation."""
    filters = []
    
    # File type filtering
    if cfg.file_path_col:
        file_col = pl.col(cfg.file_path_col).fill_null("")
        
        if cfg.exclude_gifs:
            filters.append(~file_col.str.ends_with('.gif'))
    
        excluded_extensions = ('.zip', '.mp4', '.webm', '.swf')
        for ext in excluded_extensions:
            filters.append(~file_col.str.ends_with(ext))
    
    # Tag filtering
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
                filters.append(pl.any_horizontal(*char_filters))
        
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
                filters.append(pl.any_horizontal(*copy_filters))
        
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
                filters.append(pl.any_horizontal(*artist_filters))
        
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
            filters.append(pl.any_horizontal(*rating_filters))
    
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
    
    if not filters:
        return pl.lit(True)
    return pl.all_horizontal(*filters)

def detect_metadata_structure(path: Path, cfg: Config) -> None:
    """Auto-detect metadata structure and update config accordingly."""
    logging.info("🔍 Detecting metadata structure...")
    try:
        import pandas as pd
        pdf = pd.read_parquet(str(path), engine='pyarrow')
        # First row only for structure detection
        pdf = pdf.head(5)
        
        # Detect file URL column
        priority_patterns = [
            ('file_url', lambda c: c.lower() == 'file_url'),
            ('media_asset', lambda c: 'media_asset' in c.lower()),
            ('file_path', lambda c: 'file_path' in c.lower()),
            ('url', lambda c: 'url' in c.lower() and 'file' in c.lower()),
            ('path', lambda c: 'path' in c.lower() and 'file' in c.lower()),
            ('url_only', lambda c: 'url' in c.lower()),
        ]

        file_col_found = None
        for pattern_name, pattern_func in priority_patterns:
            matching_cols = [col for col in pdf.columns if pattern_func(col)]
            if matching_cols:
                valid_cols = [c for c in matching_cols if not any(skip in c.lower() for skip in ['ext', 'extension', 'size', 'count', 'type'])]
                if valid_cols:
                    file_col_found = valid_cols[0]
                    break

        if file_col_found:
            cfg.file_path_col = file_col_found
            logging.info(f"📄 Using file column: {cfg.file_path_col}")
        else:
            cfg.file_path_col = None
            logging.warning("⚠️ No file URL column found, will rely on ID-based lookup only")
    except Exception as e:
        logging.warning(f"⚠️ Could not detect structure with pandas: {e}")
    
    # Show sample data
    try:
        lf = pl.scan_parquet(str(path))
        try:
            schema_dict = lf.collect_schema()
        except AttributeError:
            schema_dict = lf.schema
        logging.info(f"📋 Available columns: {list(schema_dict.keys())[:10]}...")
        
        try:
            sample = pl.read_parquet(str(path), n_rows=1)
            logging.info(f"📋 Sample row:\n{sample}")
        except Exception as e:
            logging.debug(f"Couldn't read sample row: {e}")
    except Exception as e:
        logging.debug(f"Failed to inspect metadata: {e}")

def collect_and_group_metadata(path: Path, cfg: Config, tar_index, stop_handler: Optional[SoftStopHandler] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect all filtered metadata and group by tar file.
    Returns dict mapping tar_name -> list of metadata rows.
    """
    logging.info("📋 Collecting and grouping metadata by tar file...")
    
    # Build lazy frame
    lf = pl.scan_parquet(str(path))
    
    # Build columns to load
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
        try:
            available_cols = set(lf.collect_schema().keys())
        except AttributeError:
            available_cols = set(lf.schema.keys())
    except Exception as e:
        logging.debug(f"Could not get schema directly: {e}")
        sample = pl.read_parquet(str(path), n_rows=1)
        available_cols = set(sample.columns)
    
    # Filter columns to only those available
    final_cols = [col for col in cols_to_load if col in available_cols]
    
    # Add optional md5 column if available
    if cfg.md5_col and cfg.md5_col in available_cols:
        final_cols.append(cfg.md5_col)
        logging.info(f"✅ MD5 column '{cfg.md5_col}' found and will be used")
    
    if final_cols:
        lf = lf.select(final_cols)
    
    # Prepare transformations
    transformations = []
    
    # Numeric column transformations
    numeric_cols = [c for c in (cfg.width_col, cfg.height_col, cfg.score_col) if c in final_cols]
    for col in numeric_cols:
        transformations.append(
            pl.when(pl.col(col).is_null())
            .then(-1 if col == cfg.score_col else 0)
            .otherwise(pl.col(col).cast(pl.Int64, strict=False))
            .alias(col)
        )
    
    # Rating column transformation
    if cfg.rating_col in final_cols:
        transformations.append(
            pl.when(pl.col(cfg.rating_col).is_null())
            .then(pl.lit(""))
            .otherwise(pl.col(cfg.rating_col).cast(pl.String).str.to_lowercase())
            .alias(cfg.rating_col)
        )
    
    # Tag column transformations
    tag_cols_to_normalize = []
    
    if cfg.enable_include_tags or cfg.enable_exclude_tags:
        tag_cols_to_normalize.append(cfg.tags_col)
    if cfg.enable_character_filtering:
        tag_cols_to_normalize.append(cfg.character_tags_col)
    if cfg.enable_copyright_filtering:
        tag_cols_to_normalize.append(cfg.copyright_tags_col)
    if cfg.enable_artist_filtering:
        tag_cols_to_normalize.append(cfg.artist_tags_col)
    
    if cfg.per_image_json:
        for col in [cfg.tags_col, cfg.character_tags_col, cfg.copyright_tags_col, cfg.artist_tags_col]:
            if col not in tag_cols_to_normalize and col in final_cols:
                tag_cols_to_normalize.append(col)
    
    for col in tag_cols_to_normalize:
        if col in final_cols:
            transformations.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(""))
                .otherwise(pl.col(col).cast(pl.String).str.to_lowercase())
                .alias(col)
            )
    
    # Apply transformations
    if transformations:
        lf = lf.with_columns(transformations)
    
    # Apply filters
    filter_expr = build_polars_filter_expr(cfg)
    lf = lf.filter(filter_expr)
    
    try:
        logging.info(f"🔥 Starting metadata collection...")
        
        # Collect the filtered data
        df = None
        try:
            df = lf.collect()
            logging.info("✅ Using regular collection")
        except Exception as e:
            logging.warning(f"⚠️ Regular collection failed: {e}")
            # Fallback to direct read
            try:
                logging.info("📄 Using direct parquet read as fallback...")
                df = pl.read_parquet(str(path))
                if final_cols:
                    df = df.select(final_cols)
                if transformations:
                    df = df.with_columns(transformations)
                df = df.filter(filter_expr)
                logging.info("✅ Using direct read")
            except Exception as e2:
                logging.error(f"❌ All collection methods failed: {e2}")
                raise
        
        if df is None or len(df) == 0:
            logging.warning("⚠️ No images match the filter criteria!")
            return {}
        
        total_rows = len(df)
        logging.info(f"📊 Filtered metadata contains {total_rows:,} matching images")
        
        # Group by tar file
        grouped_metadata = defaultdict(list)
        not_found_count = 0
        
        logging.info("🗂️ Grouping metadata by tar file...")
        for row in df.iter_rows(named=True):
            if stop_handler and stop_handler.should_stop():
                logging.info("🛑 Stopping metadata collection due to user interrupt...")
                break
                
            image_id = row[cfg.id_col]
            file_url = row.get(cfg.file_path_col, "") if cfg.file_path_col else ""
            tar_info = tar_index.find_image(image_id, file_url, row.get(cfg.md5_col))
            
            if not tar_info:
                not_found_count += 1
                continue
                
            tar_name, filename = tar_info
            row['_tar_filename'] = filename  # Store the internal tar filename
            grouped_metadata[tar_name].append(row)
        
        logging.info(f"📊 Grouped {total_rows - not_found_count:,} images into {len(grouped_metadata)} tar files")
        if not_found_count > 0:
            logging.warning(f"⚠️ {not_found_count:,} images not found in tar index")
        
        # Sort tar files for efficient processing (0000.tar, 0001.tar, etc.)
        sorted_grouped = {}
        for tar_name in sorted(grouped_metadata.keys()):
            file_count = len(grouped_metadata[tar_name])
            logging.info(f"   {tar_name}: {file_count:,} files")
            sorted_grouped[tar_name] = grouped_metadata[tar_name]
        
        return sorted_grouped
        
    except Exception as e:
        logging.error(f"❌ Error during metadata collection: {e}")
        raise

def process_tar_file_streaming(tar_name: str, 
                             metadata_list: List[Dict[str, Any]],
                             cfg: Config, 
                             source_dir: Path,
                             sharding: DirectorySharding,
                             progress_tracker: ValidatingProgressTracker,
                             json_writer: Optional[Union[BatchedJSONWriter, AsyncJSONWriter]],
                             stats: Dict[str, Any],
                             stats_lock: threading.Lock,
                             stop_handler: Optional[SoftStopHandler] = None) -> None:
    """
    Process a single tar file using true streaming extraction.
    This opens the tar once and extracts all needed files efficiently.
    """
    tar_path = source_dir / tar_name
    if not tar_path.exists():
        logging.error(f"❌ Tar file not found: {tar_path}")
        with stats_lock:
            stats['failed'] += len(metadata_list)
        return
    
    logging.info(f"📦 Processing {tar_name} ({len(metadata_list):,} files)...")
    
    # Build lookup map: filename -> metadata row
    filename_to_metadata = {}
    id_to_metadata = {}
    
    for row in metadata_list:
        image_id = row[cfg.id_col]
        filename = row.get('_tar_filename', f"{image_id}.jpg")
        filename_to_metadata[filename] = row
        id_to_metadata[image_id] = row
    
    processed_count = 0
    extracted_count = 0
    skipped_count = 0
    failed_count = 0
    
    try:
        # Open tar file for streaming extraction
        with tarfile.open(tar_path, 'r') as tar:
            # Get all members once (this is lightweight - just metadata)
            members = {member.name: member for member in tar.getmembers() if member.isfile()}
            
            logging.debug(f"   Tar contains {len(members):,} total files")
            
            # Process each metadata row
            for row in metadata_list:
                if stop_handler and stop_handler.should_stop():
                    logging.info("🛑 Stopping tar processing due to user interrupt...")
                    break
                
                image_id = row[cfg.id_col]
                
                # Skip if already completed
                if progress_tracker.is_completed(image_id):
                    skipped_count += 1
                    processed_count += 1
                    continue
                
                # Check if file already exists on disk
                file_exists, _ = sharding.file_exists(image_id)
                if file_exists:
                    progress_tracker.mark_completed(image_id)
                    skipped_count += 1
                    processed_count += 1
                    continue
                
                # Find the file in the tar
                filename = row.get('_tar_filename', f"{image_id}.jpg")
                member = None
                
                # Strategy 1: Direct filename match
                if filename in members:
                    member = members[filename]
                
                # Strategy 2: Try different extensions
                if member is None:
                    basename = os.path.splitext(filename)[0]
                    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                        test_name = f"{basename}{ext}"
                        if test_name in members:
                            member = members[test_name]
                            break
                
                # Strategy 3: Try just the image ID with extensions
                if member is None:
                    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                        test_name = f"{image_id}{ext}"
                        if test_name in members:
                            member = members[test_name]
                            break
                
                # Strategy 4: Search by image ID in filename
                if member is None:
                    id_str = str(image_id)
                    for member_name, member_obj in members.items():
                        if id_str in member_name:
                            member = member_obj
                            break
                
                if member is None:
                    # File not found in tar
                    failed_count += 1
                    processed_count += 1
                    continue
                
                # Extract and save the file
                try:
                    # True streaming extraction - only extract this one file
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        data = file_obj.read()
                        file_obj.close()
                        
                        # Save file to disk
                        start_write = time.time()
                        ext = os.path.splitext(member.name)[1] or '.jpg'
                        shard_path = sharding.get_shard_path(image_id, create=True)
                        final_path = shard_path / f"{image_id}{ext}"
                        temp_path = final_path.with_suffix(final_path.suffix + '.tmp')
                        
                        # Write with buffer
                        buffer_mb = min(max(1, cfg.write_buffer_size_mb), 16)
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
                        
                        # Write JSON metadata if enabled
                        if cfg.per_image_json and json_writer:
                            json_path = shard_path / f"{image_id}.json"
                            json_data = prepare_json_data(row, cfg)
                            json_writer.add_write(json_path, json_data)
                        
                        # Mark as completed
                        progress_tracker.mark_completed(image_id)
                        extracted_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logging.error(f"❌ Failed to extract {image_id} from {tar_name}: {e}")
                    failed_count += 1
                
                processed_count += 1
                
                # Progress update
                if processed_count % 1000 == 0:
                    progress_pct = (processed_count / len(metadata_list)) * 100
                    logging.debug(f"   Progress: {processed_count:,}/{len(metadata_list):,} ({progress_pct:.1f}%) - Extracted: {extracted_count:,}")
    
    except Exception as e:
        logging.error(f"❌ Failed to process tar file {tar_name}: {e}")
        with stats_lock:
            stats['failed'] += len(metadata_list) - processed_count
        return
    
    # Update final stats
    with stats_lock:
        stats['skipped'] += skipped_count
        stats['not_found'] += failed_count
    
    # Log completion
    mem_usage = psutil.Process().memory_info().rss / 1024**3
    logging.info(f"✅ Completed {tar_name}: {extracted_count:,} extracted, {skipped_count:,} skipped, {failed_count:,} failed | Memory: {mem_usage:.1f}GB")

def process_extractions_by_tar_file(grouped_metadata: Dict[str, List[Dict[str, Any]]],
                                   cfg: Config, dest_dir: Path,
                                   source_dir: Path,
                                   stop_handler: Optional[SoftStopHandler] = None) -> None:
    """
    Process extractions grouped by tar file for maximum efficiency.
    Each tar file is opened once and all needed files are extracted.
    """
    
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)
    
    json_writer: Optional[Union[BatchedJSONWriter, AsyncJSONWriter]] = None
    if cfg.per_image_json:
        if cfg.json_writer_workers and cfg.json_writer_workers > 0:
            json_writer = AsyncJSONWriter(workers=cfg.json_writer_workers, enable_fsync=cfg.enable_fsync)
        else:
            json_writer = BatchedJSONWriter(flush_interval=cfg.json_flush_interval, batch_size=1000, enable_fsync=cfg.enable_fsync)
    
    progress_tracker = ValidatingProgressTracker(dest_dir / "progress.json", update_interval=cfg.progress_update_interval, enable_fsync=cfg.enable_fsync)
    
    if cfg.validate_on_start:
        if cfg.full_scan:
            progress_tracker.full_filesystem_scan(sharding)
        else:
            progress_tracker.validate_files(sharding, auto_clean=True)
    
    stats: Dict[str, Any] = {
        'extracted': 0,
        'failed': 0,
        'skipped': 0,
        'not_found': 0,
        'total_bytes_written': 0,
        'write_time': 0.0
    }
    stats_lock = threading.Lock()
    
    progress_reporter = ProgressReporter(interval=10.0)
    
    if stop_handler:
        if json_writer:
            stop_handler.add_flush_hook(lambda: json_writer.close())
        stop_handler.add_flush_hook(lambda: progress_tracker.save_final())
    
    initial_stats = progress_tracker.get_statistics()
    logging.info(f"📊 Starting with {initial_stats['completed']:,} completed, {initial_stats['missing']:,} missing")
    logging.info(f"🚀 Using tar-grouped extraction with {cfg.workers} workers")
    logging.info(f"💾 RAID optimized: {cfg.write_buffer_size_mb}MB write buffers")
    
    try:
        total_tar_files = len(grouped_metadata)
        total_images = sum(len(files) for files in grouped_metadata.values())
        logging.info(f"📦 Processing {total_tar_files} tar files containing {total_images:,} images")
        
        # Process tar files in order (0000.tar, 0001.tar, etc.)
        with ThreadPoolExecutor(max_workers=min(cfg.workers, 4)) as executor:  # Limit concurrent tar files
            futures = []
            
            for i, (tar_name, metadata_list) in enumerate(grouped_metadata.items()):
                if stop_handler and stop_handler.should_stop():
                    logging.info("🛑 Stopping extraction due to user interrupt...")
                    break
                
                logging.info(f"🔄 Queuing {tar_name} ({i+1}/{total_tar_files}) - {len(metadata_list):,} files")
                
                future = executor.submit(
                    process_tar_file_streaming,
                    tar_name, metadata_list, cfg, source_dir,
                    sharding, progress_tracker, json_writer,
                    stats, stats_lock, stop_handler
                )
                futures.append(future)
                
                # Limit the number of concurrent tar files to prevent memory issues
                if len(futures) >= min(cfg.workers // 2, 2):
                    # Wait for at least one to complete
                    for future in as_completed(futures[:1]):
                        try:
                            future.result()
                        except Exception as e:
                            logging.error(f"❌ Tar processing failed: {e}")
                    futures = futures[1:]  # Remove completed future
                
                # Report progress
                if progress_reporter.should_report():
                    progress_reporter.report(stats)
            
            # Wait for remaining futures
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"❌ Tar processing failed: {e}")
        
        # Final progress report
        progress_reporter.report(stats, force=True)
        
        logging.info(
            f"✅ Extraction complete: {stats['extracted']:,} extracted, "
            f"{stats['failed']:,} failed, {stats['skipped']:,} skipped, "
            f"{stats['not_found']:,} not found"
        )
        
    finally:
        if json_writer:
            json_writer.close()
        progress_tracker.save_final()

# ---------------------------------------------------------------------------
# Tar Index
# ---------------------------------------------------------------------------
class TarIndex:
    """Index for finding images in tar files."""
    
    COMMON_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
    
    def __init__(self, source_dir: Path, rebuild: bool = False):
        self.source_dir = source_dir
        self.index: Dict[int, str] = {}
        self.index_paths: Dict[int, str] = {}
        self.lock = threading.Lock()
        
        cache_file = source_dir / ".tar_index_cache.json"
        
        if cache_file.exists() and not rebuild:
            self._load_existing_cache(cache_file)
        else:
            logging.info("📚 Building new tar index...")
            self._build_comprehensive_index()
            self._save_cache(cache_file)
    
    def _load_existing_cache(self, cache_file: Path):
        """Load the existing tar_index_cache.json."""
        try:
            logging.info("📚 Loading existing tar index cache...")
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            if 'image_to_tar' in data:
                image_to_tar = data['image_to_tar']
                
                for key, tar_name in image_to_tar.items():
                    if '.' in key:
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
                
                samples = list(self.index.items())[:5]
                for fid, tar in samples:
                    logging.info(f"  File ID {fid} -> {tar}")
                
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
            if 'files' in data:
                if isinstance(data['files'], dict):
                    paths.extend(data['files'].keys())
                elif isinstance(data['files'], list):
                    paths.extend(data['files'])
            else:
                for k in data.keys():
                    if k.isdigit() or '.' in k:
                        paths.append(k)
        elif isinstance(data, list):
            paths.extend([str(x) for x in data if x])
        return paths

    def _build_comprehensive_index(self):
        """Build index by scanning tar files directly and their manifests."""
        logging.info("📚 Building comprehensive tar index...")
        tar_files = sorted(self.source_dir.glob("*.tar"))
        
        if not tar_files:
            logging.error(f"⚠️ No tar files found in {self.source_dir}")
            return
        
        total_files_indexed = 0
        
        for tar_path in tar_files:
            tar_name = tar_path.name
            logging.info(f"📦 Indexing {tar_name}...")
            
            # First try JSON manifest
            json_indexed = self._index_from_json_manifest(tar_path, tar_name)
            
            # If no JSON or insufficient entries, scan tar directly
            if json_indexed < 100:  # Threshold for considering manifest incomplete
                logging.info(f"   JSON manifest incomplete ({json_indexed} entries), scanning tar directly...")
                tar_indexed = self._scan_tar_contents(tar_path, tar_name)
                total_files_indexed += tar_indexed
            else:
                total_files_indexed += json_indexed
        
        logging.info(f"✅ Indexed {len(self.index):,} unique file IDs across {len(tar_files)} tar files")
        logging.info(f"   Total file mappings: {total_files_indexed:,}")

    def _scan_tar_contents(self, tar_path: Path, tar_name: str) -> int:
        """Scan tar file contents directly to build index."""
        indexed_count = 0
        try:
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                logging.info(f"   Scanning {len(members)} members in {tar_name}...")
                
                for member in members:
                    if member.isfile():
                        # Extract ID from filename
                        basename = os.path.basename(member.name)
                        name_without_ext = os.path.splitext(basename)[0]
                        
                        # Try to parse as integer ID
                        try:
                            file_id = int(name_without_ext)
                            self.index[file_id] = tar_name
                            self.index_paths[file_id] = member.name
                            indexed_count += 1
                        except ValueError:
                            # Not a numeric ID, check if it contains an ID
                            match = re.search(r'(\d{4,})', name_without_ext)
                            if match:
                                file_id = int(match.group(1))
                                self.index[file_id] = tar_name
                                self.index_paths[file_id] = member.name
                                indexed_count += 1
                
                logging.info(f"   ✅ Indexed {indexed_count} files from {tar_name}")
                
        except Exception as e:
            logging.error(f"   ❌ Failed to scan {tar_name}: {e}")
        
        return indexed_count

    def _index_from_json_manifest(self, tar_path: Path, tar_name: str) -> int:
        """Try to index from JSON manifest file."""
        indexed_count = 0
        
        # Try different possible manifest filenames
        possible_manifests = [
            tar_path.with_suffix('.json'),
            tar_path.parent / f"{tar_path.stem}_manifest.json",
            tar_path.parent / f"{tar_path.stem}_files.json",
        ]
        
        for manifest_path in possible_manifests:
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Parse different manifest formats
                    if isinstance(data, dict):
                        if 'files' in data:
                            # Format: {"files": {"12345.jpg": {...}, ...}}
                            for filename in data['files']:
                                file_id = self._extract_id_from_filename(filename)
                                if file_id:
                                    self.index[file_id] = tar_name
                                    self.index_paths[file_id] = filename
                                    indexed_count += 1
                        else:
                            # Format: {"12345.jpg": {...}, ...} or {"12345": "path/to/file.jpg"}
                            for key, value in data.items():
                                file_id = self._extract_id_from_filename(key)
                                if file_id:
                                    self.index[file_id] = tar_name
                                    if isinstance(value, str):
                                        self.index_paths[file_id] = value
                                    else:
                                        self.index_paths[file_id] = key
                                    indexed_count += 1
                    elif isinstance(data, list):
                        # Format: ["12345.jpg", "67890.png", ...]
                        for filename in data:
                            file_id = self._extract_id_from_filename(filename)
                            if file_id:
                                self.index[file_id] = tar_name
                                self.index_paths[file_id] = filename
                                indexed_count += 1
                    
                    if indexed_count > 0:
                        logging.info(f"   📄 Indexed {indexed_count} files from {manifest_path.name}")
                        break
                        
                except Exception as e:
                    logging.debug(f"   Could not parse {manifest_path}: {e}")
        
        return indexed_count

    def _extract_id_from_filename(self, filename: str) -> Optional[int]:
        """Extract numeric ID from a filename."""
        if not isinstance(filename, str):
            return None
        
        # Remove path components
        basename = os.path.basename(filename)
        # Remove extension
        name_without_ext = os.path.splitext(basename)[0]
        
        # Try direct integer conversion
        try:
            return int(name_without_ext)
        except ValueError:
            # Try to extract first sequence of digits (at least 4 digits for danbooru IDs)
            match = re.search(r'(\d{4,})', name_without_ext)
            if match:
                return int(match.group(1))
        
        return None

    def _save_cache(self, cache_file: Path):
        """Save the index to cache file."""
        try:
            cache_data = {
                'image_to_tar': {str(fid): tar for fid, tar in self.index.items()},
                'image_paths': {str(fid): path for fid, path in self.index_paths.items()},
                'total_images': len(self.index),
                'unique_tars': len(set(self.index.values())),
                'created_at': time.time()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logging.info(f"💾 Saved tar index cache with {len(self.index):,} mappings")
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")
    
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
        
        if self.index:
            sample = list(self.index.items())[:5]
            logging.info("📋 Sample ID mappings:")
            for fid, tar in sample:
                path = self.index_paths.get(fid, f"{fid}.jpg")
                logging.info(f"   ID {fid} -> {tar} (path: {path})")
    
    def find_image(self, image_id: int, file_url: str = "", md5: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Find image by ID in tar files with fallback strategies."""
        with self.lock:
            # Primary: direct ID lookup
            if image_id in self.index:
                tar_name = self.index[image_id]
                internal_path = self.index_paths.get(image_id, f"{image_id}.jpg")
                return (tar_name, internal_path)
            
            # Fallback 1: Try to extract from file_url if provided
            if file_url:
                # Extract potential tar name from URL
                # URLs might be like: .../images/1234/5678.jpg
                url_parts = file_url.split('/')
                for i, part in enumerate(url_parts):
                    if part.endswith('.tar'):
                        tar_name = part
                        # Reconstruct internal path
                        if i + 1 < len(url_parts):
                            internal_path = '/'.join(url_parts[i+1:])
                            return (tar_name, internal_path)
                
                # Try to guess tar file based on ID ranges
                # Many datasets organize by ID ranges (e.g., 0-999999.tar, 1000000-1999999.tar)
                tar_files = list(self.source_dir.glob("*.tar"))
                for tar_path in tar_files:
                    tar_name = tar_path.name
                    # Check if tar name contains number ranges
                    if re.search(rf'{image_id}', tar_name) or re.search(rf'{str(image_id)[:3]}', tar_name):
                        # Make educated guess about internal path
                        for ext in self.COMMON_EXTENSIONS:
                            potential_path = f"{image_id}{ext}"
                            return (tar_name, potential_path)
        
        # No match found
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics for debugging."""
        return {
            'total_images': len(self.index),
            'unique_tars': len(set(self.index.values())) if self.index else 0,
            'sample_mappings': dict(list(self.index.items())[:10]) if self.index else {}
        }

    def validate_index(self, sample_size: int = 10) -> bool:
        """Validate that the index is working by checking sample entries."""
        if not self.index:
            logging.error("❌ Index is empty!")
            return False
        
        sample_ids = list(self.index.keys())[:sample_size]
        logging.info(f"🔍 Validating index with {len(sample_ids)} sample entries...")
        
        valid_count = 0
        for image_id in sample_ids:
            tar_name = self.index[image_id]
            tar_path = self.source_dir / tar_name
            if tar_path.exists():
                valid_count += 1
                logging.debug(f"   ✅ ID {image_id} -> {tar_name} (exists)")
            else:
                logging.warning(f"   ❌ ID {image_id} -> {tar_name} (NOT FOUND)")
        
        success_rate = valid_count / len(sample_ids) if sample_ids else 0
        logging.info(f"📊 Validation: {valid_count}/{len(sample_ids)} tar files exist ({success_rate*100:.1f}%)")
        return success_rate > 0.5

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
        description="Filter and extract Danbooru images from local tar archives.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    p.add_argument("--metadata", type=str, help="Path to Danbooru parquet")
    p.add_argument("--source", type=str, help="Source directory with tar files")
    p.add_argument("--output", type=str, help="Destination directory")
    
    p.add_argument("--include", "-i", type=_parse_tag_list, help="Tags to include (enables filter)")
    p.add_argument("--exclude", "-x", type=_parse_tag_list, help="Tags to exclude (enables filter)")
    
    p.add_argument("--include-characters", type=_parse_tag_list, help="Characters to include")
    p.add_argument("--exclude-characters", type=_parse_tag_list, help="Characters to exclude")
    p.add_argument("--include-copyrights", type=_parse_tag_list, help="Copyrights to include")
    p.add_argument("--exclude-copyrights", type=_parse_tag_list, help="Copyrights to exclude")
    p.add_argument("--include-artists", type=_parse_tag_list, help="Artists to include")
    p.add_argument("--exclude-artists", type=_parse_tag_list, help="Artists to exclude")
    
    p.add_argument("--min-score", type=int, help="Minimum score (enables filter)")
    p.add_argument("--ratings", nargs="*", help="Allowed ratings (enables filter)")
    p.add_argument("--square", action="store_true", help="Require square images (enables filter)")
    p.add_argument("--min-square-size", type=int, help="Min dimension for square images")
    p.add_argument("--min-width", type=int, help="Minimum width (enables filter)")
    p.add_argument("--min-height", type=int, help="Minimum height (enables filter)")
    p.add_argument("--max-width", type=int, help="Maximum width")
    p.add_argument("--max-height", type=int, help="Maximum height")
    
    p.add_argument("--no-extract", dest="extract", action="store_false", help="Skip extraction")
    p.add_argument("--no-json", dest="json", action="store_false", help="Don't write JSON metadata")
    p.add_argument("--dry-run", action="store_true", help="Show matches without extracting")
    p.add_argument("--include-gifs", action="store_true", help="Include .gif files")
    p.add_argument("--no-validate", dest="validate", action="store_false", help="Skip validation")
    p.add_argument("--full-scan", action="store_true", help="Perform full filesystem scan")
    p.add_argument("--rebuild-index", action="store_true", help="Rebuild tar index")
    
    p.add_argument("--workers", type=int, help="Number of extraction workers")
    p.add_argument("--io-workers", type=int, help="Number of I/O workers")
    p.add_argument("--batch-size", type=int, help="Streaming batch size")
    
    return p

def apply_cli_overrides(args: argparse.Namespace, cfg: Config) -> None:
    """Mutates cfg in-place based on CLI arguments."""
    if args.metadata:
        cfg.metadata_db_path = args.metadata
    if args.source:
        cfg.source_images_dir = args.source
    if args.output:
        cfg.output_dir = args.output
    
    if args.include is not None:
        cfg.include_tags = args.include
        cfg.enable_include_tags = True
    if args.exclude is not None:
        cfg.exclude_tags = args.exclude
        cfg.enable_exclude_tags = True
    
    if args.include_characters is not None:
        cfg.include_characters = args.include_characters
        cfg.enable_character_filtering = True
    if args.exclude_characters is not None:
        cfg.exclude_characters = args.exclude_characters
        cfg.enable_character_filtering = True
    
    if args.include_copyrights is not None:
        cfg.include_copyrights = args.include_copyrights
        cfg.enable_copyright_filtering = True
    if args.exclude_copyrights is not None:
        cfg.exclude_copyrights = args.exclude_copyrights
        cfg.enable_copyright_filtering = True
    
    if args.include_artists is not None:
        cfg.include_artists = args.include_artists
        cfg.enable_artist_filtering = True
    if args.exclude_artists is not None:
        cfg.exclude_artists = args.exclude_artists
        cfg.enable_artist_filtering = True
    
    if args.min_score is not None:
        cfg.min_score = args.min_score
        cfg.enable_score_filtering = True
    
    if args.ratings is not None:
        cfg.ratings = args.ratings
        cfg.enable_rating_filtering = True
    
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
    
    if hasattr(args, 'extract') and not args.extract:
        cfg.extract_images = False
    if hasattr(args, 'json') and not args.json:
        cfg.per_image_json = False
    if args.dry_run:
        cfg.dry_run = True
        cfg.extract_images = False
    if args.include_gifs:
        cfg.exclude_gifs = False
    if hasattr(args, 'validate') and not args.validate:
        cfg.validate_on_start = False
    if args.full_scan:
        cfg.full_scan = True
    if args.rebuild_index:
        cfg.rebuild_tar_index = True
    
    if args.workers is not None:
        cfg.workers = args.workers
    if args.io_workers is not None:
        cfg.io_workers = args.io_workers
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Main function to orchestrate the filtering and extraction process."""
    cfg = Config()
    parser = build_cli()
    args = parser.parse_args()
    apply_cli_overrides(args, cfg)
    
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    
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
    
    try:
        import polars as pl
        pl_version = pl.__version__
        logging.info(f"📦 Using Polars version: {pl_version}")
        os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())
    except Exception as e:
        logging.warning(f"⚠️ Could not check Polars version: {e}")
    
    detect_metadata_structure(meta_path, cfg)
    
    logging.info("🗂️ Initializing tar index...")
    tar_index = TarIndex(source_dir, rebuild=cfg.rebuild_tar_index)
    
    index_stats = tar_index.get_statistics()
    logging.info(f"📊 Index statistics:")
    logging.info(f"   Total indexed file IDs: {index_stats['total_images']:,}")
    logging.info(f"   Unique tar files: {index_stats.get('unique_tars', 0)}")
    if index_stats.get('sample_mappings'):
        logging.info(f"   Sample mappings: {index_stats['sample_mappings']}")

    # Validate the index
    if not tar_index.validate_index():
        logging.error("❌ Tar index validation failed! Check your tar files and manifests.")
        sys.exit(1)
    
    logging.info("ℹ️ NOTE: Using 'id' column as primary identifier")
    logging.info("   This column contains the actual image IDs that match tar files")
    logging.info("   Each metadata row describes the image with that row's ID")
    
    with SoftStopHandler() as stop_handler:
        logging.info(f"📖 Collecting and grouping metadata from {meta_path}")
        grouped_metadata = collect_and_group_metadata(meta_path, cfg, tar_index, stop_handler)
        
        if not grouped_metadata:
            logging.warning("⚠️ No matching metadata found!")
            return
        
        total_images = sum(len(files) for files in grouped_metadata.values())
        
        if cfg.dry_run:
            logging.info(f"🎯 Dry run: {total_images:,} images match criteria across {len(grouped_metadata)} tar files")
            return
        
        if cfg.extract_images:
            logging.info("🔄 Using tar-grouped extraction for maximum efficiency...")
            process_extractions_by_tar_file(grouped_metadata, cfg, out_dir, source_dir, stop_handler)
        else:
            logging.info("📄 Filtering metadata only (extraction disabled)")
            logging.info(f"✅ Found {total_images:,} images matching criteria across {len(grouped_metadata)} tar files")
    
    logging.info("🎉 Script completed successfully!")

if __name__ == "__main__":
    main()
