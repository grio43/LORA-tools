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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Set, Tuple, Callable
from functools import lru_cache
import io

import polars as pl
import psutil

# --- Lightweight zero-copy reader for bytes-like buffers or mmap ---
class _BufferReader(io.RawIOBase):
    """Minimal file-like wrapper over a bytes-like object (bytearray, memoryview, mmap).
    Implements read/readinto/seek/tell so tarfile can use it without copying.
    """
    def __init__(self, buf):
        self._buf = memoryview(buf)
        self._pos = 0
        self._closed = False

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = len(self._buf) + offset
        else:
            raise ValueError("invalid whence")
        if new < 0:
            raise ValueError("negative seek position")
        self._pos = int(new)
        return self._pos

    def readinto(self, b):
        if self._closed:
            return 0
        n = min(len(b), len(self._buf) - self._pos)
        if n <= 0:
            return 0
        mv = self._buf[self._pos:self._pos + n]
        b[:n] = mv
        self._pos += n
        return n

    def read(self, n=-1):
        if self._closed:
            return b""
        if n is None or n < 0:
            n = len(self._buf) - self._pos
        n = min(n, len(self._buf) - self._pos)
        if n <= 0:
            return b""
        start = self._pos
        self._pos += n
        return bytes(self._buf[start:self._pos])

    def close(self):
        self._closed = True
        try:
            self._buf.release()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Optimized Configuration for RAID
# ---------------------------------------------------------------------------
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

    # ---- RAID-Optimized Performance ---------------------------------------
    workers: int = 24  # Increased for 32 threads
    io_workers: int = 12  # Dedicated I/O workers for RAID
    files_per_shard: int = 10000
    batch_size: int = 1000 # Larger batches for streaming
    tar_batch_size: int = 5000  # Much larger for RAID
    use_streaming: bool = False
    
    # New RAID optimizations
    # Tune prefetching and buffer sizes based on RAID characteristics.
    # With a RAID 5 source we prefer slightly more prefetching to hide parity
    # read latency, so default to two concurrent prefetch threads.  Writes are
    # bound by the RAID 0 destination’s throughput, so use a larger write buffer.
    prefetch_tars: int = 2
    max_memory_cache_gb: int = 100  # Total RAM allocated for caching tars
    parallel_tar_extraction: bool = False  # Extract from multiple tars simultaneously
    write_buffer_size_mb: int = 512  # Larger write buffer for high‑throughput RAID 0
    read_buffer_size_mb: int = 1024  # Larger read buffer for RAID 5 sequential reads
    use_memory_mapping: bool = False  # Disable to prevent double memory usage
    concurrent_tar_limit: int = 2  # Limit concurrent tars to avoid disk contention
    memory_high_water_gb: int = 100  # Pause processing above this memory usage
    memory_low_water_gb: int = 40  # Resume processing below this memory usage
    max_pending_batches: int = 5  # Maximum tar batches to keep in memory
    enable_fsync: bool = False  # Disable fsync for speed (filesystem will handle it)
    # ---- Large TAR handling policy -----------------------------------
    # How to handle tar files >= large_tar_threshold_gb
    #   - "mmap": memory-map large tars for zero-copy random access
    #   - "full": fully cache large tars in RAM (ensure enough memory!)
    #   - "skip": do not cache; read directly from disk
    large_tars_policy: str = "mmap"
    large_tar_threshold_gb: int = 8
    # JSON writer performance
    json_flush_interval: float = 5.0
    # Basename collision handling in tar members:
    #   - "strict": require exact full path when collisions exist; otherwise skip
    #   - "prefer_top": prefer top-level entry if present; else shortest path
    on_basename_collision: str = "prefer_top"
    # Session policy: disable further mmaps after N failures (NAS/RAID quirks)
    mmap_disable_after_n: int = 3
    # Bound memory usage of not-found sampling
    not_found_max_sample: int = 5000

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
# Memory Cache Manager
# ---------------------------------------------------------------------------

class MemoryCacheManager:
    """Manages in-memory caching of tar files for fast access."""
    
    def __init__(
        self,
        max_size_gb: int = 20,
        read_buffer_size_mb: int = 256,
        use_memory_mapping: bool = False,
        large_tar_threshold_gb: int = 8,
        large_tars_policy: str = "mmap",
        mmap_disable_after_n: int = 3,
        prefetch_threads: int = 2,
    ):
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.read_buffer_size_mb = read_buffer_size_mb
        self.cache: Dict[str, bytearray] = {}
        self.cache_sizes: Dict[str, int] = {}
        self.access_times: Dict[str, float] = {}
        self.total_size = 0
        self.lock = threading.Lock()
        self.loading_tars: Dict[str, threading.Event] = {}  # Track tars being loaded with events
        self.prefetch_queue = []  # Simple list instead of Queue
        self.prefetch_lock = threading.Lock()
        # Bound the number of background prefetch threads to at least one.  This
        # value can be tuned via the ``prefetch_threads`` parameter to better
        # match the underlying storage device characteristics.  When reading
        # from a RAID 5 array, a small number of prefetch threads prevents
        # saturating the stripe set; for RAID 0 a higher number may improve
        # throughput.
        self.prefetch_threads = max(1, int(prefetch_threads))
        self.mmap_handles: Dict[str, mmap.mmap] = {}  # Memory mapped files
        self.use_memory_mapping = use_memory_mapping
        # Large tar handling
        self.large_tar_threshold_bytes = large_tar_threshold_gb * 1024 ** 3
        self.large_tar_policy = large_tars_policy
        self._closed = False
        self.prefetch_threads_list = []
        self._start_prefetch_worker()
        self.in_flight_loads: Set[str] = set()
        self.mmap_failures: int = 0
        self.mmap_disable_after_n = max(1, int(mmap_disable_after_n))

    def _start_prefetch_worker(self):
        """Start background prefetch worker."""
        def prefetch_worker():
            while True:
                try:
                    if self._closed:
                        break
                    
                    with self.prefetch_lock:
                        # Find first tar not already being loaded
                        tar_path = None
                        for i, path in enumerate(self.prefetch_queue):
                            if path and path.name not in self.in_flight_loads:
                                tar_path = self.prefetch_queue.pop(i)
                                break
                     
                    if tar_path is None:
                        time.sleep(0.1)
                        continue

                    self._load_tar_to_cache(tar_path)

                except Exception as e:
                    logging.error(f"Prefetch error: {e}")
                    
        # Start multiple prefetch threads for better throughput
        for i in range(self.prefetch_threads):
            thread = threading.Thread(target=prefetch_worker, daemon=True)
            thread.start()
            self.prefetch_threads_list.append(thread)

    def _try_memory_map(self, tar_path: Path) -> Optional[mmap.mmap]:
        """Try to memory-map a tar file for zero-copy access.

        When memory mapping is enabled this will attempt to map the entire file
        read‑only.  The descriptor is closed immediately after mapping to avoid
        leaking file descriptors.  Any I/O or OS error is logged at debug level
        and ``None`` is returned to indicate a fallback to direct reads is
        required.
        """
        if not self.use_memory_mapping:
            return None
        try:
            fd = os.open(tar_path, os.O_RDONLY)
            try:
                # Perform a stat to surface unusual filesystem errors early.
                os.fstat(fd)
            except Exception:
                os.close(fd)
                raise
            mm = mmap.mmap(fd, length=0, access=mmap.ACCESS_READ)
            os.close(fd)
            return mm
        except Exception as e:
            logging.debug(f"mmap open failed for {tar_path}: {e}")
            return None

    def _record_mmap_failure(self, tar_path: Optional[Path] = None) -> None:
        """Record a failed mmap attempt and adjust policy after too many failures.

        This helper increments an internal failure counter.  When the counter
        exceeds ``mmap_disable_after_n`` the session will disable further
        memory mapping and, if the large‑tar policy is still set to ``mmap``,
        will switch to ``skip`` so that large files are read directly from
        disk.  An optional ``tar_path`` can be supplied for additional debug
        logging.
        """
        self.mmap_failures += 1
        if tar_path:
            logging.debug(f"Recording mmap failure for {tar_path}")
        if self.mmap_failures >= self.mmap_disable_after_n:
            # Disable session mmaps after repeated failures
            self.use_memory_mapping = False
            # For large tars, prefer skipping cache and direct reads henceforth
            if self.large_tar_policy == "mmap":
                self.large_tar_policy = "skip"
            logging.warning(
                f"⚠️ Disabling memory mapping for session after {self.mmap_failures} failures; "
                f"falling back to direct reads for large tars.")
    
    
    
    def _load_tar_to_cache(self, tar_path: Path) -> Optional[bytes]:
        """Load tar file into memory cache, honoring large-file policy and using chunked streaming."""
        tar_name = tar_path.name
        # Fast-path checks and wait-on-load
        with self.lock:
            if tar_name in self.cache:
                self.access_times[tar_name] = time.time()
                return self.cache[tar_name]
            if tar_name in self.mmap_handles:
                self.access_times[tar_name] = time.time()
                return None
            load_event = self.loading_tars.get(tar_name)
        if load_event:
            logging.debug(f"Tar {tar_name} already being prepared, waiting...")
            if not load_event.wait(timeout=300):
                logging.error(f"Timeout waiting for {tar_name} to prepare")
                return None
            with self.lock:
                if tar_name in self.cache:
                    self.access_times[tar_name] = time.time()
                    return self.cache[tar_name]
                if tar_name in self.mmap_handles:
                    self.access_times[tar_name] = time.time()
                    return None
                return None
        # Mark as loading (covers both mmap and full-cache paths)
        with self.lock:
            evt = threading.Event()
            self.loading_tars[tar_name] = evt
            self.in_flight_loads.add(tar_name)

        try:
            file_size = tar_path.stat().st_size

            # Large tar handling policy
            if file_size >= self.large_tar_threshold_bytes:
                if self.large_tar_policy == "mmap" and self.use_memory_mapping:
                    mm = self._try_memory_map(tar_path)
                    if mm:
                        with self.lock:
                            self.mmap_handles[tar_name] = mm
                            self.access_times[tar_name] = time.time()
                        logging.info(f"✅ Using mmap for large tar: {tar_name}")
                        # Let the finally block release any waiters
                        return None
                    else:
                        logging.warning(f"⚠️ mmap requested but failed for {tar_name}. Falling back to direct reads.")
                        # Record the failure and allow normal loading logic to proceed
                        self._record_mmap_failure(tar_path)
                        return None
                elif self.large_tar_policy == "skip":
                    logging.info(f"⭐️ Skipping full-cache of large tar ({file_size/1024**3:.1f}GB): {tar_name}")
                    # Defer releasing the loading event to the finally block
                    return None
                # else: "full" -> proceed

            # Check capacity before loading
            if file_size > self.max_size_bytes:
                logging.warning(f"Tar file {tar_name} too large for cache ({file_size / 1024**3:.1f}GB)")
                # The finally block will release the loading event
                return None

            # Evict until enough room
            with self.lock:
                while self.total_size + file_size > self.max_size_bytes and self.cache:
                    self._evict_lru()

            # Read the file in chunks to avoid transient RSS spikes
            buffer_size = min(self.read_buffer_size_mb * 1024 * 1024, 512 * 1024 * 1024)
            data_ba = bytearray()
            with open(tar_path, 'rb') as f:
                while True:
                    chunk = f.read(buffer_size)
                    if not chunk:
                        break
                    data_ba.extend(chunk)
            # Keep as bytearray to avoid an extra copy; we will wrap with _BufferReader when opening
            data = data_ba

            with self.lock:
                self.cache[tar_name] = data
                self.cache_sizes[tar_name] = file_size
                self.access_times[tar_name] = time.time()
                self.total_size += file_size

            logging.info(f"✅ Cached {tar_name} (Cache: {self.total_size / 1024**3:.1f}/{self.max_size_bytes / 1024**3:.0f}GB)")
            return data

        except Exception as e:
            logging.error(f"Failed to cache {tar_name}: {e}")
            return None
        finally:
            # Ensure any waiters are released if we didn't already
            with self.lock:
                event = self.loading_tars.pop(tar_name, None)
                if event:
                    event.set()
                self.in_flight_loads.discard(tar_name)


    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        oldest = min(self.access_times.items(), key=lambda x: x[1])
        tar_name = oldest[0]
        
        size = self.cache_sizes[tar_name]
        del self.cache[tar_name]
        del self.cache_sizes[tar_name]
        del self.access_times[tar_name]
        self.total_size -= size

        # Also evict mmap handle if present
        if tar_name in self.mmap_handles:
            self.mmap_handles[tar_name].close()
            del self.mmap_handles[tar_name]        
        
        logging.debug(f"Evicted {tar_name} from cache")
    
    def get_tar_handle(self, tar_path: Path) -> Optional[tarfile.TarFile]:
        """Get a tar file handle, preferring cached or memory-mapped data.
        Always returns a *fresh* TarFile instance to avoid shared state.
        """
        tar_name = tar_path.name
        
        # Try to get from cache (bytearray)
        with self.lock:
            if tar_name in self.cache:
                self.access_times[tar_name] = time.time()
                data = self.cache[tar_name]
                return tarfile.open(fileobj=io.BufferedReader(_BufferReader(memoryview(data))), mode='r')

        # Try memory mapped version
        if self.use_memory_mapping:
            with self.lock:
                if tar_name in self.mmap_handles:
                    self.access_times[tar_name] = time.time()
                    mm = self.mmap_handles[tar_name]
                    return tarfile.open(fileobj=io.BufferedReader(_BufferReader(mm)), mode='r')
        
        # Load into cache if possible (may establish mmap per policy)
        data = self._load_tar_to_cache(tar_path)
        if self.use_memory_mapping:
            with self.lock:
                if tar_name in self.mmap_handles:
                    self.access_times[tar_name] = time.time()
                    mm = self.mmap_handles[tar_name]
                    return tarfile.open(fileobj=io.BufferedReader(_BufferReader(mm)), mode='r')
        if data:
            with self.lock:
                data = self.cache.get(tar_name, data)
            return tarfile.open(fileobj=io.BufferedReader(_BufferReader(memoryview(data))), mode='r')
        
        # Fall back to direct file access with large buffer
        return tarfile.open(tar_path, 'r')
    
    def prefetch(self, tar_paths: List[Path], priority: bool = False):
        """Queue tar files for prefetching."""
        for path in tar_paths:
            if path.name not in self.cache and path.name not in self.mmap_handles:
                with self.prefetch_lock:
                    if priority:
                        self.prefetch_queue.insert(0, path)
                    else:
                        self.prefetch_queue.append(path)
    
    def close(self):
        """Clean up resources."""
        self._closed = True
        
        # Signal any waiting threads
        with self.lock:
            for event in self.loading_tars.values():
                event.set()
        
        for thread in self.prefetch_threads_list:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logging.warning(f"⚠️ Prefetch thread {thread.name} did not terminate")
        # Close memory mapped files
        for mm in self.mmap_handles.values():
            mm.close()


# ---------------------------------------------------------------------------
# Memory Monitor
# ---------------------------------------------------------------------------
class MemoryMonitor:
    """Monitor memory usage and provide backpressure."""
    
    def __init__(self, high_water_gb: int = 45, low_water_gb: int = 35):
        self.high_water_bytes = high_water_gb * 1024**3
        self.low_water_bytes = low_water_gb * 1024**3
        self.is_paused = False
        self.lock = threading.Lock()
        
    def check_memory(self) -> Tuple[bool, float]:
        """Check memory and return (should_pause, current_usage_gb)."""
        process = psutil.Process()
        mem_usage = process.memory_info().rss
        mem_gb = mem_usage / 1024**3
        
        with self.lock:
            if mem_usage > self.high_water_bytes and not self.is_paused:
                self.is_paused = True
                logging.warning(f"⚠️ Memory usage high ({mem_gb:.1f}GB), pausing new processing...")
                return True, mem_gb
            elif mem_usage < self.low_water_bytes and self.is_paused:
                self.is_paused = False
                logging.info(f"✅ Memory usage normal ({mem_gb:.1f}GB), resuming processing...")
                return False, mem_gb
            
            return self.is_paused, mem_gb



# ---------------------------------------------------------------------------
# Parallel Tar Processor
# ---------------------------------------------------------------------------
class ParallelTarProcessor:
    """Process multiple tar files in parallel for RAID optimization."""
    
    def __init__(self, cfg: Config, tar_index, sharding, progress_tracker, json_writer):
        self.cfg = cfg
        self.tar_index = tar_index
        self.sharding = sharding
        self.progress_tracker = progress_tracker
        self.json_writer = json_writer
        self.cache_manager = MemoryCacheManager(
            cfg.max_memory_cache_gb,
            cfg.read_buffer_size_mb,
            cfg.use_memory_mapping,
            cfg.large_tar_threshold_gb,
            cfg.large_tars_policy,
            prefetch_threads=cfg.prefetch_tars,
        )
        self.memory_monitor = MemoryMonitor(cfg.memory_high_water_gb, cfg.memory_low_water_gb)

     
        
        # Worker pools
        # Separate pools for CPU-bound and I/O-bound operations
        self.extract_executor = ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="extract")
        
        # Statistics
        self.stats = {
            'extracted': 0,
            'failed': 0,
            'skipped': 0,
            'not_found': 0,
            'not_found_ids': deque(maxlen=cfg.not_found_max_sample),  # bounded sample
            'total_bytes_written': 0,
            'write_time': 0.0
        }
        self.stats_lock = threading.Lock()

        # Track timing
        self.last_write_report = time.time()        
        
        # Tar batches organized by tar file
        self.tar_batches: Dict[str, List[Tuple[int, str, Path, Dict]]] = defaultdict(list)
        self.active_tars: Set[str] = set()
        self.pending_futures = []
        self.futures_lock = threading.Lock()  # Lock for pending_futures
        self.tar_lock = threading.Lock()

        # Start memory monitoring thread
        self._start_memory_monitor()
     
    def _start_memory_monitor(self):
        """Start background memory monitoring."""
        def monitor_memory():
            while True:
                time.sleep(2)  # Check every 2 seconds
                should_pause, mem_gb = self.memory_monitor.check_memory()
                
                # Signal backpressure if needed
                if should_pause:
                    with self.tar_lock:
                        # Pause new submissions when memory is high
                        while self.memory_monitor.is_paused:
                            time.sleep(1)

        thread = threading.Thread(target=monitor_memory, daemon=True)
        thread.start()
    
    def process_row(self, row: Dict[str, Any]) -> bool:
        """Process a single metadata row. Returns True if should continue."""
        # Use the id column as the primary identifier - it matches the actual image
        image_id = row[self.cfg.id_col]  # This is the actual image ID that matches tar files
        file_url = row.get(self.cfg.file_path_col, "")
        
        # Check memory and actually pause if needed
        is_paused, mem_gb = self.memory_monitor.check_memory()
        while is_paused:
            logging.info(f"⏸️ Paused due to high memory ({mem_gb:.1f}GB), waiting...")
            time.sleep(2)
            is_paused, mem_gb = self.memory_monitor.check_memory()

        # Use image_id for checking existence
        file_exists, ext = self.sharding.file_exists(image_id)
        is_completed = self.progress_tracker.is_completed(image_id)
        
        if file_exists and is_completed:
            with self.stats_lock:
                self.stats['skipped'] += 1
            return True
        elif file_exists and not is_completed:
            self.progress_tracker.mark_completed(image_id)
            with self.stats_lock:
                self.stats['skipped'] += 1
            return True
        
        # Find tar file
        tar_info = self.tar_index.find_image(image_id, file_url)
        if not tar_info:
            with self.stats_lock:
                self.stats['not_found'] += 1
                # Track individual IDs for auditing
                self.stats['not_found_ids'].append(image_id)                
            return True
        
        tar_name, filename = tar_info
        # Use image_id for output path
        shard_path = self.sharding.get_shard_path(image_id, create=False)
        ext_from_tar = os.path.splitext(filename)[1]  # Get extension from tar filename
        dest_path = shard_path / f"{image_id}{ext_from_tar}"
        
        # Add to batch
        should_submit = []
        with self.tar_lock:
            # Now using image_id consistently
            self.tar_batches[tar_name].append((image_id, filename, dest_path, row))


            total_pending = sum(len(batch) for batch in self.tar_batches.values())

            if total_pending > self.cfg.max_pending_batches * self.cfg.tar_batch_size:
                logging.debug(f"📦 {total_pending} items pending extraction")

            # Process if batch is large enough
            # Reduced concurrent tar processing to avoid write contention
            if len(self.tar_batches[tar_name]) >= self.cfg.tar_batch_size:
                should_submit.append(tar_name)
            # Process smaller batches too to avoid starvation
            elif len(self.active_tars) < self.cfg.concurrent_tar_limit:
                # Sort by batch size and submit multiple if we have capacity
                sorted_batches = sorted(
                    [(name, len(batch)) for name, batch in self.tar_batches.items() if batch],
                    key=lambda x: x[1], reverse=True
                )
                for tar_name, _ in sorted_batches[:self.cfg.concurrent_tar_limit - len(self.active_tars)]:
                    if tar_name not in self.active_tars:
                        should_submit.append(tar_name)

        # Submit extraction OUTSIDE the lock
        for tar_name_to_submit in should_submit:
            self._submit_tar_extraction(tar_name_to_submit)

        
        return True
    
    
    def _submit_tar_extraction(self, tar_name: str):
        """Submit a tar batch for extraction."""
        while True:
            with self.tar_lock:
                if tar_name in self.active_tars:
                    return
                if tar_name not in self.tar_batches or not self.tar_batches[tar_name]:
                    return
                if len(self.active_tars) < self.cfg.concurrent_tar_limit:
                    self.active_tars.add(tar_name)
                    break
        # Wait for capacity if too many active
        while len(self.active_tars) >= self.cfg.concurrent_tar_limit:
            time.sleep(0.1)
            self._cleanup_completed_futures()
        
        with self.tar_lock:
            # take-and-replace to avoid losing appends racing with clear
            existing = self.tar_batches.get(tar_name, [])
            if not existing:
                self.active_tars.discard(tar_name)
                return
            batch = existing
            self.tar_batches[tar_name] = []  # new list for any concurrent producers
        

        tar_path = Path(self.cfg.source_images_dir) / tar_name
        
        # Prefetch this tar to warm cache or mmap handle
        try:
            self.cache_manager.prefetch([tar_path], priority=True)
        except Exception:
            pass

        # Submit extraction job
        future = self.extract_executor.submit(
            self._extract_tar_batch_parallel,
            tar_path, batch
        )
        

        # Track future
        with self.futures_lock:
            self.pending_futures.append(future)
        
        # Handle completion asynchronously
        def handle_completion(f):
            try:
                successful_ids = f.result()
                with self.stats_lock:
                    self.stats['extracted'] += len(successful_ids)
                    self.stats['failed'] += len(batch) - len(successful_ids)
            except Exception as e:
                logging.error(f"Extraction failed for {tar_name}: {e}")
                with self.stats_lock:
                    self.stats['failed'] += len(batch)
            finally:
                with self.tar_lock:
                    self.active_tars.discard(tar_name)
        
        future.add_done_callback(handle_completion)

    def _cleanup_completed_futures(self):
        """Remove completed futures from tracking."""
        completed = []
        with self.futures_lock:
            if self.pending_futures:
                completed = [f for f in self.pending_futures if f.done()]
                for f in completed:
                    self.pending_futures.remove(f)
        return len(completed)
    
    def _extract_tar_batch_parallel(self, tar_path: Path, 
                                   batch: List[Tuple[int, str, Path, Dict]]) -> List[int]:
        """Extract files from tar with parallel I/O operations."""
        successful_ids = []
        
        try:
            # Get tar handle (possibly from cache)
            tar = self.cache_manager.get_tar_handle(tar_path)
            if not tar:
                logging.error(f"Failed to open {tar_path}")
                return successful_ids
            
            try:                # Build member indices:
                #  - by BASENAME -> list[TarInfo] (for exact/known filename)
                #  - by numeric ID prefix -> list[TarInfo] (for extension probing)
                from collections import defaultdict as _dd
                members_by_base = _dd(list)
                members_by_id = _dd(list)
                needed_files = {fn.rsplit("/", 1)[-1] for _, fn, _, _ in batch}
                needed_ids = {img_id for (img_id, _, _, _) in batch}
                for member in tar:
                    base_name = member.name.rsplit("/", 1)[-1]
                    if base_name in needed_files:
                        members_by_base[base_name].append(member)
                    # Probe by numeric ID prefix for fallback
                    dot = base_name.find(".")
                    if dot > 0:
                        num = base_name[:dot]
                        if num.isdigit():
                            img_id = int(num)
                            if img_id in needed_ids:
                                members_by_id[img_id].append(member)

                # Track what we successfully process
                # Create a lock so concurrent extractfile() calls don't share state
                tar_io_lock = threading.Lock()
                batch_successful_ids = []                
                # Optimize chunk size for RAID stripe alignment
                chunk_size = max(50, len(batch) // self.cfg.io_workers)
                chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]
                
                with ThreadPoolExecutor(max_workers=max(1, min(self.cfg.io_workers, len(chunks)))) as io_pool:
                    futures = []
                    
                    for chunk in chunks:
                        future = io_pool.submit(
                            self._extract_chunk,
                            tar, tar_path.name, members_by_base, members_by_id, chunk, tar_io_lock
                        )
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        try:
                            ids = future.result()
                            batch_successful_ids.extend(ids)
                        except Exception as e:
                            logging.error(f"Chunk extraction failed: {e}")

                # Only mark as successful after all chunks complete
                successful_ids.extend(batch_successful_ids)

            finally:
                tar.close()
                
        except Exception as e:
            logging.error(f"Failed to process tar {tar_path}: {e}")
        
        # Mark completed AFTER successful extraction
        
        return successful_ids
    
    
    def _extract_chunk(self, tar: tarfile.TarFile, tar_name: str,
                       members_by_base: Dict[str, Any], members_by_id: Dict[int, Any],
                       chunk: List[Tuple[int, str, Path, Dict]], tar_io_lock: threading.Lock) -> List[int]:
        """Extract a chunk of files from tar.
        Access to tar is guarded by a lock because TarFile is not thread-safe.
        """
        successful_ids = []
        buffer_size = max(1, self.cfg.write_buffer_size_mb) * 1024 * 1024  # honor configured cap
        
        for image_id, filename, dest_path, row_data in chunk:
            try:
                # Look up by BASENAME (dictionary is keyed by basename)
                base = filename.rsplit('/', 1)[-1]
                candidates = members_by_base.get(base, [])
                member = None
                # If caller provided a full path, prefer exact match among candidates
                if '/' in filename and candidates:
                    for cand in candidates:
                        if cand.name == filename:
                            member = cand
                            break
                if member is None:
                    if len(candidates) == 1:
                        member = candidates[0]
                    elif len(candidates) > 1:
                        # Disambiguate according to policy
                        if self.cfg.on_basename_collision == "prefer_top":
                            top = [c for c in candidates if '/' not in c.name]
                            if len(top) == 1:
                                member = top[0]
                            else:
                                member = min(candidates, key=lambda c: c.name.count('/'))
                        else:
                            logging.debug(f"Ambiguous basename {base}, skipping (provide full path).")
                            member = None
                # Fallback: probe by numeric ID (handles extension mismatches)
                if not member:
                    id_candidates = members_by_id.get(image_id, [])
                    if len(id_candidates) == 1:
                        member = id_candidates[0]
                    elif len(id_candidates) > 1:
                        if self.cfg.on_basename_collision == "prefer_top":
                            top = [c for c in id_candidates if '/' not in c.name]
                            member = top[0] if len(top) == 1 else min(id_candidates, key=lambda c: c.name.count('/'))
                        else:
                            # deterministic but explicit: choose shortest path
                            member = min(id_candidates, key=lambda c: c.name.count('/')) if id_candidates else None
                
                if not member:
                    logging.debug(f"Member {filename} not found in tar")
                    continue
                
                # Extract file data
                try:
                    start_write = time.time()
                    with tar_io_lock:
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            data = file_obj.read()
                            file_obj.close()
                        
                        # Direct synchronous write with large buffer
                        # Recompute destination extension from member in case of ID->ext probing
                        ext_from_member = os.path.splitext(member.name)[1] or os.path.splitext(dest_path.name)[1]
                        final_dir = dest_path.parent
                        final_dir.mkdir(parents=True, exist_ok=True)
                        final_path = final_dir / f"{image_id}{ext_from_member}"
                        temp_path = final_path.with_suffix(final_path.suffix + '.tmp')
                        with open(temp_path, 'wb', buffering=buffer_size) as f:
                            f.write(data)
                            # Only fsync if explicitly enabled (usually disabled for speed)
                            if self.cfg.enable_fsync:
                                f.flush()
                                os.fsync(f.fileno())
                        temp_path.replace(final_path)
                        # Track write statistics
                        with self.stats_lock:
                            self.stats['total_bytes_written'] += len(data)
                            self.stats['write_time'] += (time.time() - start_write)

                    # Write JSON if needed
                    if self.cfg.per_image_json and self.json_writer:
                        json_path = final_dir / f"{image_id}.json"
                        json_data = prepare_json_data(row_data, self.cfg)
                        self.json_writer.add_write(json_path, json_data)
                        # Cache discovered internal path for future lookups
                        try:
                            if hasattr(self.tar_index, "cache_discovered_path"):
                                self.tar_index.cache_discovered_path(image_id, tar_name, member.name)
                        except Exception:
                            pass

                    # Only mark successful AFTER write completes
                    self.progress_tracker.mark_completed(image_id)                       
                    successful_ids.append(image_id)

                except Exception as e:
                    logging.debug(f"Direct extraction failed for {filename}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during extraction of {filename}: {e}")

        return successful_ids

    def finish_remaining(self):
        """Process all remaining batches."""
        logging.info(f"⏳ Processing {len(self.tar_batches)} remaining tar batches...")
        
        # Submit remaining batches in controlled manner
        with self.tar_lock:
            remaining = [(name, len(batch)) for name, batch in self.tar_batches.items() if batch]
            logging.info(f"📦 Remaining batches: {remaining}")
            
            # Process in smaller groups to control memory
            tar_names = list(self.tar_batches.keys())
            for i in range(0, len(tar_names), self.cfg.concurrent_tar_limit):
                group = tar_names[i:i+self.cfg.concurrent_tar_limit]
                
                for tar_name in group:
                    if self.tar_batches[tar_name] and tar_name not in self.active_tars:
                        self._submit_tar_extraction(tar_name)
                
                # Wait for this group to complete before starting next
                wait_start = time.time()
                while True:
                    with self.tar_lock:
                        if len(self.active_tars) == 0:
                            break
                    
                    # Add timeout to prevent infinite wait
                    if time.time() - wait_start > 600:  # 10 minute timeout
                        logging.error("⚠️ Timeout waiting for tar extraction, proceeding...")
                        break

                    time.sleep(0.5)
                    self._cleanup_completed_futures()
        
        # Wait for all to complete
        self.extract_executor.shutdown(wait=True)

        
        # Clean up cache
        logging.info("🧹 Cleaning up memory cache...")
        self.cache_manager.close()
    
    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        with self.stats_lock:
            return self.stats.copy()

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
            # Best-effort short flush of pending writes/progress
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
        return self.stop_event.is_set()

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
# Main optimized extraction function (THIS WAS MISSING!)
# ---------------------------------------------------------------------------
def process_extractions_optimized(metadata_stream: Iterator[Dict[str, Any]], 
                                 cfg: Config, dest_dir: Path,
                                 tar_index,
                                 stop_handler = None) -> None:
    """Process extractions with RAID optimizations."""
    
    # Initialize components
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)
    json_writer = BatchedJSONWriter(flush_interval=cfg.json_flush_interval, batch_size=1000, enable_fsync=cfg.enable_fsync) if cfg.per_image_json else None
    progress_tracker = ValidatingProgressTracker(dest_dir / "progress.json", update_interval=1000, enable_fsync=cfg.enable_fsync)
    
    # Validation
    if cfg.validate_on_start:
        if cfg.full_scan:
            progress_tracker.full_filesystem_scan(sharding)
        else:
            progress_tracker.validate_files(sharding, auto_clean=True)
    
    # Create parallel processor
    processor = ParallelTarProcessor(cfg, tar_index, sharding, progress_tracker, json_writer)
    progress_reporter = ProgressReporter(interval=10.0)
    # Register durability flush hooks for hard-exit
    if stop_handler:
        if json_writer:
            stop_handler.add_flush_hook(lambda: json_writer.close())
        stop_handler.add_flush_hook(lambda: progress_tracker.save_final())
    
    # Show starting stats
    stats = progress_tracker.get_statistics()
    logging.info(f"📊 Starting with {stats['completed']:,} completed, {stats['missing']:,} missing")
    logging.info(f"🚀 Using {cfg.workers} extraction workers, {cfg.io_workers} I/O workers")
    logging.info(f"💾 Memory cache: up to {cfg.max_memory_cache_gb}GB")
    logging.info(f"💽 RAID optimized: {cfg.read_buffer_size_mb}MB read, {cfg.write_buffer_size_mb}MB write buffers")
    
    try:
        # Process metadata stream
        batch_count = 0
        for row in metadata_stream:
            if stop_handler and stop_handler.should_stop():
                logging.info("🛑 Stopping extraction due to user interrupt...")
                break

            # Add periodic memory check and pause
            batch_count += 1
            if batch_count % 100 == 0:  # Check more frequently
                is_paused, mem_gb = processor.memory_monitor.check_memory()
                if is_paused:
                    logging.info(f"⏸️ Pausing new additions due to memory ({mem_gb:.1f}GB)...")
                    time.sleep(1)  # Give system time to process pending work
            
            
            processor.process_row(row)
            
            # Report progress
            if progress_reporter.should_report():
                progress_reporter.report(processor.get_stats())
        
        # Process remaining
        processor.finish_remaining()
        
        # Final report
        final_stats = processor.get_stats()
        progress_reporter.report(final_stats, force=True)
        # Persist a snapshot of sampled not-found IDs
        try:
            not_found_sample = list(final_stats.get('not_found_ids', []))
            if not_found_sample:
                with open(dest_dir / "not_found_ids.sample.json", "w", encoding="utf-8") as f:
                    json.dump({"count": final_stats.get('not_found', 0),
                               "sample_size": len(not_found_sample),
                               "ids": not_found_sample}, f, indent=2)
        except Exception as e:
            logging.warning(f"⚠️ Failed to persist not-found sample: {e}")

        logging.info(
            f"✅ Extraction complete: {final_stats['extracted']:,} extracted, "
            f"{final_stats['failed']:,} failed, {final_stats['skipped']:,} skipped, "
            f"{final_stats['not_found']:,} not found"
        )
        
    finally:
        # Cleanup
        if json_writer:
            json_writer.close()
        progress_tracker.save_final()

# ---------------------------------------------------------------------------
# Filtering Functions (CRITICAL - from online puller)
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
        for batch_df in collected.iter_slices(cfg.batch_size):
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
            # Use default filters (from Config class)
            %(prog)s

            # Custom tag filtering
            %(prog)s --include "1girl solo" --exclude "lowres"

            # Filter by score and rating
            %(prog)s --min-score 50 --ratings safe general

            # Character and copyright filtering
            %(prog)s --include-characters "hakurei_reimu" --include-copyrights "touhou"
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
    p.add_argument("--memory-cache", type=int, help="Memory cache size in GB")

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
    if args.memory_cache is not None:
        cfg.max_memory_cache_gb = args.memory_cache

# ---------------------------------------------------------------------------
# Tar Index (stub for now - implement based on your tar structure)
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
            logging.error("❌ No existing cache found. Building new index...")
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
            process_extractions_optimized(metadata_stream, cfg, out_dir, tar_index, stop_handler)
        else:
            logging.info("📄 Filtering metadata only (extraction disabled)")
            count = sum(1 for _ in metadata_stream)
            logging.info(f"✅ Found {count:,} images matching criteria")
    
    logging.info("🎉 Script completed successfully!")

if __name__ == "__main__":
    main()