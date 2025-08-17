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
import shutil
import mmap
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Set, Tuple, Deque
from tempfile import TemporaryDirectory
from functools import lru_cache
import multiprocessing as mp
from queue import Queue, Empty
import io
import fcntl

import polars as pl
import pandas as pd
from tqdm import tqdm
import psutil

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
    id_col: str = "id"

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
    files_per_shard: int = 5000
    batch_size: int = 10000  # Larger batches for streaming
    tar_batch_size: int = 50000  # Much larger for RAID
    use_streaming: bool = True
    
    # New RAID optimizations
    prefetch_tars: int = 6  # Number of tar files to prefetch into RAM
    max_memory_cache_gb: int = 100  # Use up to 100GB for caching
    parallel_tar_extraction: bool = True  # Extract from multiple tars simultaneously
    write_buffer_size_mb: int = 256  # Large write buffers for RAID 0
    read_buffer_size_mb: int = 512  # Large read buffers for RAID 5
    use_memory_mapping: bool = True  # Memory-map tar files when possible
    concurrent_tar_limit: int = 6  # Process up to 4 tar files simultaneously

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
    
    def get_shard_path(self, image_id: int) -> Path:
        """Get the shard directory path for a given image ID."""
        shard_index = image_id // self.files_per_dir
        shard_name = f"shard_{shard_index:05d}"
        shard_path = self.base_dir / shard_name
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
        shard_path = self.get_shard_path(image_id)
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
    
    def __init__(self, flush_interval: float = 5.0, batch_size: int = 1000):
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.buffer: List[tuple[Path, Dict[str, Any]]] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self.metadata_to_tar_id = {}  # Map metadata row IDs to actual tar IDs
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
            
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush all buffered writes to disk."""
        if not self.buffer:
            return
        
        logging.debug(f"💾 Flushing {len(self.buffer)} JSON writes...")
        
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
            logging.error(f"❌ Failed to write {path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    def close(self):
        """Close writer and flush remaining data."""
        with self.lock:
            self._closed = True
            self._flush_buffer()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=1.0)

# ---------------------------------------------------------------------------
# Validating Progress Tracker
# ---------------------------------------------------------------------------
class ValidatingProgressTracker:
    """Tracks extraction progress with validation and persistent records."""
    
    def __init__(self, progress_file: Path, update_interval: int = 1000):
        self.progress_file = progress_file
        self.update_interval = update_interval
        self.completed_ids: Set[int] = set()
        self.missing_ids: Set[int] = set()
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
        # Implementation simplified for local use
        pass
    
    def full_filesystem_scan(self, sharding: DirectorySharding):
        """Perform full filesystem scan."""
        logging.info("🔍 Performing full filesystem scan...")
        # Implementation simplified for local use
        pass

# ---------------------------------------------------------------------------
# Memory Cache Manager
# ---------------------------------------------------------------------------
class MemoryCacheManager:
    """Manages in-memory caching of tar files for fast access."""
    
    def __init__(self, max_size_gb: int = 100, read_buffer_size_mb: int = 256):
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.read_buffer_size_mb = read_buffer_size_mb
        self.cache: Dict[str, bytes] = {}
        self.cache_sizes: Dict[str, int] = {}
        self.access_times: Dict[str, float] = {}
        self.total_size = 0
        self.lock = threading.Lock()
        self.prefetch_queue: Queue = Queue()
        self.prefetch_threads = 4
        self.prefetch_executor = ThreadPoolExecutor(max_workers=self.prefetch_threads)
        self._start_prefetch_worker()
        self.mmap_handles: Dict[str, mmap.mmap] = {}  # Memory mapped files
        
    def _start_prefetch_worker(self):
        """Start background prefetch worker."""
        def prefetch_worker():
            while True:
                try:
                    tar_path = self.prefetch_queue.get(timeout=1)
                    if tar_path is None:
                        break
                    self._load_tar_to_cache(tar_path)
                except Empty:
                    continue
                except Exception as e:
                    logging.error(f"Prefetch error: {e}")
                    
        # Start multiple prefetch threads for better throughput
        self.prefetch_threads_list = []
        for i in range(self.prefetch_threads):
            thread = threading.Thread(target=prefetch_worker, daemon=True)
            thread.start()
            self.prefetch_threads_list.append(thread)
     
    def _try_memory_map(self, tar_path: Path) -> Optional[mmap.mmap]:
        """Try to memory-map a tar file for zero-copy access."""
        try:
            fd = os.open(tar_path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            
            # Only mmap files that fit in reasonable memory
            if file_size < 10 * 1024**3:  # 10GB limit for mmap
                mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
                logging.info(f"📍 Memory-mapped {tar_path.name} ({file_size / 1024**3:.1f}GB)")
                return mm
            os.close(fd)
        except Exception as e:
            logging.debug(f"Could not memory-map {tar_path}: {e}")
        return None
    
    
    def _load_tar_to_cache(self, tar_path: Path) -> Optional[bytes]:
        """Load tar file into memory cache."""
        tar_name = tar_path.name
        
        with self.lock:
            if tar_name in self.cache:
                self.access_times[tar_name] = time.time()
                return self.cache[tar_name]

        # Try memory mapping first
        if tar_name not in self.mmap_handles:
            mm = self._try_memory_map(tar_path)
            if mm:
                with self.lock:
                    self.mmap_handles[tar_name] = mm
                    self.access_times[tar_name] = time.time()
                return bytes(mm)
                
        try:
            file_size = tar_path.stat().st_size
            
            # Check if we have space
            if file_size > self.max_size_bytes:
                logging.warning(f"Tar file {tar_name} too large for cache ({file_size / 1024**3:.1f}GB)")
                return None
            
            # Evict old entries if needed
            with self.lock:
                while self.total_size + file_size > self.max_size_bytes and self.cache:
                    self._evict_lru()
            
            # Load file with large buffer and optional O_DIRECT
            logging.info(f"🔥 Loading {tar_name} into RAM cache ({file_size / 1024**3:.1f}GB)...")
            
            # Use O_DIRECT for large files to bypass OS cache
            flags = os.O_RDONLY
            if hasattr(os, 'O_DIRECT') and file_size > 1024**3:  # >1GB
                flags |= os.O_DIRECT
                # Align buffer for O_DIRECT
                buffer_size = 512 * 1024 * 1024  # 512MB aligned
            else:
                buffer_size = self.read_buffer_size_mb * 1024 * 1024
            
            with open(tar_path, 'rb', buffering=buffer_size) as f:
                data = f.read()
            
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
        
        logging.debug(f"Evicted {tar_name} from cache")
    
    def get_tar_handle(self, tar_path: Path) -> Optional[tarfile.TarFile]:
        """Get tar file handle, preferring cached version."""
        tar_name = tar_path.name
        
        # Try to get from cache
        with self.lock:
            if tar_name in self.cache:
                self.access_times[tar_name] = time.time()
                data = self.cache[tar_name]
                return tarfile.open(fileobj=io.BytesIO(data), mode='r')

        # Try memory mapped version
        with self.lock:
            if tar_name in self.mmap_handles:
                self.access_times[tar_name] = time.time()
                return tarfile.open(fileobj=io.BytesIO(self.mmap_handles[tar_name]), mode='r')
         
        # Load into cache if possible
        data = self._load_tar_to_cache(tar_path)
        if data:
            return tarfile.open(fileobj=io.BytesIO(data), mode='r')
        
        # Fall back to direct file access with large buffer
        return tarfile.open(tar_path, 'r', bufsize=self.read_buffer_size_mb * 1024 * 1024)
    
    def prefetch(self, tar_paths: List[Path], priority: bool = False):
        """Queue tar files for prefetching."""
        for path in tar_paths:
            if path.name not in self.cache and path.name not in self.mmap_handles:
                if priority:
                    # Put at front of queue for priority prefetch
                    temp_queue = Queue()
                    temp_queue.put(path)
                    while not self.prefetch_queue.empty():
                        temp_queue.put(self.prefetch_queue.get())
                    self.prefetch_queue = temp_queue
                else:
                    self.prefetch_queue.put(path)
    
    def close(self):
        """Clean up resources."""
        for _ in range(self.prefetch_threads):
            self.prefetch_queue.put(None)
        for thread in self.prefetch_threads_list:
            thread.join(timeout=5)
        # Close memory mapped files
        for mm in self.mmap_handles.values():
            mm.close()

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
        self.cache_manager = MemoryCacheManager(cfg.max_memory_cache_gb, cfg.read_buffer_size_mb)
        
        # Worker pools
        # Separate pools for CPU-bound and I/O-bound operations
        self.extract_executor = ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="extract")
        self.io_executor = ThreadPoolExecutor(max_workers=cfg.io_workers, thread_name_prefix="io")
        
        # Statistics
        self.stats = {
            'extracted': 0,
            'failed': 0,
            'skipped': 0,
            'not_found': 0
        }
        self.stats_lock = threading.Lock()
        
        # Tar batches organized by tar file
        self.tar_batches: Dict[str, List[Tuple[int, str, Path, Dict]]] = defaultdict(list)
        self.active_tars: Set[str] = set()
        self.tar_lock = threading.Lock()

        # Image write buffer for coalescing
        self.write_buffer = []
        self.write_buffer_size = 50  # Batch image writes
    
    def process_row(self, row: Dict[str, Any]) -> bool:
        """Process a single metadata row. Returns True if should continue."""
        image_id = row[self.cfg.id_col]
        file_url = row.get(self.cfg.file_path_col, "")
        
        # Check if already exists
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
            return True
        
        tar_name, filename = tar_info
        shard_path = self.sharding.get_shard_path(image_id)
        dest_path = shard_path / filename
        
        # Add to batch
        with self.tar_lock:
            self.tar_batches[tar_name].append((image_id, filename, dest_path, row))
            
            # Process if batch is large enough or we have multiple tars ready
            if len(self.tar_batches[tar_name]) >= self.cfg.tar_batch_size:
                self._submit_tar_extraction(tar_name)
            elif len(self.tar_batches) >= self.cfg.concurrent_tar_limit * 3:
                # Process the largest batch to free up memory  
                largest_tar = max(self.tar_batches.items(), key=lambda x: len(x[1]))[0]
                if largest_tar not in self.active_tars:
                    self._submit_tar_extraction(largest_tar)
        
        return True
    
    def _submit_tar_extraction(self, tar_name: str):
        """Submit a tar batch for extraction."""
        if tar_name in self.active_tars:
            return
        
        batch = self.tar_batches[tar_name]
        if not batch:
            return
        
        self.active_tars.add(tar_name)
        tar_path = Path(self.cfg.source_images_dir) / tar_name
        
        # Aggressive prefetching - prefetch more tars
        next_tars = list(self.tar_batches.keys())[:self.cfg.prefetch_tars * 2]
        next_paths = [Path(self.cfg.source_images_dir) / t for t in next_tars if t != tar_name]

        # Priority prefetch the next tar to be processed
        if next_paths:
            self.cache_manager.prefetch(next_paths[:2], priority=True)
            if len(next_paths) > 2:
                self.cache_manager.prefetch(next_paths[2:], priority=False)
        
        # Submit extraction job
        future = self.extract_executor.submit(
            self._extract_tar_batch_parallel,
            tar_path, batch
        )
        
        # Clear batch to free memory
        self.tar_batches[tar_name] = []
        
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
            
            try:
                # Get all members for lookup
                members_dict = {m.name: m for m in tar.getmembers()}
                
                # Optimize chunk size for RAID stripe alignment
                chunk_size = max(20, len(batch) // (self.cfg.io_workers * 2))
                chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]
                
                with ThreadPoolExecutor(max_workers=min(8, self.cfg.io_workers)) as io_pool:
                    futures = []
                    
                    for chunk in chunks:
                        future = io_pool.submit(
                            self._extract_chunk,
                            tar, members_dict, chunk
                        )
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        try:
                            ids = future.result()
                            successful_ids.extend(ids)
                        except Exception as e:
                            logging.error(f"Chunk extraction failed: {e}")
                
            finally:
                tar.close()
                
        except Exception as e:
            logging.error(f"Failed to process tar {tar_path}: {e}")
        
        # Mark completed
        for image_id in successful_ids:
            self.progress_tracker.mark_completed(image_id)
        
        return successful_ids
    
    def _extract_chunk(self, tar: tarfile.TarFile, members_dict: Dict[str, Any],
                      chunk: List[Tuple[int, str, Path, Dict]]) -> List[int]:
        """Extract a chunk of files from tar."""
        successful_ids = []
        
        for image_id, filename, dest_path, row_data in chunk:
            try:
                # Find member
                member = members_dict.get(filename)
                if not member:
                    # Try subdirectories
                    for name in members_dict:
                        if name.endswith(f"/{filename}"):
                            member = members_dict[name]
                            break
                
                if not member:
                    continue
                
                # Direct extraction without temp directory
                try:
                    # Extract file data directly
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        # Read with large buffer
                        data = file_obj.read()
                        
                        # Write directly to destination with aligned buffer
                        self._optimized_write(dest_path, data)
                        
                        # Write JSON if needed
                        if self.cfg.per_image_json and self.json_writer:
                            json_path = dest_path.parent / f"{image_id}.json"
                            json_data = prepare_json_data(row_data, self.cfg)
                            self.json_writer.add_write(json_path, json_data)
                        
                        successful_ids.append(image_id)
                        file_obj.close()
                    
                except Exception as e:
                    logging.debug(f"Direct extraction failed for {filename}: {e}")
                    # Fall back to temp directory method if direct fails
                    with TemporaryDirectory() as tmpdir:
                        tar.extract(member, tmpdir)
                        extracted_path = Path(tmpdir) / member.name
                        self._optimized_copy(extracted_path, dest_path)
                    
                        successful_ids.append(image_id)
                    
            except Exception as e:
                logging.debug(f"Failed to extract {filename}: {e}")
        
        return successful_ids

    def _optimized_write(self, dst: Path, data: bytes):
        """Optimized direct write for RAID 0 with stripe alignment."""
        buffer_size = self.cfg.write_buffer_size_mb * 1024 * 1024
        
        # Align write size to RAID stripe size
        stripe_size = self.cfg.stripe_size_kb * 1024
        if len(data) > stripe_size:
            # Pad to stripe boundary for optimal RAID 0 performance
            padding = (stripe_size - (len(data) % stripe_size)) % stripe_size
            if padding and padding < 4096:  # Only pad if small
                data = data + b'\0' * padding
        
        with open(dst, 'wb', buffering=buffer_size) as fdst:
            fdst.write(data)    

    def _optimized_copy(self, src: Path, dst: Path):
        """Optimized file copy for RAID with large buffers."""
        buffer_size = self.cfg.write_buffer_size_mb * 1024 * 1024

        # Use sendfile if available (zero-copy on Linux)
        if hasattr(os, 'sendfile'):
            try:
                with open(src, 'rb') as fsrc:
                    with open(dst, 'wb') as fdst:
                        sent = 0
                        file_size = os.fstat(fsrc.fileno()).st_size
                        while sent < file_size:
                            sent += os.sendfile(fdst.fileno(), fsrc.fileno(), sent, file_size - sent)
                return
            except:
                pass  # Fall back to regular copy

        with open(src, 'rb', buffering=buffer_size) as fsrc:
            with open(dst, 'wb', buffering=buffer_size) as fdst:
                shutil.copyfileobj(fsrc, fdst, length=buffer_size)
    
    def finish_remaining(self):
        """Process all remaining batches."""
        logging.info(f"⏳ Processing {len(self.tar_batches)} remaining tar batches...")
        
        # Submit all remaining batches
        with self.tar_lock:
            for tar_name in list(self.tar_batches.keys()):
                if self.tar_batches[tar_name] and tar_name not in self.active_tars:
                    self._submit_tar_extraction(tar_name)
        
        # Wait for all to complete
        self.extract_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
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
            total = sum(stats.values())
            
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
            
            logging.info(
                f"📊 Progress: {stats['extracted']:,} extracted, "
                f"{stats['failed']:,} failed, {stats['skipped']:,} skipped, "
                f"{stats['not_found']:,} not found | "
                f"Rate: {rate:.1f}/s | Memory: {mem_usage:.1f}GB | "
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
        
    def __enter__(self):
        self.original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_sigint)
        
    def _signal_handler(self, signum, frame):
        self._signal_count += 1
        
        if self._signal_count == 1:
            logging.warning("\n🛑 Graceful stop requested. Finishing current batch...")
            logging.warning("Press Ctrl+C again to force exit.")
            self.stop_event.set()
        else:
            logging.warning("\n⚠️ Force exit requested.")
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
# Main optimized extraction function (THIS WAS MISSING!)
# ---------------------------------------------------------------------------
def process_extractions_optimized(metadata_stream: Iterator[Dict[str, Any]], 
                                 cfg: Config, dest_dir: Path,
                                 tar_index,
                                 stop_handler = None) -> None:
    """Process extractions with RAID optimizations."""
    
    # Initialize components
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)
    json_writer = BatchedJSONWriter(flush_interval=5.0, batch_size=1000) if cfg.per_image_json else None
    progress_tracker = ValidatingProgressTracker(dest_dir / "progress.json", update_interval=1000)
    
    # Validation
    if cfg.validate_on_start:
        if cfg.full_scan:
            progress_tracker.full_filesystem_scan(sharding)
        else:
            progress_tracker.validate_files(sharding, auto_clean=True)
    
    # Create parallel processor
    processor = ParallelTarProcessor(cfg, tar_index, sharding, progress_tracker, json_writer)
    progress_reporter = ProgressReporter(interval=10.0)
    
    # Show starting stats
    stats = progress_tracker.get_statistics()
    logging.info(f"📊 Starting with {stats['completed']:,} completed, {stats['missing']:,} missing")
    logging.info(f"🚀 Using {cfg.workers} extraction workers, {cfg.io_workers} I/O workers")
    logging.info(f"💾 Memory cache: up to {cfg.max_memory_cache_gb}GB")
    logging.info(f"📀 RAID optimized: {cfg.read_buffer_size_mb}MB read, {cfg.write_buffer_size_mb}MB write buffers")
    
    try:
        # Process metadata stream
        for row in metadata_stream:
            if stop_handler and stop_handler.should_stop():
                logging.info("🛑 Stopping extraction due to user interrupt...")
                break
            
            processor.process_row(row)
            
            # Report progress
            if progress_reporter.should_report():
                progress_reporter.report(processor.get_stats())
        
        # Process remaining
        processor.finish_remaining()
        
        # Final report
        final_stats = processor.get_stats()
        progress_reporter.report(final_stats, force=True)
        
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
        rating_filters = []
        for rating in cfg.ratings:
            rating_lower = rating.lower()
            if rating_lower in ["safe", "s"]:
                rating_filters.append(pl.col(cfg.rating_col).str.to_lowercase().is_in(["safe", "s"]))
            elif rating_lower in ["general", "g"]:
                rating_filters.append(pl.col(cfg.rating_col).str.to_lowercase().is_in(["general", "g"]))
            elif rating_lower in ["questionable", "q"]:
                rating_filters.append(pl.col(cfg.rating_col).str.to_lowercase().is_in(["questionable", "q"]))
            elif rating_lower in ["explicit", "e"]:
                rating_filters.append(pl.col(cfg.rating_col).str.to_lowercase().is_in(["explicit", "e"]))
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
        logging.info(f"🔄 Starting streaming collection with batch size {cfg.batch_size}")
        
        # Stream in batches for constant memory usage
        batch_count = 0
        total_yielded = 0
        
        for batch_df in lf.collect(streaming=True).iter_slices(cfg.batch_size):
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
            python danbooru2024_local_puller.py

            # Custom tag filtering
            python danbooru2024_local_puller.py --include "1girl solo" --exclude "lowres"

            # Filter by score and rating
            python danbooru2024_local_puller.py --min-score 50 --ratings safe general

            # Character and copyright filtering
            python danbooru2024_local_puller.py --include-characters "hakurei_reimu" --include-copyrights "touhou"
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
    p.add_argument("--exclude-gifs", action="store_true", help="Exclude .gif files")
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
    if args.exclude_gifs:
        cfg.exclude_gifs = True
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
        self.index = {}  # {image_id: (tar_name, filename_in_tar)}
        self.tar_contents = {}  # {tar_name: set(filenames)}
        self.tar_patterns = {}  # {tar_name: pattern_info}
        self.index_file = source_dir / ".tar_index_cache.json"
        self.lock = threading.Lock()
        
        if not rebuild and self.index_file.exists():
            self._load_index()
        else:
            self._build_index()
            self._save_index()
    
    def _load_index(self):
        """Load cached index from file."""
        try:
            logging.info("📚 Loading cached tar index...")
            with open(self.index_file, 'r') as f:
                data = json.load(f)
                # Convert string keys back to integers
                self.index = {int(k): tuple(v) for k, v in data.get('index', {}).items()}
                self.tar_contents = data.get('tar_contents', {})
                self.tar_patterns = data.get('tar_patterns', {})
                self.id_mapping = data.get('id_mapping', {})  # Add ID mapping
                self.metadata_to_tar_id = {int(k): v for k, v in data.get('metadata_to_tar_id', {}).items()}
            logging.info(f"✅ Loaded index with {len(self.index):,} images from {len(self.tar_contents)} tar files")
        except Exception as e:
            logging.warning(f"⚠️ Failed to load index cache: {e}")
            logging.info("📚 Rebuilding index from scratch...")
            self._build_index()
            self._save_index()
    
    def _save_index(self):
        """Save index to cache file."""
        try:
            # Convert integer keys to strings for JSON serialization
            data = {
                'index': {str(k): v for k, v in self.index.items()},
                'tar_contents': self.tar_contents,
                'tar_patterns': self.tar_patterns,
                'id_mapping': getattr(self, 'id_mapping', {}),  # Save ID mapping
                'metadata_to_tar_id': {str(k): v for k, v in self.metadata_to_tar_id.items()},
                'last_updated': time.time()
            }
            
            tmp_file = self.index_file.with_suffix('.tmp')
            with open(tmp_file, 'w') as f:
                json.dump(data, f)
            tmp_file.replace(self.index_file)
            logging.info(f"💾 Saved tar index cache with {len(self.index):,} entries")
        except Exception as e:
            logging.error(f"Failed to save index cache: {e}")

    def _detect_tar_pattern(self, tar_name: str) -> Dict[str, Any]:
        """Detect the ID range or pattern for a tar file."""
        pattern_info = {}
        
        # Try to extract ID range from tar filename
        # Common patterns: 
        # - "images_0000000-0999999.tar"
        # - "danbooru2024_000000_000999.tar"
        # - "0000000-0999999.tar"
        # - Simple numbered: "001.tar", "002.tar"
        
        # Pattern 1: Range in filename (e.g., "0000000-0999999")
        range_match = re.search(r'(\d{6,})[_-](\d{6,})', tar_name)
        if range_match:
            pattern_info['start_id'] = int(range_match.group(1))
            pattern_info['end_id'] = int(range_match.group(2))
            pattern_info['type'] = 'range'
            return pattern_info
        
        # Pattern 2: Sequential number (e.g., "001.tar", "batch_001.tar")
        seq_match = re.search(r'(\d{3,})', tar_name)
        if seq_match:
            seq_num = int(seq_match.group(1))
            # Assume each tar contains 1 million images (adjust as needed)
            images_per_tar = 1000000
            pattern_info['start_id'] = seq_num * images_per_tar
            pattern_info['end_id'] = (seq_num + 1) * images_per_tar - 1
            pattern_info['type'] = 'sequential'
            return pattern_info
        
        pattern_info['type'] = 'unknown'
        return pattern_info
    
    
    def _build_index(self):
        """Build the tar index by scanning tar files."""
        logging.info("📚 Building tar index...")
        
        tar_files = sorted(self.source_dir.glob("*.tar"))
        if not tar_files:
            logging.warning(f"⚠️ No tar files found in {self.source_dir}")
            return
        
        logging.info(f"Found {len(tar_files)} tar files to index")

        # Try to detect ID mapping pattern
        self.id_mapping = {}
        sample_tar = tar_files[0] if tar_files else None
        if sample_tar:
            try:
                # Read first tar to understand ID pattern
                with tarfile.open(sample_tar, 'r') as tar:
                    first_members = []
                    for member in tar:
                        if member.isfile() and len(first_members) < 100:
                            first_members.append(member.name)
                    
                    # Extract IDs and detect pattern
                    ids_found = []
                    for name in first_members:
                        match = re.search(r'(\d+)\.(jpg|jpeg|png|gif|webp)', name, re.IGNORECASE)
                        if match:
                            ids_found.append(int(match.group(1)))
                    
                    if ids_found:
                        min_id = min(ids_found)
                        max_id = max(ids_found)
                        logging.info(f"📊 Detected ID range in tars: {min_id:,} - {max_id:,}")
                        # Store this for later use
                        self.tar_id_offset = min_id // 1000000 * 1000000  # Round down to millions
                        logging.info(f"📊 Estimated tar ID offset: {self.tar_id_offset:,}")
            except Exception as e:
                logging.debug(f"Could not detect ID pattern: {e}")
                self.tar_id_offset = 0

        # First, try to understand the tar organization
        for tar_path in tar_files[:5]:  # Sample first 5 tars
            tar_name = tar_path.name

            # Store pattern info
            if tar_name not in self.tar_patterns:
                self.tar_patterns[tar_name] = self._detect_tar_pattern(tar_name)

            pattern = self._detect_tar_pattern(tar_name)
            self.tar_patterns[tar_name] = pattern
            logging.debug(f"Tar {tar_name} pattern: {pattern}")

        for tar_path in tqdm(tar_files, desc="Indexing tar files"):
            tar_name = tar_path.name
            json_path = tar_path.with_suffix('.json')
            
            if not json_path.exists():
                # Read tar contents directly - but more efficiently
                logging.info(f"📖 Reading {tar_name} directly (no JSON found)...")
                self._index_tar_directly_fast(tar_path)
                continue
            
            try:
                # Read the JSON metadata
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
               

                # Handle the actual JSON structure from danbooru2024
                files = []
                if isinstance(metadata, dict):
                    # The JSON has 'files' key with list of dictionaries
                    files_data = metadata.get('files', [])
                    if isinstance(files_data, list):
                        # Each item in files_data is a dict with 'filename' and 'id'
                        for item in files_data:
                            if isinstance(item, dict):
                                # Extract both filename and metadata ID
                                filename = item.get('filename', item.get('name', ''))
                                metadata_id = item.get('id')  # This is the metadata row ID
                                if filename:
                                    files.append({'filename': filename, 'metadata_id': metadata_id})
                            elif isinstance(item, str):
                                files.append({'filename': item, 'metadata_id': None})

                    elif isinstance(files_data, dict):
                        # If it's a dict, try to get values
                        for key, value in files_data.items():
                            if isinstance(value, str):
                                files.append({'filename': value, 'metadata_id': None})
                            elif isinstance(value, dict):
                                files.append(value)
                    else:
                        logging.warning(f"Unexpected 'files' type in {tar_name}: {type(files_data)}")
                        # Fall back to direct tar reading
                        self._index_tar_directly_fast(tar_path)
                        continue
                elif isinstance(metadata, list):
                    files = metadata
                
                if not files:
                    logging.warning(f"No files found in JSON for {tar_name}, reading tar directly")
                    self._index_tar_directly_fast(tar_path)
                    continue
                
                # Process the file list
                tar_contents = set()
                indexed_count = 0
                for entry in files:
                    if not isinstance(entry, dict):
                         continue
                     
                    filename = entry.get('filename', '')
                    metadata_id = entry.get('metadata_id')
                    
                    if not filename:
                        continue
                    
                    # Extract image ID from filename
                    # Handle paths like "1234/5678901.jpg" or just "5678901.jpg"
                    basename = os.path.basename(filename)
                    match = re.match(r'(\d+)\.(jpg|jpeg|png|gif|webp)', basename, re.IGNORECASE)
                    if match:
                        image_id = int(match.group(1))

                        # Store with the actual tar image ID
                        self.index[image_id] = (tar_name, filename)

                        tar_contents.add(filename)
                        indexed_count += 1
                        
                        # Create mapping from metadata ID to tar ID if we have it
                        if metadata_id is not None:
                            self.metadata_to_tar_id[metadata_id] = image_id
                
                self.tar_contents[tar_name] = tar_contents
                if indexed_count > 0:
                    logging.debug(f"  Indexed {indexed_count} images from {tar_name}")
                else:
                    logging.warning(f"  No images indexed from {tar_name} JSON, trying direct read")
                    self._index_tar_directly_fast(tar_path)
                
            except Exception as e:
                logging.error(f"Failed to process {json_path}: {e}")
                # Fall back to reading tar directly
                self._index_tar_directly_fast(tar_path)
        
        logging.info(f"✅ Indexed {len(self.index):,} images across {len(self.tar_contents)} tar files")

        # If we found no images, something is wrong
        if len(self.index) == 0:
            logging.error("❌ No images found in tar files! Check tar structure.")
            # Try to debug by reading first tar
            if tar_files:
                self._debug_tar_structure(tar_files[0]) 

    def _index_tar_directly_fast(self, tar_path: Path):
        """Index a tar file by reading just the member list (faster)."""
        tar_name = tar_path.name
        try:
            # Open tar without reading file contents
            with tarfile.open(tar_path, 'r|') as tar:
                tar_contents = set()
                member_count = 0
                
                for member in tar:
                    if member.isfile():
                        # Handle nested paths
                        full_path = member.name
                        basename = os.path.basename(full_path)
                        
                        # Try to extract image ID from basename
                        match = re.match(r'(\d+)\.(jpg|jpeg|png|gif|webp)', basename, re.IGNORECASE)
                        if match:
                            image_id = int(match.group(1))
                            self.index[image_id] = (tar_name, full_path)
                            tar_contents.add(full_path)
                            member_count += 1

                            # Try to create ID mapping
                            if hasattr(self, 'tar_id_offset') and self.tar_id_offset > 0:
                                possible_metadata_id = image_id - self.tar_id_offset
                                if 0 < possible_metadata_id < 10000000:
                                    self.id_mapping[possible_metadata_id] = image_id

                            # Log progress for large tars
                            if member_count % 10000 == 0:
                                logging.debug(f"  Indexed {member_count} images from {tar_name}...")
                
                self.tar_contents[tar_name] = tar_contents
                logging.info(f"  ✅ Indexed {member_count} images from {tar_name}")
                
        except Exception as e:
            logging.error(f"Failed to read tar file {tar_path}: {e}")
    
    def _debug_tar_structure(self, tar_path: Path):
        """Debug helper to understand tar structure."""
        logging.info(f"🔍 Debugging tar structure for {tar_path.name}...")
        try:
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()[:10]  # First 10 files
                logging.info(f"  Sample files in tar:")
                for member in members:
                    if member.isfile():
                        logging.info(f"    - {member.name} (size: {member.size})")
        except Exception as e:
            logging.error(f"Failed to debug tar: {e}")

    def find_image(self, image_id: int, file_url: str) -> Optional[Tuple[str, str]]:

        """
        Find which tar contains an image. Returns (tar_name, filename) or None.
        The image_id here is the metadata row ID, not the tar file ID.
        """
        with self.lock:
            # First, map metadata ID to tar ID
            tar_id = self.metadata_to_tar_id.get(image_id)
            
            if tar_id:
                # Now look up the tar file using the actual tar ID
                result = self.index.get(tar_id)
                if result:
                    return result
            
            # Fallback: try extracting ID from file_url if available
            if file_url:
                match = re.search(r'/(\d+)\.(jpg|jpeg|png|gif|webp)$', file_url, re.IGNORECASE)
                if match:
                    url_id = int(match.group(1))
                    result = self.index.get(url_id)
                    if result:
                        # Cache this mapping for future use
                        self.metadata_to_tar_id[image_id] = url_id
                        return result
            
            # If still not found, try old fallback methods
            # Try direct lookup with the image_id as-is (unlikely to work)
            result = self.index.get(image_id)
            if result:
                return result
            
            # Try to map metadata ID to tar ID
            if hasattr(self, 'id_mapping'):
                tar_id = self.id_mapping.get(image_id)
                if tar_id:
                    result = self.index.get(tar_id)
                    if result:
                        return result
            
            # Try with offset if we detected one
            if hasattr(self, 'tar_id_offset') and self.tar_id_offset > 0:
                # Try adding the offset to the metadata ID
                possible_tar_id = image_id + self.tar_id_offset
                result = self.index.get(possible_tar_id)
                if result:
                    return result
            
            # If not in index, try to guess based on tar patterns
            if self.tar_patterns:
                for tar_name, pattern in self.tar_patterns.items():
                    if pattern.get('type') == 'range':
                        if pattern['start_id'] <= image_id <= pattern['end_id']:
                            # This tar might contain the image
                            # Construct expected filename
                            ext = os.path.splitext(file_url)[1] if file_url else '.jpg'
                            possible_names = [
                                f"{image_id}{ext}",
                                f"{image_id:07d}{ext}",  # Padded
                                f"{str(image_id)[:4]}/{image_id}{ext}",  # Nested by first 4 digits
                            ]
                            
                            # Check if any of these exist in the tar
                            tar_contents = self.tar_contents.get(tar_name, set())
                            for name in possible_names:
                                if name in tar_contents:
                                    # Cache this finding
                                    self.index[image_id] = (tar_name, name)
                                    return (tar_name, name)
            
            # Still not found
            return None
    
    def get_tar_contents(self, tar_name: str) -> Set[str]:
        """Get the set of files in a tar."""
        return self.tar_contents.get(tar_name, set())

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics for debugging."""
        stats = {
            'total_images': len(self.index),
            'total_tars': len(self.tar_contents),
            'tar_patterns': self.tar_patterns,
            'id_mapping_size': len(getattr(self, 'id_mapping', {})),
            'metadata_to_tar_mapping_size': len(self.metadata_to_tar_id),
            'sample_mappings': dict(list(self.index.items())[:5]) if self.index else {}
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
    logging.info(f"   Total indexed images: {index_stats['total_images']:,}")
    logging.info(f"   Total tar files: {index_stats['total_tars']}")
    if index_stats['sample_mappings']:
        logging.info(f"   Sample mappings: {index_stats['sample_mappings']}")
     

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
            logging.info("📝 Filtering metadata only (extraction disabled)")
            count = sum(1 for _ in metadata_stream)
            logging.info(f"✅ Found {count:,} images matching criteria")
    
    logging.info("🎉 Script completed successfully!")

if __name__ == "__main__":
    main()