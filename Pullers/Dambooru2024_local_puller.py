 #!/usr/bin/env python3
"""
Local Danbooru dataset filtering and extraction script with:
- Tar file extraction from local dataset
- File validation on startup to detect missing files
- Progress tracking and resumability
- Directory sharding for O(1) lookups
- Memory-efficient streaming with tar batching
- Graceful interruption handling
- Batched JSON writing
- Optimized for HDD I/O bottleneck with high RAM availability
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Set, Tuple
from tempfile import TemporaryDirectory

import polars as pl
import pandas as pd
from tqdm import tqdm

# Global extraction lock for thread safety
EXTRACTION_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger("danbooru_local_search")

# ---------------------------------------------------------------------------
# Tar Index Manager
# ---------------------------------------------------------------------------
class TarIndexManager:
    """Manages index of images across tar files for efficient lookup."""
    
    def __init__(self, images_dir: Path, cache_file: Path = None):
        """Initialize the tar index manager."""
        self.images_dir = images_dir
        self.cache_file = cache_file or images_dir / ".tar_index_cache.json"
        self.image_to_tar: Dict[str, str] = {}  # Maps "id.ext" to tar filename
        self.tar_to_images: Dict[str, Set[str]] = defaultdict(set)  # Maps tar to set of images
        self.loaded = False
    
    def build_index(self, force_rebuild: bool = False) -> None:
        """Build or load the tar file index."""
        if not force_rebuild and self.cache_file.exists():
            try:
                self._load_cache()
                log.info(f"✅ Loaded tar index from cache: {len(self.image_to_tar):,} images indexed")
                self.loaded = True
                return
            except Exception as e:
                log.warning(f"⚠️ Failed to load cache, rebuilding: {e}")
        
        log.info("🔍 Building tar file index (this may take a few minutes)...")
        self.image_to_tar.clear()
        self.tar_to_images.clear()
        
        # Find all tar files and their associated JSON files
        tar_files = sorted(self.images_dir.glob("*.tar"))
        
        if not tar_files:
            log.error(f"❌ No tar files found in {self.images_dir}")
            return
        
        log.info(f"📦 Found {len(tar_files)} tar files to index")
        
        with tqdm(total=len(tar_files), desc="Indexing tar files", unit="tar") as pbar:
            for tar_path in tar_files:
                # Look for associated JSON file
                json_path = tar_path.with_suffix('.json')
                
                if json_path.exists():
                    # Use JSON for fast indexing
                    self._index_from_json(tar_path.name, json_path)
                else:
                    # Fall back to reading tar file directly
                    self._index_from_tar(tar_path)
                
                pbar.update(1)
        
        # Save cache
        self._save_cache()
        log.info(f"✅ Indexed {len(self.image_to_tar):,} images across {len(self.tar_to_images)} tar files")
        self.loaded = True
    
    def _index_from_json(self, tar_name: str, json_path: Path) -> None:
        """Index a tar file using its associated JSON."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
                # Handle both {"files": {...}} and direct {...} formats
                if "files" in data:
                    files = data["files"]
                else:
                    files = data
                
                for filename in files.keys():
                    # Store both with and without extension for flexibility
                    self.image_to_tar[filename] = tar_name
                    self.tar_to_images[tar_name].add(filename)
                    
                    # Also store just the ID for easier lookup
                    image_id = filename.split('.')[0]
                    if image_id.isdigit():
                        self.image_to_tar[image_id] = tar_name
                        
        except Exception as e:
            log.warning(f"⚠️ Failed to index {json_path}: {e}")
    
    def _index_from_tar(self, tar_path: Path) -> None:
        """Index a tar file by reading its contents directly."""
        try:
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        filename = os.path.basename(member.name)
                        self.image_to_tar[filename] = tar_path.name
                        self.tar_to_images[tar_path.name].add(filename)
                        
                        # Also store just the ID
                        image_id = filename.split('.')[0]
                        if image_id.isdigit():
                            self.image_to_tar[image_id] = tar_path.name
        except Exception as e:
            log.warning(f"⚠️ Failed to index {tar_path}: {e}")
    
    def _save_cache(self) -> None:
        """Save the index to a cache file."""
        try:
            cache_data = {
                "image_to_tar": self.image_to_tar,
                "tar_to_images": {k: list(v) for k, v in self.tar_to_images.items()},
                "timestamp": time.time()
            }
            
            tmp_path = self.cache_file.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(cache_data, f)
            tmp_path.replace(self.cache_file)
            
            log.info(f"💾 Saved tar index cache to {self.cache_file}")
        except Exception as e:
            log.warning(f"⚠️ Failed to save cache: {e}")
    
    def _load_cache(self) -> None:
        """Load the index from cache file."""
        with open(self.cache_file, 'r') as f:
            cache_data = json.load(f)
            
        self.image_to_tar = cache_data["image_to_tar"]
        self.tar_to_images = {k: set(v) for k, v in cache_data["tar_to_images"].items()}
    
    def find_image(self, image_id: int, file_url: str = None) -> Optional[Tuple[str, str]]:
        """
        Find which tar file contains an image.
        Returns (tar_filename, image_filename) or None.
        """
        # Try with extension from file_url if provided
        if file_url:
            ext = os.path.splitext(file_url)[1]
            filename_with_ext = f"{image_id}{ext}"
            if filename_with_ext in self.image_to_tar:
                return (self.image_to_tar[filename_with_ext], filename_with_ext)
        
        # Try just the ID (will match any extension)
        str_id = str(image_id)
        if str_id in self.image_to_tar:
            tar_name = self.image_to_tar[str_id]
            # Find the actual filename in the tar
            for filename in self.tar_to_images[tar_name]:
                if filename.startswith(str_id + "."):
                    return (tar_name, filename)
        
        # Try common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            filename = f"{image_id}{ext}"
            if filename in self.image_to_tar:
                return (self.image_to_tar[filename], filename)
        
        return None

# ---------------------------------------------------------------------------
# Directory Sharding (unchanged from original)
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
            log.error(f"Failed to create shard directory {shard_path}: {e}")
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
    
    def get_all_existing_files(self) -> Set[int]:
        """Scan all shard directories and return set of existing file IDs."""
        existing_ids = set()
        if not self.base_dir.exists():
            return existing_ids
        
        # Pattern to match image files
        image_pattern = re.compile(r'^(\d+)\.(jpg|jpeg|png|gif|webp)$', re.IGNORECASE)
        
        # Scan all shard directories
        for shard_dir in self.base_dir.glob("shard_*"):
            if shard_dir.is_dir():
                for file_path in shard_dir.iterdir():
                    if file_path.is_file():
                        match = image_pattern.match(file_path.name)
                        if match:
                            image_id = int(match.group(1))
                            existing_ids.add(image_id)
        
        return existing_ids

# ---------------------------------------------------------------------------
# Enhanced Progress Tracking with Validation (unchanged from original)
# ---------------------------------------------------------------------------
class ValidatingProgressTracker:
    """Tracks download progress with validation and persistent records."""
    
    def __init__(self, progress_file: Path, update_interval: int = 100):
        self.progress_file = progress_file
        self.update_interval = update_interval
        self.completed_ids: Set[int] = set()
        self.missing_ids: Set[int] = set()  # Track files that went missing
        self.update_counter = 0
        self.lock = threading.Lock()
        self.validation_performed = False
        self._load_progress()
    
    def _load_progress(self):
        """Load existing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_ids = set(data.get('completed_ids', []))
                    self.missing_ids = set(data.get('missing_ids', []))
                log.info(f"📈 Loaded progress: {len(self.completed_ids):,} completed extractions")
                if self.missing_ids:
                    log.warning(f"⚠️ Found {len(self.missing_ids):,} previously missing files in records")
            except Exception as e:
                log.warning(f"⚠️ Failed to load progress file: {e}")
    
    def validate_files(self, sharding: DirectorySharding, auto_clean: bool = True) -> Dict[str, Any]:
        """
        Validate that all "completed" files actually exist on disk.
        Returns statistics about the validation.
        """
        log.info("🔍 Starting file validation...")
        
        # Create a copy to iterate over safely
        with self.lock:
            ids_to_check = self.completed_ids.copy()
        
        log.info(f"   Checking {len(ids_to_check):,} files marked as completed...")
        
        stats = {
            'total_marked_complete': len(ids_to_check),
            'files_verified': 0,
            'files_missing': 0,
            'files_recovered': 0,
            'validation_time': 0
        }
        
        start_time = time.time()
        
        # Check each completed ID
        missing_now = set()
        verified = set()
        recovered = set()
        
        with tqdm(total=len(ids_to_check), desc="Validating files", unit="files") as pbar:
            for image_id in ids_to_check:
                exists, ext = sharding.file_exists(image_id)
                if exists:
                    verified.add(image_id)
                    if image_id in self.missing_ids:
                        recovered.add(image_id)
                else:
                    missing_now.add(image_id)
                pbar.update(1)
        
        stats['files_verified'] = len(verified)
        stats['files_missing'] = len(missing_now)
        stats['files_recovered'] = len(recovered)
        stats['validation_time'] = time.time() - start_time
        
        # Report findings
        log.info(f"✅ Validation complete in {stats['validation_time']:.1f}s:")
        log.info(f"   • Files verified: {stats['files_verified']:,}")
        log.info(f"   • Files missing: {stats['files_missing']:,}")
        if stats['files_recovered'] > 0:
            log.info(f"   • Files recovered: {stats['files_recovered']:,}")
        
        # Update internal state with proper locking
        if missing_now and auto_clean:
            log.warning(f"⚠️ Removing {len(missing_now):,} missing files from completed list...")
            with self.lock:
                self.completed_ids -= missing_now
                self.missing_ids -= recovered
                self.missing_ids.update(missing_now)
                self._save_progress()
            log.info(f"✅ Progress file cleaned. Ready to re-extract missing files.")
        elif missing_now:
            log.warning(f"⚠️ Found {len(missing_now):,} missing files but auto_clean is disabled")
        elif recovered:
            with self.lock:
                self.missing_ids -= recovered
                self._save_progress()
            log.info(f"✅ Updated progress file to reflect {len(recovered):,} recovered files")
        
        self.validation_performed = True
        return stats
    
    def full_filesystem_scan(self, sharding: DirectorySharding) -> Dict[str, Any]:
        """
        Perform a complete filesystem scan to find all image files,
        then reconcile with progress tracker.
        """
        log.info("🔍 Performing full filesystem scan...")
        
        stats = {
            'files_on_disk': 0,
            'files_in_tracker': len(self.completed_ids),
            'untracked_files': 0,
            'missing_from_disk': 0,
            'scan_time': 0
        }
        
        start_time = time.time()
        
        # Get all files from filesystem
        log.info("   Scanning all shard directories...")
        files_on_disk = sharding.get_all_existing_files()
        
        stats['files_on_disk'] = len(files_on_disk)
        
        # Find discrepancies
        untracked = files_on_disk - self.completed_ids
        missing = self.completed_ids - files_on_disk
        
        stats['untracked_files'] = len(untracked)
        stats['missing_from_disk'] = len(missing)
        stats['scan_time'] = time.time() - start_time
        
        # Report findings
        log.info(f"✅ Filesystem scan complete in {stats['scan_time']:.1f}s:")
        log.info(f"   • Total files on disk: {stats['files_on_disk']:,}")
        log.info(f"   • Files in tracker: {stats['files_in_tracker']:,}")
        log.info(f"   • Untracked files found: {stats['untracked_files']:,}")
        log.info(f"   • Missing from disk: {stats['missing_from_disk']:,}")
        
        # Optionally fix discrepancies
        if untracked:
            log.info(f"💡 Found {len(untracked):,} files on disk not in progress tracker")
            response = input("   Add these to completed list? (y/n): ").strip().lower()
            if response == 'y':
                with self.lock:
                    self.completed_ids.update(untracked)
                    self._save_progress()
                log.info(f"✅ Added {len(untracked):,} files to progress tracker")
        
        if missing:
            log.warning(f"⚠️ Found {len(missing):,} files in tracker but missing from disk")
            response = input("   Remove these from completed list? (y/n): ").strip().lower()
            if response == 'y':
                with self.lock:
                    self.completed_ids -= missing
                    self.missing_ids.update(missing)
                    self._save_progress()
                log.info(f"✅ Cleaned {len(missing):,} missing entries from progress tracker")
        
        return stats
    
    def mark_completed(self, image_id: int):
        """Mark an image as completed and update progress file atomically."""
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
    
    def get_statistics(self) -> Dict[str, int]:
        """Get current statistics."""
        return {
            'completed': len(self.completed_ids),
            'missing': len(self.missing_ids),
            'total_processed': len(self.completed_ids) + len(self.missing_ids)
        }

# ---------------------------------------------------------------------------
# Batched JSON Writer (unchanged from original)
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
        # Wait for flush thread to complete
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=1.0)

# ---------------------------------------------------------------------------
# Signal Handler (unchanged from original)
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
# Configuration (modified for local dataset)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Holds all runtime parameters (may be overridden from CLI)."""

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

    # ---- Filtering Criteria (same as original) ---------------------------
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
    extract_images: bool = True  # Changed from download_images
    save_filtered_metadata: bool = True
    filtered_metadata_format: str = "json"
    strip_json_details: bool = True
    exclude_gifs: bool = True
    dry_run: bool = False
    validate_on_start: bool = True
    full_scan: bool = False
    rebuild_tar_index: bool = False  # New: force rebuild tar index

    # ---- Performance ------------------------------------------------------
    workers: int = 10
    files_per_shard: int = 5000
    batch_size: int = 5000
    use_streaming: bool = True
    tar_batch_size: int = 4000  # New: how many files to extract from one tar at once

# ---------------------------------------------------------------------------
# CLI (modified for local dataset)
# ---------------------------------------------------------------------------
def _parse_tag_list(value: str) -> List[str]:
    """Parses comma- or space-separated tag strings into a list."""
    return [t for t in re.split(r"[\s,]+", value.strip()) if t]

def build_cli() -> argparse.ArgumentParser:
    """Defines and configures the command-line argument parser."""
    p = argparse.ArgumentParser(
        prog="danbooru-local-search",
        description="Filter Danbooru metadata & extract matching images from local tar files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
            Example Usage:
            --------------
            # Extract with automatic validation on startup
            python local_search.py --metadata /data/danbooru2024/metadata.parquet --source /data/danbooru2024/images/ --include "1girl solo" --min-score 100 --output ./My_Filtered_Images

            # Perform full filesystem scan to find untracked files
            python local_search.py --full-scan --output ./My_Filtered_Images

            # Rebuild tar index cache
            python local_search.py --rebuild-index --include cat_ears --output ./output
            """),
    )

    # Paths / IO
    p.add_argument("--metadata", type=str, help="Path to Danbooru parquet")
    p.add_argument("--source", type=str, help="Directory containing tar files")
    p.add_argument("--output", type=str, help="Destination directory")

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
    p.add_argument("--no-extract", dest="extract", action="store_false", help="Skip extraction")
    p.add_argument("--no-save-metadata", dest="save_meta", action="store_false", help="No metadata")
    p.add_argument("--dry-run", action="store_true", help="Exit after stats")
    p.add_argument("--exclude-gifs", action="store_true", help="Exclude .gif files")
    
    # Validation flags
    p.add_argument("--no-validate", dest="validate", action="store_false", 
                   help="Skip file validation on startup")
    p.add_argument("--full-scan", action="store_true", 
                   help="Perform full filesystem scan to find all files")
    p.add_argument("--rebuild-index", action="store_true",
                   help="Force rebuild of tar file index")
    
    # Performance
    p.add_argument("--workers", type=int, help="Number of extraction workers")
    p.add_argument("--files-per-shard", type=int, help="Files per directory shard")
    p.add_argument("--batch-size", type=int, help="Streaming batch size")
    p.add_argument("--tar-batch-size", type=int, help="Files to extract from one tar at once")
    p.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming")

    return p

def apply_cli_overrides(args: argparse.Namespace, cfg: Config) -> None:
    """Mutates cfg in-place based on CLI arguments."""
    if args.metadata: cfg.metadata_db_path = args.metadata
    if args.source: cfg.source_images_dir = args.source
    if args.output: cfg.output_dir = args.output

    if args.include is not None: 
        cfg.include_tags = args.include
        cfg.enable_include_tags = True
    if args.exclude is not None: 
        cfg.exclude_tags = args.exclude
        cfg.enable_exclude_tags = True  
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
    if hasattr(args, 'tar_batch_size') and args.tar_batch_size:
        cfg.tar_batch_size = args.tar_batch_size
    if hasattr(args, 'streaming') and not args.streaming:
        cfg.use_streaming = False
    
    if hasattr(args, 'extract') and not args.extract:
        cfg.extract_images = False
    if hasattr(args, 'save_meta') and not args.save_meta:
        cfg.save_filtered_metadata = False
    
    if hasattr(args, 'validate') and not args.validate:
        cfg.validate_on_start = False
    if args.full_scan:
        cfg.full_scan = True
    if args.rebuild_index:
        cfg.rebuild_tar_index = True

    if args.dry_run:
        cfg.dry_run = True
        cfg.extract_images = False

    if args.exclude_gifs: 
        cfg.exclude_gifs = True

# ---------------------------------------------------------------------------
# Streaming Metadata (same filtering logic as original)
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
    if cfg.extract_images:
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
# Extraction Functions (new, replacing download)
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

def extract_from_tar_batch(tar_path: Path, files_to_extract: List[Tuple[int, str, Path]], 
                          json_writer: Optional[BatchedJSONWriter], rows_data: Dict[int, Dict],
                          cfg: Config) -> List[int]:
    """
    Extract multiple files from a single tar file.
    Returns list of successfully extracted image IDs.
    """
    successful_ids = []
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Get all members for efficient lookup
            members_dict = {m.name: m for m in tar.getmembers()}
            
            for image_id, filename, dest_path in files_to_extract:
                try:
                    # Find the member (might be in subdirectory)
                    member = None
                    if filename in members_dict:
                        member = members_dict[filename]
                    else:
                        # Try to find in subdirectories
                        for name, m in members_dict.items():
                            if name.endswith(f"/{filename}") or name == filename:
                                member = m
                                break
                    
                    if member:
                        # Extract to temporary location then move
                        with TemporaryDirectory() as tmpdir:
                            tar.extract(member, tmpdir)
                            extracted_path = Path(tmpdir) / member.name
                            
                            # Move to destination
                            shutil.move(str(extracted_path), str(dest_path))
                            
                            # Write JSON if needed
                            if cfg.per_image_json and json_writer and image_id in rows_data:
                                json_path = dest_path.parent / f"{image_id}.json"
                                json_data = prepare_json_data(rows_data[image_id], cfg)
                                json_writer.add_write(json_path, json_data)
                            
                            successful_ids.append(image_id)
                    else:
                        log.warning(f"⚠️ File {filename} not found in {tar_path.name}")
                        
                except Exception as e:
                    log.error(f"❌ Failed to extract {filename}: {e}")
                    
    except Exception as e:
        log.error(f"❌ Failed to open tar file {tar_path}: {e}")
    
    return successful_ids

def process_extractions(metadata_stream: Iterator[Dict[str, Any]], 
                       cfg: Config, dest_dir: Path,
                       tar_index: TarIndexManager,
                       stop_handler: Optional[SoftStopHandler] = None) -> None:
    """
    Process extractions from tar files with batching by tar file.
    """
    # Initialize all systems
    sharding = DirectorySharding(dest_dir, cfg.files_per_shard)
    json_writer = BatchedJSONWriter(flush_interval=5.0, batch_size=100) if cfg.per_image_json else None
    progress_tracker = ValidatingProgressTracker(dest_dir / "progress.json")
    
    # Perform validation if enabled
    if cfg.validate_on_start:
        if cfg.full_scan:
            progress_tracker.full_filesystem_scan(sharding)
        else:
            progress_tracker.validate_files(sharding, auto_clean=True)
    
    # Show current statistics
    stats = progress_tracker.get_statistics()
    log.info(f"📊 Starting with {stats['completed']:,} completed, {stats['missing']:,} missing")
    
    extracted = 0
    failed = 0
    skipped = 0
    not_found = 0
    
    # Group images by tar file for efficient extraction
    tar_batches: Dict[str, List[Tuple[int, str, Path, Dict]]] = defaultdict(list)
    
    try:
        for row in metadata_stream:
            # Check for stop signal
            if stop_handler and stop_handler.should_stop():
                log.info("🛑 Stopping extraction due to user interrupt...")
                break
            
            image_id = row[cfg.id_col]
            file_url = row.get(cfg.file_path_col, "")
            
            # Check if file exists and is tracked
            file_exists, ext = sharding.file_exists(image_id)
            is_completed = progress_tracker.is_completed(image_id)
            
            if file_exists and is_completed:
                skipped += 1
                continue
            elif file_exists and not is_completed:
                progress_tracker.mark_completed(image_id)
                skipped += 1
                continue
            
            # Find the tar file containing this image
            tar_info = tar_index.find_image(image_id, file_url)
            
            if not tar_info:
                log.debug(f"⚠️ Image {image_id} not found in any tar file")
                not_found += 1
                continue
            
            tar_name, filename = tar_info
            shard_path = sharding.get_shard_path(image_id)
            dest_path = shard_path / filename
            
            # Add to batch for this tar file
            tar_batches[tar_name].append((image_id, filename, dest_path, row))
            
            # Process batch if it's large enough
            if len(tar_batches[tar_name]) >= cfg.tar_batch_size:
                tar_path = Path(cfg.source_images_dir) / tar_name
                files_to_extract = [(id, fn, dp) for id, fn, dp, _ in tar_batches[tar_name]]
                rows_data = {id: r for id, _, _, r in tar_batches[tar_name]}
                
                log.info(f"📦 Extracting {len(files_to_extract)} files from {tar_name}")
                successful = extract_from_tar_batch(tar_path, files_to_extract, json_writer, rows_data, cfg)
                
                for id in successful:
                    progress_tracker.mark_completed(id)
                    extracted += 1
                
                failed += len(files_to_extract) - len(successful)
                tar_batches[tar_name].clear()
            
            # Progress reporting
            total_processed = extracted + failed + skipped + not_found
            if total_processed % 100 == 0:
                log.info(f"📥 Progress: {extracted:,} extracted, "
                       f"{failed:,} failed, {skipped:,} skipped, {not_found:,} not found")
        
        # Process remaining batches
        log.info("⏳ Processing remaining tar batches...")
        for tar_name, batch in tar_batches.items():
            if batch:
                tar_path = Path(cfg.source_images_dir) / tar_name
                files_to_extract = [(id, fn, dp) for id, fn, dp, _ in batch]
                rows_data = {id: r for id, _, _, r in batch}
                
                log.info(f"📦 Extracting {len(files_to_extract)} files from {tar_name}")
                successful = extract_from_tar_batch(tar_path, files_to_extract, json_writer, rows_data, cfg)
                
                for id in successful:
                    progress_tracker.mark_completed(id)
                    extracted += 1
                
                failed += len(files_to_extract) - len(successful)
    
    finally:
        # Cleanup
        if json_writer:
            json_writer.close()
        progress_tracker.save_final()
        
        log.info(f"✅ Complete: {extracted:,} extracted, "
               f"{failed:,} failed, {skipped:,} skipped, {not_found:,} not found")

# ---------------------------------------------------------------------------
# Main Execution (modified for local dataset)
# ---------------------------------------------------------------------------
def main() -> None:
    """Main function to orchestrate the filtering and extraction process."""
    # --- Configuration and Argument Parsing ---
    cfg = Config()
    parser = build_cli()
    args = parser.parse_args()
    apply_cli_overrides(args, cfg)

    # --- Setup Paths ---
    meta_path = Path(cfg.metadata_db_path)
    source_dir = Path(cfg.source_images_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build tar index ---
    tar_index = TarIndexManager(source_dir)
    tar_index.build_index(force_rebuild=cfg.rebuild_tar_index)
    
    if not tar_index.loaded:
        log.error("❌ Failed to build tar index. Exiting.")
        sys.exit(1)

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
            
            # Process extractions
            if cfg.extract_images:
                process_extractions(metadata_stream, cfg, out_dir, tar_index, stop_handler)
            elif cfg.save_filtered_metadata:
                log.info("📝 Saving metadata only (extraction disabled)")
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

    log.info("🎉 Script completed successfully!")

if __name__ == "__main__":
    main()
