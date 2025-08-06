#!/usr/bin/env python3
"""
Optimized Danbooru dataset filter and downloader with performance improvements.
Includes directory sharding, batched writing, progress tracking, and graceful shutdown.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import json
import sys
import signal
import textwrap
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Set, Any
from collections import deque
from contextlib import contextmanager
import tempfile
import shutil

import polars as pl
import pandas as pd
from huggingface_hub import HfFolder, login

try:
    from cheesechaser.datapool import DanbooruNewestDataPool as DataPool
except ModuleNotFoundError:
    DataPool = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger("danbooru_puller")

# ---------------------------------------------------------------------------
# Performance Components
# ---------------------------------------------------------------------------

class DirectorySharding:
    """Implements directory sharding for O(1) file lookups."""
    
    def __init__(self, base_dir: Path, files_per_shard: int = 5000):
        self.base_dir = base_dir
        self.files_per_shard = files_per_shard
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_shard_path(self, file_id: int) -> Path:
        """Get the shard directory for a given file ID."""
        shard_num = file_id // self.files_per_shard
        shard_dir = self.base_dir / f"shard_{shard_num:06d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir
    
    def get_file_path(self, file_id: int, extension: str = ".jpg") -> Path:
        """Get the full path for a file with sharding."""
        shard_dir = self.get_shard_path(file_id)
        return shard_dir / f"{file_id}{extension}"
    
    def file_exists(self, file_id: int, extensions: List[str] = [".jpg", ".png", ".gif", ".json"]) -> bool:
        """Check if a file exists with any of the given extensions."""
        shard_dir = self.get_shard_path(file_id)
        return any((shard_dir / f"{file_id}{ext}").exists() for ext in extensions)
    
    def get_existing_ids(self) -> Set[int]:
        """Get all existing file IDs across all shards."""
        existing_ids = set()
        for shard_dir in self.base_dir.glob("shard_*"):
            if shard_dir.is_dir():
                for file_path in shard_dir.iterdir():
                    if file_path.stem.isdigit():
                        existing_ids.add(int(file_path.stem))
        return existing_ids


class BatchedJSONWriter:
    """Thread-safe batched JSON writer with atomic operations."""
    
    def __init__(self, sharding: DirectorySharding, batch_size: int = 100, 
                 flush_interval: float = 5.0):
        self.sharding = sharding
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer = deque()
        self.lock = threading.Lock()
        self.last_flush = time.time()
        self.stopped = False
        self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self.flush_thread.start()
    
    def _periodic_flush(self):
        """Background thread to periodically flush the buffer."""
        while not self.stopped:
            time.sleep(self.flush_interval)
            self.flush()
    
    def write(self, file_id: int, data: Dict[str, Any]):
        """Add a JSON record to the write buffer."""
        with self.lock:
            self.buffer.append((file_id, data))
            if len(self.buffer) >= self.batch_size:
                self._flush_locked()
    
    def _flush_locked(self):
        """Flush buffer to disk (must be called with lock held)."""
        if not self.buffer:
            return
        
        batch = list(self.buffer)
        self.buffer.clear()
        
        # Write each file atomically
        for file_id, data in batch:
            json_path = self.sharding.get_file_path(file_id, ".json")
            temp_path = json_path.with_suffix(".tmp")
            
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                # Atomic rename
                temp_path.replace(json_path)
            except Exception as e:
                log.error(f"Failed to write JSON for ID {file_id}: {e}")
                if temp_path.exists():
                    temp_path.unlink()
    
    def flush(self):
        """Flush any pending writes."""
        with self.lock:
            self._flush_locked()
    
    def close(self):
        """Close the writer and flush remaining data."""
        self.stopped = True
        self.flush()


class ProgressTracker:
    """Tracks download progress and enables efficient resumption."""
    
    def __init__(self, total_items: int, update_interval: int = 100):
        self.total_items = total_items
        self.update_interval = update_interval
        self.processed_items = 0
        self.failed_items = 0
        self.skipped_items = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.last_update = 0
    
    def update(self, processed: int = 0, failed: int = 0, skipped: int = 0):
        """Update progress counters."""
        with self.lock:
            self.processed_items += processed
            self.failed_items += failed
            self.skipped_items += skipped
            
            total_done = self.processed_items + self.failed_items + self.skipped_items
            if total_done - self.last_update >= self.update_interval:
                self._print_progress()
                self.last_update = total_done
    
    def _print_progress(self):
        """Print progress information."""
        elapsed = time.time() - self.start_time
        total_done = self.processed_items + self.failed_items + self.skipped_items
        rate = total_done / elapsed if elapsed > 0 else 0
        remaining = self.total_items - total_done
        eta = remaining / rate if rate > 0 else 0
        
        log.info(f"Progress: {total_done}/{self.total_items} "
                f"({100*total_done/self.total_items:.1f}%) | "
                f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s | "
                f"✓{self.processed_items} ✗{self.failed_items} ⊘{self.skipped_items}")
    
    def finish(self):
        """Print final statistics."""
        elapsed = time.time() - self.start_time
        total_done = self.processed_items + self.failed_items + self.skipped_items
        log.info(f"🎉 Completed {total_done} items in {elapsed:.1f}s")
        log.info(f"   Processed: {self.processed_items}")
        log.info(f"   Failed: {self.failed_items}")
        log.info(f"   Skipped: {self.skipped_items}")


class SoftStopHandler:
    """Two-stage graceful shutdown handler for Ctrl+C."""
    
    def __init__(self):
        self.stop_requested = False
        self.force_stop = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle interrupt signals."""
        if not self.stop_requested:
            self.stop_requested = True
            log.warning("\n⚠️  Graceful shutdown initiated. Press Ctrl+C again to force stop.")
        else:
            self.force_stop = True
            log.error("\n❌ Force stop requested. Exiting immediately.")
            sys.exit(1)
    
    def should_stop(self) -> bool:
        """Check if stop has been requested."""
        return self.stop_requested


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Configuration with performance optimizations."""
    
    # Paths
    metadata_db_path: str = r"/media/andrewk/qnap-public/New file/Danbooru2004/metadata.parquet"
    output_dir: str = r"/mnt/raid0/DAb/"
    
    # Hugging Face
    dataset_repo: str = "deepghs/danbooru2024"
    hf_auth_token: Optional[str] = os.getenv("HF_TOKEN", None)
    
    # Column names
    tags_col: str = "tags"
    character_tags_col: str = "tag_string_character"
    copyright_tags_col: str = "tag_string_copyright"
    artist_tags_col: str = "tag_string_artist"
    score_col: str = "score"
    rating_col: str = "rating"
    width_col: str = "width"
    height_col: str = "height"
    file_path_col: str = "file_url"
    id_col: str = "id"
    
    # Filtering toggles
    enable_include_tags: bool = False
    enable_exclude_tags: bool = False
    enable_character_filtering: bool = False
    enable_copyright_filtering: bool = False
    enable_artist_filtering: bool = False
    enable_score_filtering: bool = False
    enable_rating_filtering: bool = False
    enable_dimension_filtering: bool = False
    per_image_json: bool = True
    
    # Filtering criteria
    include_tags: List[str] = field(default_factory=lambda: ["*eye"])
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
        "ai generated", "ai art", "ai_generated", "ai_artwork", "ai artifact",
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
    
    # Behavior flags
    download_images: bool = True
    save_filtered_metadata: bool = True
    filtered_metadata_format: str = "json"
    strip_json_details: bool = True
    exclude_gifs: bool = True
    dry_run: bool = False
    
    # Performance settings
    workers: int = 15
    files_per_shard: int = 5000
    json_batch_size: int = 100
    json_flush_interval: float = 5.0
    progress_update_interval: int = 100
    batch_size: int = 1000
    max_outstanding_multiplier: int = 10


# ---------------------------------------------------------------------------
# Metadata loading with streaming
# ---------------------------------------------------------------------------
def load_and_filter_metadata(cfg: Config) -> pl.LazyFrame:
    """Load metadata using Polars lazy evaluation for memory efficiency."""
    path = Path(cfg.metadata_db_path)
    
    if not path.exists():
        log.error(f"❌ Metadata path not found: {path}")
        sys.exit(1)
    
    log.info(f"📖 Loading metadata with Polars lazy evaluation...")
    
    # Start with lazy frame
    lf = pl.scan_parquet(str(path))
    
    # Build filter expressions
    filters = []
    
    # File type filtering
    if cfg.exclude_gifs:
        filters.append(~pl.col(cfg.file_path_col).str.ends_with(".gif"))
    
    excluded_extensions = ('.zip', '.mp4', '.webm', '.swf')
    for ext in excluded_extensions:
        filters.append(~pl.col(cfg.file_path_col).str.ends_with(ext))
    
    # Tag filtering with Polars expressions
    if cfg.enable_include_tags and cfg.include_tags:
        for tag in cfg.include_tags:
            pattern = build_tag_pattern(tag)
            filters.append(pl.col(cfg.tags_col).str.contains(pattern))
    
    if cfg.enable_exclude_tags and cfg.exclude_tags:
        patterns = [build_tag_pattern(tag) for tag in cfg.exclude_tags]
        combined_pattern = "|".join(patterns)
        filters.append(~pl.col(cfg.tags_col).str.contains(combined_pattern))
    
    # Score filtering
    if cfg.enable_score_filtering and cfg.min_score is not None:
        filters.append(pl.col(cfg.score_col) >= cfg.min_score)
    
    # Rating filtering
    if cfg.enable_rating_filtering and cfg.ratings:
        filters.append(pl.col(cfg.rating_col).is_in(cfg.ratings))
    
    # Dimension filtering
    if cfg.enable_dimension_filtering:
        if cfg.square_only:
            filters.append(pl.col(cfg.width_col) == pl.col(cfg.height_col))
            filters.append(pl.col(cfg.width_col) >= cfg.min_square_size)
        else:
            if cfg.min_width > 0:
                filters.append(pl.col(cfg.width_col) >= cfg.min_width)
            if cfg.min_height > 0:
                filters.append(pl.col(cfg.height_col) >= cfg.min_height)
            if cfg.max_width > 0:
                filters.append(pl.col(cfg.width_col) <= cfg.max_width)
            if cfg.max_height > 0:
                filters.append(pl.col(cfg.height_col) <= cfg.max_height)
    
    # Apply all filters
    if filters:
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter = combined_filter & f
        lf = lf.filter(combined_filter)
    
    return lf


def build_tag_pattern(tag: str) -> str:
    """Build regex pattern for tag matching."""
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


def stream_filtered_metadata(lf: pl.LazyFrame, cfg: Config, batch_size: int = 1000):
    """Stream filtered metadata in batches for memory efficiency."""
    # Convert to streaming iterator
    df_iter = lf.collect(streaming=True).iter_slices(batch_size)
    
    for batch_df in df_iter:
        # Convert Polars DataFrame to pandas for compatibility
        yield batch_df.to_pandas()


def normalize_rating(rating: str) -> Optional[str]:
    """Normalize rating values according to the mapping."""
    rating_map = {'g': 'safe', 's': None, 'q': 'questionable', 'e': 'explicit'}
    return rating_map.get(rating, rating)


def process_metadata_record(row: pd.Series, cfg: Config) -> Dict[str, Any]:
    """Process a single metadata record with all transformations."""
    record = row.dropna().to_dict()
    
    # Remove file path column
    if cfg.file_path_col in record:
        del record[cfg.file_path_col]
    
    # Strip details if requested
    if cfg.strip_json_details:
        for key in [cfg.score_col, cfg.width_col, cfg.height_col]:
            if key in record:
                del record[key]
    
    # Rating transformation
    if 'rating' in record:
        normalized = normalize_rating(record['rating'])
        if normalized is None:
            del record['rating']
        else:
            record['rating'] = normalized
    
    # Merge tag fields
    tag_source_fields = [
        'tag_string_general',
        'tag_string_character',
        'tag_string_copyright',
        'tag_string_artist',
        'tag_string_meta'
    ]
    
    tag_parts = []
    for field in tag_source_fields:
        if field in record:
            value = str(record.get(field, '')).strip()
            if value:
                tag_parts.append(value)
    
    if tag_parts:
        merged_tags = ' '.join(tag_parts).lower()
        merged_tags = ' '.join(merged_tags.split())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in merged_tags.split():
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        record['tags'] = ' '.join(unique_tags)
    
    # Clean up original tag columns
    for field in tag_source_fields:
        if field in record:
            del record[field]
    
    return record


# ---------------------------------------------------------------------------
# Download with optimizations
# ---------------------------------------------------------------------------
def download_with_optimizations(df: pd.DataFrame, cfg: Config, sharding: DirectorySharding,
                               json_writer: BatchedJSONWriter, stop_handler: SoftStopHandler):
    """Download images with all optimizations enabled."""
    if DataPool is None:
        log.error("CheeseChaser is not installed. Install with 'pip install cheesechaser'.")
        return
    
    # Setup progress tracking
    progress = ProgressTracker(len(df), cfg.progress_update_interval)
    
    # Filter out existing files
    existing_ids = sharding.get_existing_ids()
    df_to_download = df[~df[cfg.id_col].isin(existing_ids)]
    
    if len(df_to_download) == 0:
        log.info("All files already exist. Nothing to download.")
        return
    
    log.info(f"Starting optimized download of {len(df_to_download)} images...")
    
    # Process in batches for better memory usage
    batch_size = cfg.batch_size
    pool = DataPool()
    
    for i in range(0, len(df_to_download), batch_size):
        if stop_handler.should_stop():
            log.warning("Stopping download due to user request...")
            break
        
        batch = df_to_download.iloc[i:i+batch_size]
        ids = batch[cfg.id_col].tolist()
        
        # Download batch
        try:
            # Create temporary directory for batch
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download to temp directory first
                pool.batch_download_to_directory(
                    resource_ids=ids,
                    dst_dir=temp_path,
                    max_workers=cfg.workers
                )
                
                # Move files to sharded directories and write JSON
                for file_id in ids:
                    # Find downloaded file
                    downloaded_files = list(temp_path.glob(f"{file_id}.*"))
                    
                    if downloaded_files:
                        src_file = downloaded_files[0]
                        dst_file = sharding.get_file_path(file_id, src_file.suffix)
                        
                        # Atomic move
                        shutil.move(str(src_file), str(dst_file))
                        
                        # Write JSON metadata only after successful download
                        if cfg.save_filtered_metadata and cfg.per_image_json:
                            row = batch[batch[cfg.id_col] == file_id].iloc[0]
                            metadata = process_metadata_record(row, cfg)
                            json_writer.write(file_id, metadata)
                        
                        progress.update(processed=1)
                    else:
                        progress.update(failed=1)
                
        except Exception as e:
            log.error(f"Batch download failed: {e}")
            progress.update(failed=len(ids))
    
    # Cleanup
    json_writer.close()
    progress.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_tag_list(value: str) -> List[str]:
    """Parse comma- or space-separated tag strings into a list."""
    return [t for t in re.split(r"[\s,]+", value.strip()) if t]


def build_cli() -> argparse.ArgumentParser:
    """Define command-line argument parser."""
    p = argparse.ArgumentParser(
        prog="danbooru-puller-optimized",
        description="Optimized Danbooru metadata filter & image downloader.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    # Paths
    p.add_argument("--metadata", type=str, help="Path to Danbooru parquet")
    p.add_argument("--output", type=str, help="Destination directory")
    
    # Auth
    p.add_argument("--token", type=str, help="HF token")
    
    # Filters
    p.add_argument("--include", "-i", type=_parse_tag_list, help="Tags to include")
    p.add_argument("--exclude", "-x", type=_parse_tag_list, help="Tags to exclude")
    p.add_argument("--min-score", type=int, help="Minimum score")
    p.add_argument("--ratings", nargs="*", help="Allowed ratings")
    p.add_argument("--square", action="store_true", help="Require square images")
    p.add_argument("--min-square-size", type=int, help="Min dimension for square")
    p.add_argument("--min-width", type=int)
    p.add_argument("--min-height", type=int)
    p.add_argument("--max-width", type=int)
    p.add_argument("--max-height", type=int)
    
    # Behavior
    p.add_argument("--no-download", dest="download", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--batch-size", type=int, help="Batch size for processing")
    
    # Performance
    p.add_argument("--workers", type=int, help="Number of download workers")
    p.add_argument("--files-per-shard", type=int, help="Files per shard directory")
    
    return p


def apply_cli_overrides(args: argparse.Namespace, cfg: Config) -> None:
    """Apply CLI arguments to configuration."""
    if args.metadata: cfg.metadata_db_path = args.metadata
    if args.output: cfg.output_dir = args.output
    if args.token: cfg.hf_auth_token = args.token
    
    # Auto-enable filters when arguments provided
    if args.include is not None:
        cfg.include_tags = args.include
        cfg.enable_include_tags = True
    if args.exclude is not None:
        cfg.exclude_tags = args.exclude
        cfg.enable_exclude_tags = True
    
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
    
    if args.workers is not None: cfg.workers = args.workers
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.files_per_shard is not None: cfg.files_per_shard = args.files_per_shard
    
    if hasattr(args, 'download') and not args.download:
        cfg.download_images = False
    
    if args.dry_run:
        cfg.dry_run = True
        cfg.download_images = False


def verify_hf_auth(cfg: Config) -> None:
    """Verify Hugging Face authentication."""
    log.info("🔐 Verifying Hugging Face authentication...")
    token = cfg.hf_auth_token or HfFolder.get_token()
    
    if not token:
        log.error("❌ No Hugging Face token found.")
        sys.exit(1)
    
    try:
        from huggingface_hub import HfApi
        user = HfApi().whoami(token=token)
        log.info(f"✅ Authenticated as: {user['name']}")
    except Exception as e:
        log.error(f"❌ HF authentication failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Main execution with all optimizations."""
    # Setup
    cfg = Config()
    parser = build_cli()
    args = parser.parse_args()
    apply_cli_overrides(args, cfg)
    
    # Initialize components
    stop_handler = SoftStopHandler()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sharding = DirectorySharding(out_dir, cfg.files_per_shard)
    json_writer = BatchedJSONWriter(sharding, cfg.json_batch_size, cfg.json_flush_interval)
    
    # Verify auth if downloading
    if cfg.download_images:
        verify_hf_auth(cfg)
    
    # Load and filter metadata using Polars
    log.info("Loading and filtering metadata...")
    lf = load_and_filter_metadata(cfg)
    
    # Count matches
    match_count = lf.select(pl.count()).collect()[0, 0]
    log.info(f"✅ Found {match_count:,} records matching criteria")
    
    if cfg.dry_run:
        log.info(f"Dry run: Would process {match_count:,} items")
        return
    
    if match_count == 0:
        log.info("No matching records found.")
        return
    
    # Process in streaming fashion
    if cfg.download_images:
        # For downloads, we need the full dataframe
        df = lf.collect(streaming=True).to_pandas()
        download_with_optimizations(df, cfg, sharding, json_writer, stop_handler)
    elif cfg.save_filtered_metadata and not cfg.per_image_json:
        # Save as single file
        df = lf.collect(streaming=True).to_pandas()
        outfile = out_dir / f"filtered_metadata.{cfg.filtered_metadata_format}"
        if cfg.filtered_metadata_format == "json":
            df.to_json(outfile, orient="records", lines=True, force_ascii=False)
        log.info(f"💾 Saved metadata to {outfile}")
    
    log.info("🎉 Processing complete!")


if __name__ == "__main__":
    main()