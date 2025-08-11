#!/usr/bin/env python3
"""
Enhanced Danbooru cleanup script - Remove corrupted and orphan files.
Features:
- Multi-threaded processing (1 worker per shard)
- Detects multiple types of corruption
- Removes orphan files (JSON without images, images without JSON)
- Validates image files can actually be opened
- Updates progress tracker for redownloading
- Compatible with main puller script's progress format
"""

from __future__ import annotations

import argparse
import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger("danbooru_cleanup")

# ---------------------------------------------------------------------------
# File Validation
# ---------------------------------------------------------------------------
class FileValidator:
    """Validates image files for various types of corruption."""
    
    def __init__(self, min_file_size: int = 1000, 
                 min_dimensions: Tuple[int, int] = (64, 64),
                 check_image_validity: bool = True):
        self.min_file_size = min_file_size
        self.min_width, self.min_height = min_dimensions
        self.check_image_validity = check_image_validity
    
    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate an image file.
        Returns: (is_valid, reason_if_invalid)
        """
        # Check file exists
        if not file_path.exists():
            return False, "file_not_found"
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size < self.min_file_size:
                return False, f"too_small_{file_size}bytes"
        except OSError as e:
            return False, f"stat_error_{e}"
        
        # Check if it's actually an image that can be opened
        if self.check_image_validity:
            try:
                with Image.open(file_path) as img:
                    # Check dimensions
                    width, height = img.size
                    if width < self.min_width or height < self.min_height:
                        return False, f"dimensions_{width}x{height}"
                    
                    # Try to load the image data (catches truncated files)
                    img.verify()
                    
            except Image.UnidentifiedImageError:
                return False, "not_an_image"
            except Image.DecompressionBombError:
                return False, "decompression_bomb"
            except Exception as e:
                # Truncated, corrupted, or other issues
                return False, f"invalid_image_{type(e).__name__}"
        
        return True, "valid"

# ---------------------------------------------------------------------------
# Progress Tracker (Compatible with main puller)
# ---------------------------------------------------------------------------
class ProgressTracker:
    """Tracks download progress - compatible with main puller script."""
    
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.completed_ids: Set[int] = set()
        self._lock = Lock()
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
    
    def remove_completed(self, image_ids: Set[int]):
        """Remove IDs from completed set (mark as incomplete). Thread-safe."""
        with self._lock:
            self.completed_ids = self.completed_ids - image_ids
            self._save_progress()
    
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
            log.info(f"💾 Updated progress: {len(self.completed_ids):,} remain completed")
        except Exception as e:
            log.error(f"❌ Failed to save progress: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

# ---------------------------------------------------------------------------
# Thread-safe Progress Reporter
# ---------------------------------------------------------------------------
class ProgressReporter:
    """Thread-safe progress reporting for parallel operations."""
    
    def __init__(self, total_tasks: int, task_name: str = "tasks"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.task_name = task_name
        self._lock = Lock()
        self.start_time = time.time()
    
    def update(self, increment: int = 1, message: str = None):
        """Update progress counter."""
        with self._lock:
            self.completed_tasks += increment
            elapsed = time.time() - self.start_time
            rate = self.completed_tasks / elapsed if elapsed > 0 else 0
            
            if self.completed_tasks % 10 == 0 or self.completed_tasks == self.total_tasks:
                progress_msg = f"  ⚡ Processed {self.completed_tasks}/{self.total_tasks} {self.task_name}"
                if rate > 0:
                    eta = (self.total_tasks - self.completed_tasks) / rate
                    progress_msg += f" ({rate:.1f}/s, ETA: {eta:.0f}s)"
                
                if message:
                    progress_msg += f" - {message}"
                
                log.info(progress_msg)

# ---------------------------------------------------------------------------
# Enhanced File System Scanner with Multi-threading
# ---------------------------------------------------------------------------
class EnhancedFileSystemScanner:
    """Scans for corrupted and orphan files with multi-threading support."""
    
    def __init__(self, base_dir: Path, validator: FileValidator, max_workers: int = None):
        self.base_dir = base_dir
        self.validator = validator
        self.max_workers = max_workers
        
    def scan_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Scan a directory for all issues."""
        results = {
            'valid_pairs': {},  # id -> (image_path, json_path, size)
            'corrupted_images': {},  # id -> (path, size, reason)
            'orphan_jsons': {},  # id -> path (JSON without image)
            'orphan_images': {},  # id -> (path, size) (image without JSON)
            'unknown_files': [],  # Files that don't match expected patterns
            'total_size': 0,
            'file_count': 0,
            'shard_name': dir_path.name  # Track which shard this is from
        }
        
        # First, collect all files
        image_files = {}  # id -> (path, extension)
        json_files = {}   # id -> path
        
        for file_path in dir_path.iterdir():
            if file_path.is_dir():
                continue
                
            results['file_count'] += 1
            
            # Try to extract ID from filename
            stem = file_path.stem
            
            # Handle image files
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                try:
                    image_id = int(stem)
                    image_files[image_id] = file_path
                except ValueError:
                    # Not a valid ID format
                    results['unknown_files'].append(str(file_path))
                    
            # Handle JSON files
            elif file_path.suffix.lower() == '.json':
                try:
                    image_id = int(stem)
                    json_files[image_id] = file_path
                except ValueError:
                    results['unknown_files'].append(str(file_path))
            
            # Track size
            try:
                results['total_size'] += file_path.stat().st_size
            except OSError:
                pass
        
        # Now analyze the files
        all_ids = set(image_files.keys()) | set(json_files.keys())
        
        for image_id in all_ids:
            has_image = image_id in image_files
            has_json = image_id in json_files
            
            if has_image and has_json:
                # Both exist - validate the image
                img_path = image_files[image_id]
                json_path = json_files[image_id]
                
                is_valid, reason = self.validator.validate_file(img_path)
                
                if is_valid:
                    try:
                        size = img_path.stat().st_size
                        results['valid_pairs'][image_id] = (str(img_path), str(json_path), size)
                    except OSError:
                        pass
                else:
                    try:
                        size = img_path.stat().st_size
                    except OSError:
                        size = 0
                    results['corrupted_images'][image_id] = (str(img_path), size, reason)
                    
            elif has_image and not has_json:
                # Image without JSON - orphan image
                img_path = image_files[image_id]
                
                # Still check if image is valid
                is_valid, reason = self.validator.validate_file(img_path)
                
                try:
                    size = img_path.stat().st_size
                except OSError:
                    size = 0
                    
                if not is_valid:
                    results['corrupted_images'][image_id] = (str(img_path), size, reason)
                else:
                    results['orphan_images'][image_id] = (str(img_path), size)
                    
            elif has_json and not has_image:
                # JSON without image - orphan JSON
                results['orphan_jsons'][image_id] = str(json_files[image_id])
        
        return results
    
    def _scan_shard_wrapper(self, shard_info: Tuple[int, Path], progress: ProgressReporter) -> Dict[str, Any]:
        """Wrapper for scanning a single shard with progress reporting."""
        idx, shard_dir = shard_info
        thread_id = threading.current_thread().name
        
        log.debug(f"  🔍 [{thread_id}] Scanning {shard_dir.name}...")
        results = self.scan_directory(shard_dir)
        
        # Report progress
        issues_found = (len(results['corrupted_images']) + 
                       len(results['orphan_jsons']) + 
                       len(results['orphan_images']))
        progress.update(1, f"{shard_dir.name}: {issues_found} issues")
        
        return results
    
    def scan_all_shards(self) -> Dict[str, Any]:
        """Scan all shard directories or base directory using multiple threads."""
        shard_dirs = sorted([d for d in self.base_dir.glob("shard_*") if d.is_dir()])
        
        if not shard_dirs:
            # No shards, scan base directory
            log.info(f"🔍 No shard directories found, scanning base directory...")
            results = self.scan_directory(self.base_dir)
            return self._compile_results([results])
        
        log.info(f"🔍 Scanning {len(shard_dirs)} shard directories...")
        
        # Determine number of workers (1 per shard, but cap at reasonable limit)
        if self.max_workers:
            num_workers = min(self.max_workers, len(shard_dirs))
        else:
            # Default: min of shard count or CPU count * 2
            num_workers = min(len(shard_dirs), os.cpu_count() * 2 if os.cpu_count() else 8)
        
        log.info(f"  🚀 Using {num_workers} parallel workers")
        
        all_results = []
        progress = ProgressReporter(len(shard_dirs), "shards")
        
        # Process shards in parallel
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="Scanner") as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._scan_shard_wrapper, (idx, shard_dir), progress): shard_dir
                for idx, shard_dir in enumerate(shard_dirs)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                shard_dir = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    log.error(f"❌ Error scanning {shard_dir}: {e}")
                    # Create empty result for failed shard
                    all_results.append({
                        'valid_pairs': {},
                        'corrupted_images': {},
                        'orphan_jsons': {},
                        'orphan_images': {},
                        'unknown_files': [],
                        'file_count': 0,
                        'total_size': 0,
                        'shard_name': shard_dir.name
                    })
        
        return self._compile_results(all_results)
    
    def _compile_results(self, results_list: List[Dict]) -> Dict[str, Any]:
        """Compile results from multiple directories."""
        total_stats = {
            'total_files': 0,
            'total_size': 0,
            'valid_pairs_count': 0,
            'corrupted_count': 0,
            'orphan_jsons_count': 0,
            'orphan_images_count': 0,
            'unknown_files_count': 0,
            'corrupted_images': {},  # id -> (path, size, reason)
            'orphan_jsons': {},  # id -> path
            'orphan_images': {},  # id -> (path, size)
            'unknown_files': [],
            'corruption_reasons': {},  # reason -> count
            'shards_processed': len(results_list)
        }
        
        for results in results_list:
            total_stats['total_files'] += results['file_count']
            total_stats['total_size'] += results['total_size']
            total_stats['valid_pairs_count'] += len(results['valid_pairs'])
            total_stats['corrupted_count'] += len(results['corrupted_images'])
            total_stats['orphan_jsons_count'] += len(results['orphan_jsons'])
            total_stats['orphan_images_count'] += len(results['orphan_images'])
            total_stats['unknown_files_count'] += len(results['unknown_files'])
            
            # Collect all issues
            total_stats['corrupted_images'].update(results['corrupted_images'])
            total_stats['orphan_jsons'].update(results['orphan_jsons'])
            total_stats['orphan_images'].update(results['orphan_images'])
            total_stats['unknown_files'].extend(results['unknown_files'])
            
            # Track corruption reasons
            for _, _, reason in results['corrupted_images'].values():
                total_stats['corruption_reasons'][reason] = \
                    total_stats['corruption_reasons'].get(reason, 0) + 1
        
        total_stats['total_size_gb'] = total_stats['total_size'] / (1024**3)
        total_stats['total_issues'] = (
            total_stats['corrupted_count'] + 
            total_stats['orphan_jsons_count'] + 
            total_stats['orphan_images_count']
        )
        
        return total_stats

# ---------------------------------------------------------------------------
# Multi-threaded Cleanup Functions
# ---------------------------------------------------------------------------
class ParallelFileRemover:
    """Handles parallel file removal operations."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(16, os.cpu_count() * 2 if os.cpu_count() else 8)
        self.stats_lock = Lock()
        self.stats = {
            'removed_images': 0,
            'removed_jsons': 0,
            'removed_size': 0,
            'failed_removals': [],
            'removed_ids': set()
        }
    
    def _remove_file_pair(self, task: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        """Remove a single file or file pair."""
        result = {
            'removed_images': 0,
            'removed_jsons': 0,
            'removed_size': 0,
            'removed_ids': set(),
            'failures': []
        }
        
        image_id = task['id']
        file_type = task['type']
        
        if file_type == 'corrupted':
            img_path, size, reason = task['data']
            
            if dry_run:
                result['removed_images'] = 1
                result['removed_size'] = size
                result['removed_ids'].add(image_id)
                log.debug(f"[DRY RUN] Would remove corrupted image: {img_path} (reason: {reason})")
            else:
                try:
                    img_file = Path(img_path)
                    json_file = img_file.with_suffix('.json')
                    
                    # Remove image
                    if img_file.exists():
                        img_file.unlink()
                        result['removed_images'] = 1
                        result['removed_size'] = size
                        result['removed_ids'].add(image_id)
                    
                    # Remove associated JSON
                    if json_file.exists():
                        json_file.unlink()
                        result['removed_jsons'] = 1
                        
                except Exception as e:
                    result['failures'].append((img_path, str(e)))
        
        elif file_type == 'orphan_json':
            json_path = task['data']
            
            if dry_run:
                result['removed_jsons'] = 1
                log.debug(f"[DRY RUN] Would remove orphan JSON: {json_path}")
            else:
                try:
                    json_file = Path(json_path)
                    if json_file.exists():
                        json_file.unlink()
                        result['removed_jsons'] = 1
                        result['removed_ids'].add(image_id)
                except Exception as e:
                    result['failures'].append((json_path, str(e)))
        
        elif file_type == 'orphan_image':
            img_path, size = task['data']
            
            if dry_run:
                result['removed_images'] = 1
                result['removed_size'] = size
                log.debug(f"[DRY RUN] Would remove orphan image: {img_path}")
            else:
                try:
                    img_file = Path(img_path)
                    if img_file.exists():
                        img_file.unlink()
                        result['removed_images'] = 1
                        result['removed_size'] = size
                        result['removed_ids'].add(image_id)
                except Exception as e:
                    result['failures'].append((img_path, str(e)))
        
        return result
    
    def _update_stats(self, result: Dict[str, Any]):
        """Update global stats with result from a single removal."""
        with self.stats_lock:
            self.stats['removed_images'] += result['removed_images']
            self.stats['removed_jsons'] += result['removed_jsons']
            self.stats['removed_size'] += result['removed_size']
            self.stats['removed_ids'].update(result['removed_ids'])
            self.stats['failed_removals'].extend(result['failures'])
    
    def remove_files(self, scan_results: Dict[str, Any], 
                    remove_orphans: bool = True,
                    dry_run: bool = False) -> Dict[str, Any]:
        """Remove corrupted and optionally orphan files in parallel."""
        
        # Build task list
        tasks = []
        
        # Add corrupted images
        for image_id, data in scan_results['corrupted_images'].items():
            tasks.append({
                'id': image_id,
                'type': 'corrupted',
                'data': data
            })
        
        # Add orphan files if requested
        if remove_orphans:
            for image_id, data in scan_results['orphan_jsons'].items():
                tasks.append({
                    'id': image_id,
                    'type': 'orphan_json',
                    'data': data
                })
            
            for image_id, data in scan_results['orphan_images'].items():
                tasks.append({
                    'id': image_id,
                    'type': 'orphan_image',
                    'data': data
                })
        
        if not tasks:
            return self.stats
        
        # Process removals in parallel
        total_tasks = len(tasks)
        log.info(f"🗑️  Removing {total_tasks:,} problem files using {self.max_workers} workers...")
        
        progress = ProgressReporter(total_tasks, "files")
        
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="Remover") as executor:
            futures = {
                executor.submit(self._remove_file_pair, task, dry_run): task
                for task in tasks
            }
            
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    self._update_stats(result)
                    progress.update(1)
                except Exception as e:
                    log.error(f"❌ Error removing files for ID {task['id']}: {e}")
                    progress.update(1)
        
        return self.stats

# ---------------------------------------------------------------------------
# Main Cleanup Function
# ---------------------------------------------------------------------------
def run_cleanup(cfg: Config):
    """Main cleanup operation."""
    dest_dir = Path(cfg.output_dir)
    
    if not dest_dir.exists():
        log.error(f"❌ Output directory does not exist: {dest_dir}")
        return 1
    
    log.info("="*70)
    log.info("ENHANCED MULTI-THREADED DANBOORU CLEANUP UTILITY")
    log.info("="*70)
    log.info(f"📁 Directory: {dest_dir}")
    log.info(f"📏 Min file size: {cfg.min_file_size} bytes")
    log.info(f"📐 Min dimensions: {cfg.min_width}x{cfg.min_height}")
    log.info(f"🔍 Validate images: {cfg.check_validity}")
    log.info(f"🗑️  Remove orphans: {cfg.remove_orphans}")
    log.info(f"🚀 Max workers: {cfg.max_workers if cfg.max_workers else 'auto'}")
    log.info(f"🔧 Mode: {'DRY RUN' if cfg.dry_run else 'REMOVE FILES'}")
    
    # Initialize validator
    validator = FileValidator(
        min_file_size=cfg.min_file_size,
        min_dimensions=(cfg.min_width, cfg.min_height),
        check_image_validity=cfg.check_validity
    )
    
    # Step 1: Scan filesystem
    log.info("\n" + "="*50)
    log.info("STEP 1: Scanning filesystem for issues...")
    log.info("="*50)
    
    scanner = EnhancedFileSystemScanner(dest_dir, validator, max_workers=cfg.max_workers)
    scan_start = time.time()
    scan_results = scanner.scan_all_shards()
    scan_time = time.time() - scan_start
    
    # Display scan results
    log.info(f"\n📊 Scan Results (completed in {scan_time:.1f}s):")
    log.info(f"  Total files: {scan_results['total_files']:,}")
    log.info(f"  Total size: {scan_results['total_size_gb']:.2f} GB")
    log.info(f"  Shards processed: {scan_results.get('shards_processed', 1)}")
    log.info(f"  ✅ Valid pairs: {scan_results['valid_pairs_count']:,}")
    log.info(f"  ❌ Corrupted images: {scan_results['corrupted_count']:,}")
    log.info(f"  📄 Orphan JSONs: {scan_results['orphan_jsons_count']:,}")
    log.info(f"  🖼️  Orphan images: {scan_results['orphan_images_count']:,}")
    log.info(f"  ❓ Unknown files: {scan_results['unknown_files_count']:,}")
    log.info(f"  🔧 Total issues: {scan_results['total_issues']:,}")
    
    # Show corruption reasons breakdown
    if scan_results['corruption_reasons']:
        log.info(f"\n  Corruption reasons:")
        for reason, count in sorted(scan_results['corruption_reasons'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
            log.info(f"    {reason}: {count}")
    
    # Show samples
    if scan_results['corrupted_images']:
        sample = list(scan_results['corrupted_images'].items())[:3]
        log.info(f"\n  Sample corrupted files:")
        for image_id, (path, size, reason) in sample:
            log.info(f"    ID {image_id}: {Path(path).name} ({size} bytes, {reason})")
    
    if scan_results['orphan_jsons']:
        sample = list(scan_results['orphan_jsons'].items())[:3]
        log.info(f"\n  Sample orphan JSONs:")
        for image_id, path in sample:
            log.info(f"    ID {image_id}: {Path(path).name}")
    
    if scan_results['total_issues'] == 0:
        log.info("\n✅ No issues found! Dataset is clean.")
        return 0
    
    # Step 2: Remove problem files
    if not cfg.skip_removal:
        log.info("\n" + "="*50)
        log.info("STEP 2: Removing problem files...")
        log.info("="*50)
        
        if cfg.dry_run:
            log.info("🔍 DRY RUN MODE - No files will be removed")
        
        # Confirm before removing (unless dry run or auto-confirm)
        if not cfg.dry_run and not cfg.auto_confirm:
            response = input(f"\n⚠️  Remove {scan_results['total_issues']:,} problem files? (y/N): ")
            if response.lower() != 'y':
                log.info("❌ Removal cancelled by user")
                return 1
        
        # Remove files
        remover = ParallelFileRemover(max_workers=cfg.max_workers)
        removal_start = time.time()
        removal_stats = remover.remove_files(
            scan_results,
            remove_orphans=cfg.remove_orphans,
            dry_run=cfg.dry_run
        )
        removal_time = time.time() - removal_start
        
        # Display removal results
        log.info(f"\n📊 Removal Results (completed in {removal_time:.1f}s):")
        log.info(f"  ✅ Removed images: {removal_stats['removed_images']:,}")
        log.info(f"  ✅ Removed JSONs: {removal_stats['removed_jsons']:,}")
        log.info(f"  💾 Space freed: {removal_stats['removed_size'] / (1024**3):.2f} GB")
        if removal_stats['failed_removals']:
            log.info(f"  ❌ Failed: {len(removal_stats['failed_removals'])} files")
            for path, error in removal_stats['failed_removals'][:5]:
                log.error(f"    {path}: {error}")
        
        # Step 3: Update progress tracking
        if not cfg.dry_run and removal_stats['removed_ids']:
            log.info("\n" + "="*50)
            log.info("STEP 3: Updating progress tracking...")
            log.info("="*50)
            
            # Update main progress file
            progress_file = dest_dir / "progress.json"
            if progress_file.exists():
                tracker = ProgressTracker(progress_file)
                original_count = len(tracker.completed_ids)
                tracker.remove_completed(removal_stats['removed_ids'])
                removed_from_progress = original_count - len(tracker.completed_ids)
                log.info(f"📝 Removed {removed_from_progress:,} IDs from progress tracker")
            else:
                log.info("📝 No progress file found - will be created by puller")
            
            total_time = time.time() - scan_start
            log.info(f"\n✅ Cleanup complete in {total_time:.1f}s! Removed files will be redownloaded by the main puller.")
    else:
        log.info("\n✅ Scan complete (removal skipped)")
    
    return 0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Configuration for cleanup operations."""
    output_dir: str = "/media/andrewk/qnap-public/Training data/"
    min_file_size: int = 1000  # Files smaller than this are considered corrupted
    min_width: int = 64  # Minimum image width
    min_height: int = 64  # Minimum image height
    check_validity: bool = True  # Actually try to open images
    remove_orphans: bool = True  # Remove orphan files
    dry_run: bool = False
    auto_confirm: bool = False
    skip_removal: bool = False
    max_workers: int = None  # None = auto-detect

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_cli() -> argparse.ArgumentParser:
    """Build command-line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-threaded Danbooru Cleanup - Remove corrupted and orphan files",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
This script detects and removes:
1. Corrupted images (too small, invalid format, truncated)
2. Orphan JSON files (JSON without corresponding image)
3. Orphan image files (image without corresponding JSON)
4. Updates progress tracking for redownloading

Multi-threading:
- Automatically uses 1 worker per shard for scanning
- Parallel file removal for faster cleanup
- Thread-safe progress tracking

Examples:
  # Dry run to see all issues
  python cleanup.py --dry-run
  
  # Remove all problem files (will prompt)
  python cleanup.py
  
  # Remove without confirmation
  python cleanup.py --yes
  
  # Only remove corrupted, keep orphans
  python cleanup.py --keep-orphans
  
  # Quick mode (only check file size)
  python cleanup.py --quick
  
  # Custom thresholds
  python cleanup.py --min-size 5000 --min-width 128 --min-height 128
  
  # Limit worker threads
  python cleanup.py --workers 8
        """
    )
    
    parser.add_argument('--output', type=str, 
                       default="/media/andrewk/qnap-public/Training data/",
                       help='Output directory to clean')
    parser.add_argument('--min-size', type=int, default=1000,
                       help='Minimum file size in bytes')
    parser.add_argument('--min-width', type=int, default=64,
                       help='Minimum image width in pixels')
    parser.add_argument('--min-height', type=int, default=64,
                       help='Minimum image height in pixels')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode - only check file size, not validity')
    parser.add_argument('--keep-orphans', action='store_true',
                       help='Keep orphan files (only remove corrupted)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be removed without removing')
    parser.add_argument('--yes', '-y', action='store_true', dest='auto_confirm',
                       help='Auto-confirm removal (no prompt)')
    parser.add_argument('--scan-only', action='store_true', dest='skip_removal',
                       help='Only scan, do not remove files')
    parser.add_argument('--workers', type=int, default=None,
                       help='Max number of worker threads (default: auto)')
    
    return parser

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = build_cli()
    args = parser.parse_args()
    
    cfg = Config()
    
    if args.output:
        cfg.output_dir = args.output
    if args.min_size:
        cfg.min_file_size = args.min_size
    if args.min_width:
        cfg.min_width = args.min_width
    if args.min_height:
        cfg.min_height = args.min_height
    if args.quick:
        cfg.check_validity = False
    if args.keep_orphans:
        cfg.remove_orphans = False
    if args.dry_run:
        cfg.dry_run = True
    if args.auto_confirm:
        cfg.auto_confirm = True
    if args.skip_removal:
        cfg.skip_removal = True
    if args.workers:
        cfg.max_workers = args.workers
    
    try:
        # Check for PIL/Pillow
        try:
            from PIL import Image
        except ImportError:
            log.warning("⚠️  Pillow not installed. Install with 'pip install Pillow' for full validation.")
            cfg.check_validity = False
        
        return run_cleanup(cfg)
    except KeyboardInterrupt:
        log.warning("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        log.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())