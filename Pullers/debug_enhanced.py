#!/usr/bin/env python3
"""
Enhanced debug script to understand the ID mapping between metadata and tar files.
"""

import tarfile
import json
import re
import os
from pathlib import Path
import sys

def analyze_json_structure(json_path: Path):
    """Deeply analyze the JSON structure to find ID mappings."""
    print(f"\n{'='*60}")
    print(f"Deep analysis of {json_path.name}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Print overall structure
        print(f"Top-level type: {type(data)}")
        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")
            
            # Analyze 'files' structure
            if 'files' in data:
                files_data = data['files']
                print(f"\n'files' analysis:")
                print(f"  Type: {type(files_data)}")
                print(f"  Length: {len(files_data)}")
                
                if isinstance(files_data, list) and len(files_data) > 0:
                    # Check first few items
                    print(f"\n  First 5 items structure:")
                    for i, item in enumerate(files_data[:5]):
                        print(f"    Item {i}:")
                        if isinstance(item, dict):
                            print(f"      Type: dict")
                            print(f"      Keys: {list(item.keys())}")
                            # Print all key-value pairs
                            for k, v in item.items():
                                if isinstance(v, str) and len(v) > 50:
                                    print(f"        {k}: {v[:50]}...")
                                else:
                                    print(f"        {k}: {v}")
                        else:
                            print(f"      Type: {type(item)}")
                            print(f"      Value: {item}")
                    
                    # Look for ID patterns
                    print(f"\n  Looking for ID mappings...")
                    id_mappings = []
                    for item in files_data[:20]:  # Check first 20
                        if isinstance(item, dict):
                            # Look for any 'id' field
                            if 'id' in item:
                                filename = item.get('filename', item.get('name', 'unknown'))
                                # Extract ID from filename
                                match = re.search(r'(\d+)\.(jpg|jpeg|png|gif|webp)', filename, re.I)
                                if match:
                                    file_id = int(match.group(1))
                                    metadata_id = item['id']
                                    id_mappings.append((metadata_id, file_id, filename))
                    
                    if id_mappings:
                        print(f"  Found ID mappings (metadata_id -> file_id):")
                        for mid, fid, fname in id_mappings[:10]:
                            print(f"    Metadata ID {mid} -> File ID {fid} ({fname})")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def compare_with_metadata(metadata_path: str, json_paths: list):
    """Compare metadata IDs with JSON mappings."""
    print(f"\n{'='*60}")
    print("Comparing metadata with JSON mappings")
    
    try:
        import polars as pl
        
        # Read sample of metadata
        df = pl.read_parquet(metadata_path).select(['id', 'file_url']).head(100)
        metadata_ids = set(df['id'].to_list())
        
        # Check if any JSON contains these IDs
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if 'files' in data and isinstance(data['files'], list):
                json_ids = set()
                for item in data['files']:
                    if isinstance(item, dict) and 'id' in item:
                        json_ids.add(item['id'])
                
                overlap = metadata_ids.intersection(json_ids)
                if overlap:
                    print(f"\n{json_path.name}: Found {len(overlap)} matching IDs!")
                    print(f"  Sample matches: {list(overlap)[:5]}")
                else:
                    print(f"\n{json_path.name}: No matching IDs found")
                    print(f"  JSON ID range: {min(json_ids) if json_ids else 'N/A'} - {max(json_ids) if json_ids else 'N/A'}")
                    print(f"  Metadata ID range: {min(metadata_ids)} - {max(metadata_ids)}")
                    
    except Exception as e:
        print(f"ERROR: {e}")

def find_id_pattern(source_dir: str):
    """Try to find the pattern of how metadata IDs map to file IDs."""
    source_path = Path(source_dir)
    tar_files = sorted(source_path.glob("*.tar"))[:10]  # First 10 tars
    
    print(f"\n{'='*60}")
    print("Searching for ID patterns across tar files")
    
    all_file_ids = []
    tar_patterns = {}
    
    for tar_path in tar_files:
        tar_name = tar_path.name
        # Extract tar number if present
        tar_num_match = re.search(r'(\d{4})\.tar', tar_name)
        if tar_num_match:
            tar_num = int(tar_num_match.group(1))
        else:
            tar_num = None
        
        # Sample file IDs from tar
        with tarfile.open(tar_path, 'r') as tar:
            file_ids = []
            for member in tar:
                if member.isfile():
                    match = re.search(r'(\d+)\.(jpg|jpeg|png|gif|webp)', member.name, re.I)
                    if match:
                        file_ids.append(int(match.group(1)))
                        if len(file_ids) >= 100:
                            break
            
            if file_ids:
                min_id = min(file_ids)
                max_id = max(file_ids)
                all_file_ids.extend(file_ids[:10])
                
                tar_patterns[tar_name] = {
                    'tar_num': tar_num,
                    'min_id': min_id,
                    'max_id': max_id,
                    'sample_ids': file_ids[:5]
                }
                
                print(f"\n{tar_name}:")
                print(f"  Tar number: {tar_num}")
                print(f"  ID range: {min_id:,} - {max_id:,}")
                print(f"  Sample IDs: {file_ids[:5]}")
    
    # Try to find pattern
    print(f"\n{'='*60}")
    print("Pattern Analysis:")
    
    # Check if there's a consistent offset
    if tar_patterns:
        sorted_tars = sorted(tar_patterns.items(), key=lambda x: x[1]['tar_num'] if x[1]['tar_num'] is not None else 0)
        
        for i, (tar_name, info) in enumerate(sorted_tars[:5]):
            if info['tar_num'] is not None:
                # Check if there's a pattern like: file_id = base + tar_num * 1000 + offset
                print(f"\nTar {info['tar_num']:04d}: {info['min_id']:,} - {info['max_id']:,}")
                
                # Look for modulo pattern
                for sample_id in info['sample_ids'][:3]:
                    mod_1000 = sample_id % 1000
                    div_1000 = sample_id // 1000
                    print(f"  {sample_id:,} = {div_1000:,} * 1000 + {mod_1000}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_enhanced.py <source_images_dir> [metadata_path]")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    source_path = Path(source_dir)
    
    # Analyze first few JSON files in detail
    json_files = sorted(source_path.glob("*.json"))[:5]
    for json_path in json_files:
        analyze_json_structure(json_path)
    
    # Find ID patterns
    find_id_pattern(source_dir)
    
    # Compare with metadata if provided
    if len(sys.argv) > 2:
        metadata_path = sys.argv[2]
        compare_with_metadata(metadata_path, json_files)
