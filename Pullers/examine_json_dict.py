 
#!/usr/bin/env python3
"""
Examine the dictionary structure in JSON files.
"""

import json
from pathlib import Path
import sys
import re

def examine_json_dict(json_path: Path):
    """Examine the dictionary structure of the 'files' field."""
    print(f"\n{'='*60}")
    print(f"Examining {json_path.name}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'files' in data and isinstance(data['files'], dict):
        files_dict = data['files']
        print(f"'files' is a dictionary with {len(files_dict)} entries")
        
        # Get first 10 key-value pairs
        items = list(files_dict.items())[:10]
        
        print("\nFirst 10 entries (key -> value):")
        for key, value in items:
            print(f"  '{key}' -> '{value}'")
        
        # Analyze key patterns
        print("\nAnalyzing keys:")
        keys = list(files_dict.keys())[:100]
        
        # Check if keys are numeric
        numeric_keys = []
        for k in keys:
            if k.isdigit():
                numeric_keys.append(int(k))
        
        if numeric_keys:
            print(f"  Keys appear to be numeric IDs")
            print(f"  Sample numeric keys: {numeric_keys[:10]}")
            print(f"  Key range: {min(numeric_keys)} - {max(numeric_keys)}")
        else:
            print(f"  Keys are not simple numbers")
            print(f"  Sample keys: {keys[:5]}")
        
        # Analyze value patterns
        print("\nAnalyzing values:")
        values = list(files_dict.values())[:100]
        
        # Check if values are filenames
        for v in values[:5]:
            print(f"  '{v}'")
            # Extract ID from filename if present
            match = re.search(r'(\d+)\.(jpg|jpeg|png|gif|webp)', str(v), re.I)
            if match:
                file_id = int(match.group(1))
                print(f"    -> Contains file ID: {file_id}")

def examine_tar_index_cache(cache_path: Path):
    """Examine the tar index cache structure."""
    print(f"\n{'='*60}")
    print(f"Examining {cache_path.name}")
    
    with open(cache_path, 'r') as f:
        data = json.load(f)
    
    print(f"Top-level keys: {list(data.keys())}")
    
    if 'image_to_tar' in data:
        img_to_tar = data['image_to_tar']
        if isinstance(img_to_tar, dict):
            print(f"\n'image_to_tar' dictionary has {len(img_to_tar)} entries")
            # Sample entries
            items = list(img_to_tar.items())[:10]
            print("Sample mappings (image_id -> tar_info):")
            for key, value in items:
                print(f"  '{key}' -> {value}")
    
    if 'tar_to_images' in data:
        tar_to_img = data['tar_to_images']
        if isinstance(tar_to_img, dict):
            print(f"\n'tar_to_images' dictionary has {len(tar_to_img)} entries")
            # Sample entries
            for tar_name, images in list(tar_to_img.items())[:2]:
                print(f"  {tar_name}: {len(images) if isinstance(images, list) else 'not a list'} images")
                if isinstance(images, list):
                    print(f"    Sample: {images[:5]}")

def test_metadata_matching(json_path: Path, metadata_ids: list):
    """Test if metadata IDs match any keys in the JSON."""
    print(f"\n{'='*60}")
    print(f"Testing metadata ID matching in {json_path.name}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'files' in data and isinstance(data['files'], dict):
        files_dict = data['files']
        
        # Convert metadata IDs to strings (keys might be strings)
        metadata_id_strs = [str(mid) for mid in metadata_ids]
        
        found_matches = []
        for mid_str in metadata_id_strs:
            if mid_str in files_dict:
                found_matches.append((mid_str, files_dict[mid_str]))
        
        if found_matches:
            print(f"✅ Found {len(found_matches)} matching metadata IDs!")
            for mid, filename in found_matches:
                print(f"  Metadata ID {mid} -> {filename}")
        else:
            print("❌ No matching metadata IDs found as keys")
            
            # Check if metadata IDs might be in values somehow
            print("\nChecking if metadata IDs appear in values...")
            all_values_str = str(list(files_dict.values())[:100])
            for mid in metadata_ids[:5]:
                if str(mid) in all_values_str:
                    print(f"  Found '{mid}' somewhere in values")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examine_json_dict.py <source_images_dir>")
        sys.exit(1)
    
    source_dir = Path(sys.argv[1])
    
    # Check the tar index cache
    cache_path = source_dir / ".tar_index_cache.json"
    if cache_path.exists():
        examine_tar_index_cache(cache_path)
    
    # Check first JSON file
    json_files = sorted(source_dir.glob("*.json"))
    # Skip the cache file
    json_files = [f for f in json_files if not f.name.startswith('.')]
    
    if json_files:
        examine_json_dict(json_files[0])
        
        # Test with some metadata IDs
        metadata_ids = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]  # From your metadata
        test_metadata_matching(json_files[0], metadata_ids)