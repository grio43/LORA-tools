 
#!/usr/bin/env python3
"""
Test if the ID mapping between metadata and tar files is working.
"""

import json
import sys
from pathlib import Path
import polars as pl

def test_mapping(source_dir: str, metadata_path: str):
    """Test if we can map metadata IDs to tar files."""
    
    source_path = Path(source_dir)
    
    # Read sample metadata
    print("Reading metadata sample...")
    df = pl.read_parquet(metadata_path).head(10)
    
    print(f"\nMetadata sample (first 10 rows):")
    for row in df.iter_rows(named=True):
        print(f"  ID: {row['id']}, URL: {row.get('file_url', 'N/A')}")
    
    # Check first JSON file for mapping
    json_files = sorted(source_path.glob("*.json"))[:1]
    if not json_files:
        print("No JSON files found!")
        return
    
    json_path = json_files[0]
    print(f"\n Checking {json_path.name} for ID mappings...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Look for metadata IDs in JSON
    metadata_ids = set(df['id'].to_list())
    found_mappings = []
    
    if 'files' in data and isinstance(data['files'], list):
        for item in data['files']:
            if isinstance(item, dict) and 'id' in item:
                if item['id'] in metadata_ids:
                    found_mappings.append({
                        'metadata_id': item['id'],
                        'filename': item.get('filename', 'unknown')
                    })
    
    if found_mappings:
        print(f"\n✅ Found {len(found_mappings)} matching IDs!")
        for mapping in found_mappings:
            print(f"  Metadata ID {mapping['metadata_id']} -> {mapping['filename']}")
    else:
        print("\n❌ No matching IDs found between metadata and JSON")
        print("\nThis means the JSON files might not contain the metadata ID mapping.")
        print("The IDs in the tar files might be completely independent of metadata IDs.")
        print("\nPossible solutions:")
        print("1. Check if there's a separate mapping file")
        print("2. Use file URLs from metadata to match with tar contents")
        print("3. The dataset might use a different ID system than expected")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_id_mapping.py <source_images_dir> <metadata_path>")
        sys.exit(1)
    
    test_mapping(sys.argv[1], sys.argv[2])