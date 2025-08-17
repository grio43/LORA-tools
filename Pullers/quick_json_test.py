#!/usr/bin/env python3
"""
Quick test to check if JSON files contain metadata ID to file ID mapping.
"""

import json
from pathlib import Path
import sys

def test_json_mapping(json_path: Path):
    """Test if JSON contains the ID mapping we need."""
    print(f"\nTesting {json_path.name}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'files' in data and isinstance(data['files'], list):
        # Check if files list contains dicts or just strings
        if len(data['files']) > 0:
            first_item = data['files'][0]
            print(f"First item type: {type(first_item)}")
            
            if isinstance(first_item, dict):
                print("Files contain dictionaries!")
                print(f"Keys in first item: {list(first_item.keys())}")
                
                # Check a few items for ID field
                for i, item in enumerate(data['files'][:5]):
                    if 'id' in item:
                        print(f"  Item {i}: id={item.get('id')}, filename={item.get('filename', 'N/A')}")
                    else:
                        print(f"  Item {i}: No 'id' field. Keys: {list(item.keys())}")
            
            elif isinstance(first_item, str):
                print("Files contain strings (just filenames)")
                print(f"Sample: {data['files'][:5]}")
            
            elif isinstance(first_item, list):
                print("Files contain lists")
                print(f"First item: {first_item}")
    else:
        print("No 'files' field or it's not a list")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_json_test.py <json_file_path>")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    test_json_mapping(json_path)