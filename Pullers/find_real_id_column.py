 
#!/usr/bin/env python3
"""
Find which column in the metadata contains the actual file IDs (8349000, etc.)
"""

import polars as pl
import sys
from pathlib import Path

def find_file_id_column(metadata_path: str):
    """Find which column contains the file IDs that match tar files."""
    
    print("Examining metadata structure...")
    
    # Read metadata
    df = pl.scan_parquet(metadata_path)
    
    # Get all columns
    columns = df.columns
    print(f"\nAll columns in metadata: {columns}")
    
    # Read a sample
    sample_df = df.head(100).collect()
    
    # Look for columns that might contain large IDs like 8349000
    print("\n" + "="*60)
    print("Checking each column for file IDs (looking for values like 8349000)...")
    
    for col in columns:
        try:
            # Get column data type
            dtype = sample_df[col].dtype
            
            # Only check numeric columns
            if dtype in [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
                values = sample_df[col].drop_nulls().head(10).to_list()
                if values:
                    max_val = max(v for v in values if v is not None)
                    min_val = min(v for v in values if v is not None)
                    
                    # Check if values are in the millions (like our file IDs)
                    if max_val > 1000000:
                        print(f"\n✅ '{col}' - LIKELY MATCH!")
                        print(f"   Type: {dtype}")
                        print(f"   Range: {min_val:,.0f} - {max_val:,.0f}")
                        print(f"   Sample: {values[:5]}")
                        
                        # Check if these IDs would match our tar files
                        if any(8000000 < v < 9000000 for v in values if v is not None):
                            print(f"   ⭐ VALUES IN TAR FILE RANGE (8-9 million)!")
                    elif max_val > 100:
                        print(f"\n'{col}' - possible but values too small")
                        print(f"   Range: {min_val} - {max_val}")
        except Exception as e:
            pass
    
    # Also check for specific column names that are commonly used
    print("\n" + "="*60)
    print("Checking for common ID column names...")
    
    common_names = ['post_id', 'file_id', 'image_id', 'danbooru_id', 'pixiv_id', 'md5', 'file_url']
    for name in common_names:
        if name in columns:
            print(f"\nFound column: '{name}'")
            values = sample_df[name].head(5).to_list()
            print(f"  Sample values: {values}")

def test_id_matching(metadata_path: str):
    """Test if we can find the file IDs in metadata."""
    
    # Known file IDs from your tar files
    known_file_ids = [8349000, 8348000, 8347000, 8346000, 8345000]
    
    print(f"\n" + "="*60)
    print(f"Testing if metadata contains these known file IDs: {known_file_ids}")
    
    df = pl.read_parquet(metadata_path)
    
    # Check each numeric column
    for col in df.columns:
        try:
            dtype = df[col].dtype
            if dtype in [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
                # Check if any of our known IDs exist in this column
                matches = df.filter(pl.col(col).is_in(known_file_ids))
                if len(matches) > 0:
                    print(f"\n🎯 FOUND MATCHES in column '{col}'!")
                    print(f"   Matched {len(matches)} file IDs")
                    print(f"   Sample matches:")
                    for row in matches.head(3).iter_rows(named=True):
                        print(f"     Row ID {row['id']}: {col}={row[col]}")
                    return col
        except:
            pass
    
    print("\n❌ No column found with matching file IDs")
    print("The metadata might use a different ID system than the files")
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_real_id_column.py <metadata_path>")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    
    find_file_id_column(metadata_path)
    matching_column = test_id_matching(metadata_path)
    
    if matching_column:
        print(f"\n" + "="*60)
        print(f"✅ SUCCESS! Use column '{matching_column}' instead of 'id'")
        print(f"Update your Config class:")
        print(f"  cfg.id_col = '{matching_column}'  # Instead of 'id'")
    else:
        print(f"\n" + "="*60)
        print("⚠️ No matching column found. Possible solutions:")
        print("1. The metadata might be from a different dataset version")
        print("2. The file IDs might need transformation (e.g., adding an offset)")
        print("3. Check if there's a separate mapping file in the dataset")