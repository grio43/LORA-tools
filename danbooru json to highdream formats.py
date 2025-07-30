import os
import json
import shutil

# A list of common image file extensions to look for.
# You can add or remove extensions as needed for your dataset.
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff']

def process_and_organize_files(source_dir, processed_dir, backup_dir):
    """
    Processes JSON files from a source directory, saves the transformed versions
    to a processed directory, and moves the originals to a backup directory.

    Args:
        source_dir (str): Path to the folder with original images and JSONs.
        processed_dir (str): Path to the folder where new JSONs will be saved.
        backup_dir (str): Path to the folder where original JSONs will be moved.
    """
    # Create the destination folders if they don't already exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    print(f"Reading from: {source_dir}")
    print(f"Saving processed JSONs to: {processed_dir}")
    print(f"Backing up original JSONs to: {backup_dir}")
    print("-" * 30)


    # Get a list of all files to avoid issues with moving files during iteration
    try:
        all_files = os.listdir(source_dir)
    except FileNotFoundError:
        print(f"FATAL ERROR: The source directory was not found at '{source_dir}'")
        return

    for filename in all_files:
        # We only trigger the process for JSON files
        if not filename.endswith('.json'):
            continue

        base_name = os.path.splitext(filename)[0]
        original_json_path = os.path.join(source_dir, filename)

        # --- 1. Find the matching image file in the source directory ---
        found_image_name = None
        for ext in IMAGE_EXTENSIONS:
            potential_image_name = base_name + ext
            if os.path.exists(os.path.join(source_dir, potential_image_name)):
                found_image_name = potential_image_name
                break # Stop looking once found

        if not found_image_name:
            print(f"Warning: No matching image found for '{filename}' in '{source_dir}'. Skipping file.")
            continue

        # --- 2. Process the JSON and move the original ---
        try:
            # Define final paths for the new and old JSON
            processed_json_path = os.path.join(processed_dir, filename)
            backup_json_path = os.path.join(backup_dir, filename)

            # Read the original JSON data
            with open(original_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Combine artist tag and general tags
            tags = []
            if data.get("tag_string_artist"):
                # Replace underscores for a more natural tag
                tags.append(data.get("tag_string_artist").replace('_', ' '))
            if data.get("tag_string"):
                tags.extend(data.get("tag_string").split())

            # Create the new, restructured JSON data
            new_data = {
                "file_name": found_image_name,
                "tags": ", ".join(tags),
                "caption": "" # Leave caption empty as requested
            }

            # Write the new JSON file to the 'processed' directory
            with open(processed_json_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)

            # Move the original JSON file to the 'backup' directory
            shutil.move(original_json_path, backup_json_path)

            print(f"Processed: '{filename}' -> Moved original to backup, created new in processed.")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{filename}'. It might be corrupted. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")

# --- Configuration: SET YOUR FOLDER PATHS HERE ---

# 1. The folder containing your original images and original .json files.
#    Example: "C:/Users/YourUser/Desktop/MyDataset"
SOURCE_DIRECTORY = r"J:\New file\Ready for captions without tags"

# 2. The folder where the new, transformed .json files will be saved.
#    Example: "D:/Processed_JSONs"
PROCESSED_DIRECTORY = r"J:\New file\json_processed"

# 3. The folder where the original .json files will be moved for backup.
#    Example: "D:/Backup_JSONs"
BACKUP_DIRECTORY = r"J:\New file\json_backup"

# --------------------------------------------------

# --- Main execution ---
if __name__ == "__main__":
    # Basic check to ensure user has changed the default paths
    if "C:/path/to/" in SOURCE_DIRECTORY:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE THE DIRECTORY PATHS IN THE SCRIPT    !!!")
        print("!!! You must set SOURCE_DIRECTORY, PROCESSED_DIRECTORY,!!!")
        print("!!! and BACKUP_DIRECTORY before running.               !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        process_and_organize_files(
            source_dir=SOURCE_DIRECTORY,
            processed_dir=PROCESSED_DIRECTORY,
            backup_dir=BACKUP_DIRECTORY
        )
        print("\nProcessing complete.")