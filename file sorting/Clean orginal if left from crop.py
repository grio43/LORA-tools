import os
import glob

# --- CONFIGURATION ---------------------------------------------------
# The folder containing the original images you want to delete.
SOURCE_FOLDER = r"J:\New file\Danbooru2004\Images\new"

# The folder containing the generated crops.
CROP_FOLDER = r"J:\New file\Danbooru2004\Images\new\crop"

# File types to check for.
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"]

# --- DANGER ZONE ---
# Set to True to actually delete files. 
# It is STRONGLY recommended to run with False first to see what will be deleted.
PERFORM_DELETION = False
# ---------------------------------------------------------------------

def list_files(folder, patterns):
    """Returns a list of files matching the patterns in the given folder."""
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files.sort()
    return files

def main():
    """
    Main function to check for existing crops and delete the corresponding
    original files.
    """
    if PERFORM_DELETION:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: DELETION IS ENABLED. FILES WILL BE REMOVED. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        # Give the user a moment to cancel
        try:
            input("Press Enter to continue, or close this window (Ctrl+C) to cancel...")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return
    else:
        print("--- RUNNING IN PREVIEW MODE. NO FILES WILL BE DELETED. ---")
        print("--- Set PERFORM_DELETION to True to enable deletion. ---\n")

    originals = list_files(SOURCE_FOLDER, EXTENSIONS)
    if not originals:
        print(f"No image files found in the source folder: {SOURCE_FOLDER}")
        return

    deleted_count = 0
    skipped_count = 0
    error_count = 0

    print(f"Scanning {len(originals)} files in the source folder...")

    for fpath in originals:
        try:
            base_name_with_ext = os.path.basename(fpath)
            base_name, ext = os.path.splitext(base_name_with_ext)

            # We only need to check for the existence of the FIRST crop.
            # If `_1` exists, we can be confident the process ran.
            expected_crop_path = os.path.join(CROP_FOLDER, f"{base_name}_1{ext}")

            if os.path.exists(expected_crop_path):
                print(f"[FOUND] Match for '{base_name_with_ext}'. Marked for deletion.")
                
                if PERFORM_DELETION:
                    original_json_path = os.path.splitext(fpath)[0] + ".json"
                    # Delete the image file
                    os.remove(fpath)
                    # Delete the corresponding json, if it exists
                    if os.path.exists(original_json_path):
                        os.remove(original_json_path)
                    print(f"  -> DELETED: {base_name_with_ext} and its .json")
                deleted_count += 1
            else:
                # This handles originals that were never cropped (e.g., they were too small)
                print(f"[SKIPPED] No crop found for '{base_name_with_ext}'. Leaving it untouched.")
                skipped_count += 1

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while processing {fpath}: {e}")
            error_count += 1
            continue
    
    print("\n--- Cleanup Complete ---")
    if PERFORM_DELETION:
        print(f"Files Deleted: {deleted_count}")
    else:
        print(f"Files that would be deleted: {deleted_count}")
    print(f"Files Skipped (No Crops Found): {skipped_count}")
    print(f"Errors: {error_count}")
    print("------------------------")


if __name__ == "__main__":
    main()