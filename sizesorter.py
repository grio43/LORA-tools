import os
import shutil
from PIL import Image
import concurrent.futures
from functools import partial
import time
import threading
import sys
import signal
import warnings

# --- Configuration ---
# Set the path to the folder containing your images and JSON files.
start_folder = r"J:\New file\Danbooru2004\Images"
# Set the path to the folder where non-1:1 or unreadable images will be moved.
end_folder = r"J:\New file\Danbooru2004\Images\new"

# Set the number of worker threads for ALL concurrent operations (analysis, moving, deleting).
# The optimal number depends on your disk.
# - For a mechanical HDD, start with 4-8.
# - For a fast SSD (NVMe), you can often increase this to 16 or even 32.
# Experimentation is key to finding the sweet spot for your specific hardware.
MAX_WORKERS = 20

# --- Optional Feature ---
# Set to True to delete files from start_folder if they already exist in end_folder.
# Set to False to disable this feature.
DELETE_DUPLICATES = True  # Default is False for safety
# --- End of Configuration ---

# A set of common image file extensions for faster checking.
image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

# Global event to signal threads to stop gracefully
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handle Ctrl+C by setting the stop_event."""
    print("\n--- Stop signal received! ---")
    print("Finishing current tasks. The script will exit when they are complete.")
    print("Please wait to avoid data corruption...")
    stop_event.set()

# CHANGED: Added exif_error_files and exif_lock to the function signature
def analyze_file(filename, start_folder, end_folder, existing_end_folder_files, exif_error_files, exif_lock):
    """
    Analyzes a single file. Moves files that are not 1:1, are unreadable, or are excessively large.
    Logs files with corrupt EXIF data.
    Returns a tuple describing the action ('move' or 'delete') or None if no action is needed.
    """
    if stop_event.is_set(): return None
    
    file_base, file_ext = os.path.splitext(filename)
    if file_ext.lower() not in image_extensions:
        return None

    image_path = os.path.join(start_folder, filename)
    json_path = os.path.join(start_folder, file_base + '.json')

    # --- Duplicate Check ---
    if DELETE_DUPLICATES and filename in existing_end_folder_files:
        return ('delete', (image_path, json_path))

    # --- Aspect Ratio, Readability & Size Check ---
    if not os.path.isfile(image_path):
        return None

    try:
        # ADDED: Context manager to catch and inspect warnings without stopping execution.
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always") # Capture all warnings within this block

            with Image.open(image_path) as img:
                img.load()  # This can trigger warnings (EXIF) or errors (DecompressionBomb)
                width, height = img.size

            # ADDED: Inspect the warnings we caught after the image is loaded.
            is_decompression_bomb = False
            for w in caught_warnings:
                # Check for decompression bombs, which are treated as a reason to move.
                if issubclass(w.category, Image.DecompressionBombWarning):
                    is_decompression_bomb = True
                    print(f"INFO: Marking {filename} for move (Decompression Bomb).")
                    break # This is the most critical issue, no need to check other warnings.
                
                # Check for EXIF data warnings.
                elif issubclass(w.category, UserWarning) and 'corrupt exif' in str(w.message).lower():
                    print(f"INFO: Corrupt EXIF data detected in {filename}")
                    with exif_lock:
                        # Add the file to our global list for the final report.
                        if filename not in exif_error_files:
                             exif_error_files.append(filename)
                    # Don't break, as a decompression bomb is a more severe issue.
        
        # Now decide on the action based on what was found.
        if is_decompression_bomb:
            pass # A bomb warning means we automatically move it, so we fall through.
        elif width == height:
            return None # If it's a perfect square and not a bomb, we're done.
    
    except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
        print(f"INFO: Marking {filename} for move (corrupted/unreadable). Reason: {e}")
        pass

    except Exception as e:
        print(f"WARNING: An unexpected error occurred analyzing {filename}: {e}. Skipping file.")
        return None
    
    # --- Move Logic ---
    # This logic is reached if the image is:
    # 1. A decompression bomb.
    # 2. Readable but not 1:1 aspect ratio.
    # 3. Corrupted or unreadable (caught by the except block).
    dest_image_path = os.path.join(end_folder, filename)
    dest_json_path = os.path.join(end_folder, os.path.basename(json_path))
    return ('move', (image_path, dest_image_path, json_path, dest_json_path))

def move_item(paths):
    """
    Helper function to move an image and its JSON file atomically.
    If the JSON move fails, it will attempt to move the image back.
    """
    if stop_event.is_set(): return (False, "Operation cancelled.")
    image_src, image_dest, json_src, json_dest = paths

    json_exists = os.path.exists(json_src)

    try:
        shutil.move(image_src, image_dest)

        if json_exists:
            try:
                shutil.move(json_src, json_dest)
            except Exception:
                print(f"\nWARNING: Moved image {os.path.basename(image_src)} but failed to move its JSON. Moving image back...")
                try:
                    shutil.move(image_dest, image_src)
                    return (False, f"SKIPPED: Failed to move JSON for {os.path.basename(image_src)}. Image moved back.")
                except Exception as e_revert:
                    return (False, f"CATASTROPHIC FAILURE: Failed to move JSON and failed to move image back: {e_revert}. MANUAL INTERVENTION REQUIRED.")
        
        return (True, None)

    except OSError as e_img:
        return (False, f"FAILED to move {os.path.basename(image_src)}: {e_img}")
    except Exception as e:
        return (False, f"An unexpected error occurred moving {os.path.basename(image_src)}: {e}")


def delete_item(paths):
    """Helper function to delete an image and its JSON file."""
    if stop_event.is_set(): return (False, "Operation cancelled.")
    image_path, json_path = paths
    try:
        os.remove(image_path)
        if os.path.exists(json_path):
            os.remove(json_path)
        return (True, None)
    except OSError as e:
        return (False, f"FAILED to delete {os.path.basename(image_path)}: {e}")

def execute_concurrently(operation_func, items_to_process, action_name):
    """
    Executes a given file operation (move or delete) concurrently.
    Returns the number of successful operations.
    """
    if not items_to_process or stop_event.is_set():
        return 0
    
    success_count = 0
    total_items = len(items_to_process)
    
    print(f"--- {action_name.capitalize()}ing up to {total_items} items concurrently with {MAX_WORKERS} workers ---")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(operation_func, item): item for item in items_to_process}
        
        processed_count = 0
        for future in concurrent.futures.as_completed(future_to_path):
            if stop_event.is_set():
                for f in future_to_path:
                    if not f.done(): f.cancel()
                break

            processed_count += 1
            try:
                success, error_msg = future.result()
                if success:
                    success_count += 1
                elif error_msg and "cancelled" not in error_msg.lower():
                    if "SKIPPED" in error_msg or "CATASTROPHIC" in error_msg:
                        print(f"\n{error_msg}")
                    else:
                        print(error_msg)
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                path = future_to_path[future]
                print(f"An unexpected error occurred processing item {os.path.basename(path[0])}: {e}")
            
            if processed_count % 1000 == 0 or processed_count == total_items:
                print(f"Progress: {action_name}d {processed_count}/{total_items} items...", end='\r')

    print(f"\n--- Successfully {action_name}d {success_count}/{processed_count} items before stopping/finishing. ---")
    return success_count

def main():
    """Main function to set up and run the processing."""
    start_time = time.time()
    print("--- Script Starting --- (Press Ctrl+C to stop gracefully)")
    print(f"Configuration: DELETE_DUPLICATES={DELETE_DUPLICATES}, MAX_WORKERS={MAX_WORKERS}")
    
    # ADDED: A thread-safe list to store filenames with EXIF errors for the final report.
    exif_error_files = []
    exif_lock = threading.Lock()

    # Set up folders
    if not os.path.exists(end_folder):
        os.makedirs(end_folder)

    # Get file lists
    try:
        print("Scanning source folder to get file list...")
        with os.scandir(start_folder) as it:
            all_files = [entry.name for entry in it if entry.is_file()]
        total_files = len(all_files)
        print(f"Found {total_files} total files to analyze.")
    except FileNotFoundError:
        print(f"Error: The start folder was not found: {start_folder}")
        return

    existing_files_in_end = set()
    if DELETE_DUPLICATES:
        try:
            print("Scanning destination folder for duplicate check...")
            with os.scandir(end_folder) as it:
                existing_files_in_end = {entry.name for entry in it if entry.is_file()}
            print(f"Duplicate check is ON. Found {len(existing_files_in_end)} files in the destination folder.")
        except FileNotFoundError:
            pass
    
    # --- 1. ANALYSIS PHASE ---
    print(f"\n--- Starting Analysis Phase ---")
    analysis_start_time = time.time()
    files_to_move = []
    files_to_delete = []
    
    # CHANGED: Pass the new list and lock to the analysis function.
    func = partial(analyze_file, start_folder=start_folder, end_folder=end_folder, 
                   existing_end_folder_files=existing_files_in_end, 
                   exif_error_files=exif_error_files, exif_lock=exif_lock)

    processed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        try:
            future_to_file = {executor.submit(func, f): f for f in all_files}
            for future in concurrent.futures.as_completed(future_to_file):
                if stop_event.is_set():
                    for f in future_to_file:
                        if not f.done(): f.cancel()
                    break

                processed_count += 1
                try:
                    result = future.result()
                    if result:
                        action, data = result
                        if action == 'move':
                            files_to_move.append(data)
                        elif action == 'delete':
                            files_to_delete.append(data)
                except concurrent.futures.CancelledError:
                    pass
                except Exception as exc:
                     print(f'\n{future_to_file[future]} generated an exception: {exc}')
                
                if processed_count % 1000 == 0:
                    print(f"Analyzed {processed_count}/{total_files}... Found {len(files_to_delete)} to delete, {len(files_to_move)} to move.", end='\r')

        except KeyboardInterrupt:
            stop_event.set()
            print("\nForce-stopping analysis phase...")
    
    analysis_duration = time.time() - analysis_start_time
    print(f"\nAnalysis completed on {processed_count}/{total_files} files in {analysis_duration:.2f} seconds.")
    print(f"Found {len(files_to_delete)} total files to delete.")
    print(f"Found {len(files_to_move)} total files to move (non 1:1, unreadable, or too large).")

    # --- 2. EXECUTION PHASE ---
    total_deleted = 0
    total_moved = 0
    if not stop_event.is_set():
        print(f"\n--- Starting Execution Phase ---")
        if DELETE_DUPLICATES and files_to_delete:
            total_deleted = execute_concurrently(delete_item, files_to_delete, "delete")
        
        if not stop_event.is_set() and files_to_move:
            total_moved = execute_concurrently(move_item, files_to_move, "move")

    # --- 3. FINAL SUMMARY ---
    end_time = time.time()
    print("\n--- All Operations Processed ---")
    if stop_event.is_set():
        print("NOTE: The script was stopped prematurely by the user.")
    if DELETE_DUPLICATES:
        print(f"Total duplicates deleted in this session: {total_deleted}")
    print(f"Total images moved in this session: {total_moved}")
    print(f"Script ran for {(end_time - start_time):.2f} seconds.")

    # ADDED: Final report for files that had EXIF data corruption.
    if exif_error_files:
        print(f"\n--- Found {len(exif_error_files)} File(s) with Corrupt EXIF Data ---")
        for f_name in sorted(exif_error_files): # Sort the list for easier reading
            print(f_name)
    
    if stop_event.is_set():
        sys.exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()