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
# ==> EDIT THESE PATHS to match your system before running!
# Set the path to the folder containing your images and JSON files.
start_folder = r"J:\New file\Danbooru2004\Images"
# Set the path to the folder where non-1:1 or unreadable images will be moved.
end_folder = r"J:\New file\Danbooru2004\Images\new"

# Set the number of worker threads for ALL concurrent operations (analysis, moving, deleting).
MAX_WORKERS = 6

# --- Optional Feature ---
# Set to True to delete files from start_folder if they already exist in end_folder.
DELETE_DUPLICATES = True
# --- End of Configuration ---

image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handle Ctrl+C by setting the stop_event."""
    print("\n--- Stop signal received! ---")
    print("Finishing current tasks. The script will exit when they are complete.")
    stop_event.set()

def analyze_file(filename, start_folder, end_folder, existing_end_folder_files, exif_error_files, exif_lock):
    """Analyzes a single file and returns an action if necessary."""
    if stop_event.is_set(): return None

    file_base, file_ext = os.path.splitext(filename)
    if file_ext.lower() not in image_extensions:
        return None

    image_path = os.path.join(start_folder, filename)
    json_path = os.path.join(start_folder, file_base + '.json')

    if DELETE_DUPLICATES and filename in existing_end_folder_files:
        return ('delete', (image_path, json_path))

    if not os.path.isfile(image_path):
        return None

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            with Image.open(image_path) as img:
                img.load()
                width, height = img.size

            is_decompression_bomb = False
            for w in caught_warnings:
                if issubclass(w.category, Image.DecompressionBombWarning):
                    is_decompression_bomb = True
                    print(f"\nINFO: Marking {filename} for move (Decompression Bomb).")
                    break
                elif issubclass(w.category, UserWarning) and 'corrupt exif' in str(w.message).lower():
                    print(f"\nINFO: Corrupt EXIF data detected in {filename}")
                    with exif_lock:
                        if filename not in exif_error_files:
                            exif_error_files.append(filename)
        
        # If it's a decompression bomb OR not a 1:1 ratio, mark for moving.
        if is_decompression_bomb or width != height:
            dest_image_path = os.path.join(end_folder, filename)
            dest_json_path = os.path.join(end_folder, os.path.basename(json_path))
            return ('move', (image_path, dest_image_path, json_path, dest_json_path))

    except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
        # Also move files that are corrupted or cannot be read.
        print(f"\nINFO: Marking {filename} for move (corrupted/unreadable). Reason: {e}")
        dest_image_path = os.path.join(end_folder, filename)
        dest_json_path = os.path.join(end_folder, os.path.basename(json_path))
        return ('move', (image_path, dest_image_path, json_path, dest_json_path))
    
    except Exception as e:
        print(f"\nWARNING: An unexpected error occurred analyzing {filename}: {e}. Skipping file.")
    
    return None

def move_item(paths):
    """Helper function to move an image and its JSON file."""
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
                    return (False, f"CATASTROPHIC: Failed to move JSON and revert image: {e_revert}.")
        return (True, None)
    except Exception as e:
        return (False, f"FAILED to move {os.path.basename(image_src)}: {e}")

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
    """Executes a file operation concurrently with progress reporting."""
    if not items_to_process or stop_event.is_set():
        return 0
    
    success_count = 0
    total_items = len(items_to_process)
    print(f"--- {action_name.capitalize()}ing {total_items} items with {MAX_WORKERS} workers ---")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(operation_func, item): item for item in items_to_process}
        
        processed_count = 0
        for future in concurrent.futures.as_completed(future_to_path):
            if stop_event.is_set():
                [f.cancel() for f in future_to_path if not f.done()]
                break

            processed_count += 1
            try:
                success, error_msg = future.result()
                if success:
                    success_count += 1
                elif error_msg:
                     print(f"\n{error_msg}")
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                print(f"\nAn unexpected error occurred processing {future_to_path[future]}: {e}")
            
            if processed_count % 100 == 0 or processed_count == total_items:
                progress = (processed_count / total_items) * 100
                sys.stdout.write(f"\rProgress: {processed_count}/{total_items} ({progress:.1f}%) files {action_name}d...")
                sys.stdout.flush()

    print(f"\n--- Successfully {action_name}d {success_count}/{processed_count} items. ---")
    return success_count

def main():
    """Main function to set up and run the processing."""
    start_time = time.time()
    print("--- Script Starting --- (Press Ctrl+C to stop gracefully)")
    print(f"Configuration: DELETE_DUPLICATES={DELETE_DUPLICATES}, MAX_WORKERS={MAX_WORKERS}")
    
    exif_error_files = []
    exif_lock = threading.Lock()

    if not os.path.exists(end_folder):
        os.makedirs(end_folder)

    try:
        print("Scanning source folder...")
        with os.scandir(start_folder) as it:
            all_files = [entry.name for entry in it if entry.is_file()]
        total_files = len(all_files)
        print(f"Found {total_files} total files to analyze.")
    except FileNotFoundError:
        print(f"Error: Start folder not found: {start_folder}")
        return

    existing_files_in_end = set()
    if DELETE_DUPLICATES:
        try:
            print("Scanning destination folder for duplicates...")
            with os.scandir(end_folder) as it:
                existing_files_in_end = {entry.name for entry in it if entry.is_file()}
            print(f"Duplicate check ON. Found {len(existing_files_in_end)} files in destination.")
        except FileNotFoundError:
            # It's okay if the destination folder doesn't exist yet for the duplicate check
            pass
    
    print(f"\n--- Starting Analysis Phase ---")
    analysis_start_time = time.time()
    files_to_move = []
    files_to_delete = []
    
    func = partial(analyze_file, start_folder=start_folder, end_folder=end_folder, 
                   existing_end_folder_files=existing_files_in_end, 
                   exif_error_files=exif_error_files, exif_lock=exif_lock)

    processed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        try:
            future_to_file = {executor.submit(func, f): f for f in all_files}
            for future in concurrent.futures.as_completed(future_to_file):
                if stop_event.is_set():
                    [f.cancel() for f in future_to_file if not f.done()]
                    break

                processed_count += 1
                try:
                    result = future.result()
                    if result:
                        action, data = result
                        if action == 'move': files_to_move.append(data)
                        elif action == 'delete': files_to_delete.append(data)
                except concurrent.futures.CancelledError:
                    pass
                except Exception as exc:
                     print(f'\n{future_to_file[future]} generated an exception: {exc}')
                
                if processed_count % 100 == 0 or processed_count == total_files:
                    progress = (processed_count / total_files) * 100
                    status_line = (f"\rAnalyzed: {processed_count}/{total_files} ({progress:.1f}%) | "
                                   f"To Delete: {len(files_to_delete)} | To Move: {len(files_to_move)}")
                    sys.stdout.write(status_line)
                    sys.stdout.flush()

        except KeyboardInterrupt:
            stop_event.set()
            print("\nAnalysis phase stopped by user.")
    
    analysis_duration = time.time() - analysis_start_time
    print(f"\nAnalysis completed on {processed_count}/{total_files} files in {analysis_duration:.2f}s.")
    print(f"Found {len(files_to_delete)} files to delete and {len(files_to_move)} files to move.")

    total_deleted, total_moved = 0, 0
    if not stop_event.is_set():
        print(f"\n--- Starting Execution Phase ---")
        if DELETE_DUPLICATES and files_to_delete:
            total_deleted = execute_concurrently(delete_item, files_to_delete, "delete")
        
        if not stop_event.is_set() and files_to_move:
            total_moved = execute_concurrently(move_item, files_to_move, "move")

    end_time = time.time()
    print("\n\n--- All Operations Processed ---")
    if stop_event.is_set():
        print("NOTE: The script was stopped prematurely by the user.")
    if DELETE_DUPLICATES:
        print(f"Total duplicates deleted: {total_deleted}")
    print(f"Total images moved: {total_moved}")
    print(f"Script ran for {(end_time - start_time):.2f} seconds.")

    if exif_error_files:
        print(f"\n--- Found {len(exif_error_files)} File(s) with Corrupt EXIF Data ---")
        for f_name in sorted(exif_error_files):
            print(f_name)
    
    if stop_event.is_set():
        sys.exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()