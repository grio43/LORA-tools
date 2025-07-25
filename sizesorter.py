import os
import shutil
from PIL import Image
import concurrent.futures

# --- Configuration ---
# Set the path to the folder containing your images and JSON files.
start_folder = r"J:\New file\Danbooru2004\Images"
# Set the path to the folder where non-1:1 images will be moved.
end_folder = r"J:\New file\Danbooru2004\Images\new"
# Set the number of worker threads. The optimal number depends on your disk.
# For an HDD, start with 4-8. For a fast SSD, you can try 8-16.
# More is not always better, so you may need to experiment.
MAX_WORKERS = 10
# --- End of Configuration ---

# A set of common image file extensions for faster checking.
image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

def process_file(filename):
    """
    This function contains the logic to process a single file.
    It will be executed by each worker thread.
    """
    file_base, file_ext = os.path.splitext(filename)

    if file_ext.lower() in image_extensions:
        image_path = os.path.join(start_folder, filename)

        if not os.path.isfile(image_path):
            return # Skip if it's a directory or other non-file type

        try:
            with Image.open(image_path) as img:
                width, height = img.size

            if width != height:
                json_filename = file_base + '.json'
                json_path = os.path.join(start_folder, json_filename)

                # Move the image
                try:
                    shutil.move(image_path, os.path.join(end_folder, filename))
                except Exception as e:
                    print(f"FAILED to move image {filename}: {e}")
                    return # Stop processing this file if image move fails

                # Move the JSON
                if os.path.exists(json_path):
                    try:
                        shutil.move(json_path, os.path.join(end_folder, json_filename))
                    except Exception as e:
                        print(f"FAILED to move JSON {json_filename} for image {filename}: {e}")
                else:
                    print(f"MISSING JSON for image: {filename}")

        except (IOError, SyntaxError) as e:
            print(f"FAILED to process image {filename} (corrupted or unreadable): {e}")
        except Exception as e:
            print(f"An unexpected error occurred with file {filename}: {e}")


def main():
    """
    Main function to set up the process.
    """
    # Create the destination folder if it doesn't already exist.
    if not os.path.exists(end_folder):
        try:
            os.makedirs(end_folder)
        except OSError as e:
            print(f"Error creating directory {end_folder}: {e}")
            return # Exit if we can't create the destination

    # CRITICAL STEP: Get the list of all files *before* starting workers.
    # This prevents race conditions.
    try:
        all_files = os.listdir(start_folder)
        print(f"Found {len(all_files)} total items to check.")
    except FileNotFoundError:
        print(f"Error: The start folder was not found: {start_folder}")
        return


    # Use ThreadPoolExecutor to manage worker threads.
    # The 'with' statement ensures threads are cleaned up properly.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 'map' applies the 'process_file' function to every item in 'all_files'.
        # It automatically handles distributing the work and collecting results.
        # We wrap it in list() to ensure we wait for all threads to complete.
        list(executor.map(process_file, all_files))
    
    print("Script finished.")


if __name__ == "__main__":
    main()