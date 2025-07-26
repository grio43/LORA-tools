import os
import glob
import concurrent.futures

def delete_file(file_path):
    """
    Deletes a single file and prints the action.
    """
    try:
        os.remove(file_path)
        return f"Removed: {file_path}"
    except OSError as e:
        return f"Error removing {file_path}: {e}"

def clean_dataset(directory_path, num_workers=4):
    """
    Cleans a directory by removing orphaned image and JSON files using multiple workers.

    Args:
        directory_path (str): The path to the directory containing the dataset.
        num_workers (int): The number of worker threads to use for deletion.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    print(f"Scanning directory: {directory_path}")

    image_extensions = ['.png', '.jpg', '.jpeg']
    json_extension = '.json'

    # Use glob to find all relevant files
    all_files = glob.glob(os.path.join(directory_path, '*'))

    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]
    json_files = [f for f in all_files if os.path.splitext(f)[1].lower() == json_extension]

    image_basenames = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
    json_basenames = {os.path.splitext(os.path.basename(f))[0] for f in json_files}

    # Find files to delete
    files_to_delete = []
    for image_file in image_files:
        basename = os.path.splitext(os.path.basename(image_file))[0]
        if basename not in json_basenames:
            files_to_delete.append(image_file)

    for json_file in json_files:
        basename = os.path.splitext(os.path.basename(json_file))[0]
        if basename not in image_basenames:
            files_to_delete.append(json_file)

    if not files_to_delete:
        print("No orphaned files found. Dataset is already clean.")
        return

    print(f"Found {len(files_to_delete)} orphaned files to remove.")

    # Use a ThreadPoolExecutor to delete files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all delete tasks to the executor
        future_to_file = {executor.submit(delete_file, f): f for f in files_to_delete}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            print(result)

if __name__ == "__main__":
    # Hardcoded directory path
    target_directory = r"J:\New file\Danbooru2004\Images"

    # You can adjust the number of workers based on your system's capabilities
    # A good starting point is the number of CPU cores you have.
    number_of_workers = 8 

    clean_dataset(target_directory, num_workers=number_of_workers)
    
    print("\nDataset cleanup process complete.")