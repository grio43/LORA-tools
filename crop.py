import cv2
import os
import glob
import sys
import shutil
import math

# ---------- USER SETTINGS ---------------------------------
# 💾 Place the images you want to crop in this folder.
SOURCE_FOLDER = r"J:\New file\Danbooru2004\Images\new" # ↖ CHANGE THIS

# 📂 Destination for all generated crops.
MULTI_CROP_FOLDER = r"J:\New file\Danbooru2004\Images\new\crop" # ↖ CHANGE THIS

# 🖼️ File types to search for.
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"]

# 📏 Minimum width and height for a cropped square.
MIN_SIZE = 1024

# 📈 Multi-Crop Trigger Factor.
MULTI_CROP_THRESHOLD_FACTOR = 1.1

# 📐 Overlap percentage for sequential crops.
OVERLAP_PERCENT = 0.1

# ----------------------------------------------------------
# The interactive preview has been removed for bulk processing.
# ----------------------------------------------------------

def list_files(folder, patterns):
    """Returns a sorted list of files matching the patterns in the given folder."""
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files.sort()
    return files

def generate_controlled_rois(img_w, img_h, min_size, overlap_percent, threshold_factor):
    """
    Generates a list of regions of interest (ROIs) for cropping based on
    image dimensions and a controlled, overlapping strategy.
    """
    rois = []
    short_side = min(img_w, img_h)
    long_side = max(img_w, img_h)

    side = short_side

    if long_side < short_side * threshold_factor:
        x = (img_w - side) // 2
        y = (img_h - side) // 2
        rois.append((x, y, side, side))
        print(f"  -> Generating 1 centered crop.")
        return rois

    print(f"  -> Generating overlapping crops with {int(overlap_percent*100)}% overlap.")
    step = int(side * (1 - overlap_percent))
    if step == 0: step = 1

    if img_w > img_h: # Wide image
        num_crops = math.ceil((img_w - side) / step)
        for i in range(num_crops):
            x = min(img_w - side, i * step)
            y = 0
            if (x, y, side, side) not in rois:
                rois.append((x, y, side, side))
            if x == img_w - side: break
    else: # Tall or square image
        num_crops = math.ceil((img_h - side) / step)
        for i in range(num_crops):
            x = 0
            y = min(img_h - side, i * step)
            if (x, y, side, side) not in rois:
                rois.append((x, y, side, side))
            if y == img_h - side: break

    return rois

def main():
    """Main function to iterate through images, generate controlled crops, and save them."""
    files = list_files(SOURCE_FOLDER, EXTENSIONS)
    if not files:
        print(f"❌ No images found – check your SOURCE_FOLDER path or EXTENSIONS list.\nSOURCE_FOLDER: {SOURCE_FOLDER}")
        return

    os.makedirs(MULTI_CROP_FOLDER, exist_ok=True)

    for idx, fpath in enumerate(files):
        try:
            base_name_with_ext = os.path.basename(fpath)
            print(f"\n[{idx + 1}/{len(files)}] Processing {base_name_with_ext}")

            img = cv2.imread(fpath)
            if img is None:
                print(f"⚠️ Skipping unreadable file: {fpath}")
                continue

            img_h, img_w, _ = img.shape

            if img_h == img_w and img_h >= MIN_SIZE:
                print(f"  -> Skipping already valid square image.")
                continue

            if min(img_h, img_w) < MIN_SIZE:
                print(f"  -> Skipping image smaller than MIN_SIZE ({min(img_h, img_w)} < {MIN_SIZE}).")
                continue

            # --- Generate ROIs based on the controlled strategy ---
            collected_rois = generate_controlled_rois(img_w, img_h, MIN_SIZE, OVERLAP_PERCENT, MULTI_CROP_THRESHOLD_FACTOR)

            # --- PROCESS FILE (SAVE CROPS) ---
            if collected_rois:
                print(f"  -> Saving {len(collected_rois)} new crop(s)...")
                base_name, ext = os.path.splitext(base_name_with_ext)
                original_json_path = os.path.splitext(fpath)[0] + ".json"

                for i, roi in enumerate(collected_rois, 1):
                    rx, ry, rside, _ = map(int, roi)
                    cropped_img = img[ry:ry+rside, rx:rx+rside]
                    new_filename = f"{base_name}_{i}{ext}"
                    out_path = os.path.join(MULTI_CROP_FOLDER, new_filename)
                    params = [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_PNG_COMPRESSION, 0]
                    cv2.imwrite(out_path, cropped_img, params)

                    if os.path.exists(original_json_path):
                        new_json_filename = f"{base_name}_{i}.json"
                        new_json_path = os.path.join(MULTI_CROP_FOLDER, new_json_filename)
                        shutil.copy2(original_json_path, new_json_path)
                
                # --- SAFELY COMMENTED OUT ---
                # This section is disabled to prevent accidental data loss.
                # If you are certain you want to delete the originals after cropping,
                # you can uncomment the following lines.
                # --------------------------------------------------------------------
                # print(f"  -> Removing original file: {base_name_with_ext}")
                # try:
                #     if os.path.exists(fpath): os.remove(fpath)
                #     if os.path.exists(original_json_path): os.remove(original_json_path)
                #     print(f"  -> Removed Original: {base_name_with_ext}")
                # except OSError as e:
                #     print(f"  -> ❌ Error removing original file: {e}")
                # --------------------------------------------------------------------

        except Exception as e:
            print(f"An unexpected ERROR occurred while processing {fpath}: {e}")
            continue

    print("\nAll done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user. Exiting.")