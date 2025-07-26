import cv2
import os
import glob
import sys
import shutil

# ---------- USER SETTINGS ---------------------------------
# 💾 Place the images you want to crop in this folder.
SOURCE_FOLDER = r"C:\path\to\your\images" # ↖ CHANGE THIS

# 📂 Destination for images that have multiple crops.
# Single-cropped images will overwrite the originals in the SOURCE_FOLDER.
MULTI_CROP_FOLDER = r"C:\path\to\your\multi_crop_output" # ↖ CHANGE THIS

# 🖼️ File types to search for.
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"]

# 📏 Minimum width and height for a cropped square.
MIN_SIZE = 1024
# ----------------------------------------------------------

# ---------- HOTKEY TABLE ----------------------------------
#  In the main preview window:
#    'Y' : Save Max Crop - Selects the largest possible centered square.
#    'D' : Draw Custom Crop - Opens a new window to draw a custom ROI.
#    'M' : Toggle Multi-Crop - Switches between saving one crop (overwrite) or
#          multiple new crops (saved to MULTI_CROP_FOLDER).
#    'S' : Next Image - Finishes with the current image and moves to the next.
#    'Q' : Quit Program - Exits the script immediately.
#
#  When drawing a custom crop (after pressing 'D'):
#    ENTER / SPACE : Confirm the drawn rectangle.
#    'C' : Cancel drawing and return to the main preview.
# ----------------------------------------------------------


def list_files(folder, patterns):
    """Returns a sorted list of files matching the patterns in the given folder."""
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files.sort()
    return files

def get_max_square_crop(img_shape):
    """Calculates the largest possible centered square crop for a given image shape."""
    h, w = img_shape[:2]
    side = min(h, w)
    x = (w - side) // 2
    y = (h - side) // 2
    return x, y, side, side

def draw_text(img, text, y_pos=30, color=(255, 255, 0)):
    """Draws text with a background on an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (5, y_pos - text_h - 5), (10 + text_w, y_pos + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (10, y_pos), font, font_scale, color, thickness)

def main():
    """Main function to iterate through images, select ROIs, and save based on workflow."""
    files = list_files(SOURCE_FOLDER, EXTENSIONS)
    if not files:
        print(f"❌ No images found – check your SOURCE_FOLDER path or EXTENSIONS list.\nSOURCE_FOLDER: {SOURCE_FOLDER}")
        return

    # Create the destination for multi-cropped images if it doesn't exist
    os.makedirs(MULTI_CROP_FOLDER, exist_ok=True)

    idx = 0
    while idx < len(files):
        fpath = files[idx]
        try:
            img = cv2.imread(fpath)
            if img is None:
                print(f"⚠️ Skipping unreadable file: {fpath}")
                idx += 1
                continue
        except Exception as e:
            print(f"ERROR reading {fpath}: {e}")
            idx += 1
            continue

        h, w, _ = img.shape
        if h == w and h >= MIN_SIZE:
            print(f"ℹ️ Skipping already valid square image: {os.path.basename(fpath)}")
            idx += 1
            continue

        print(f"\n[{idx + 1}/{len(files)}] Processing {os.path.basename(fpath)}")
        
        multi_crop_enabled = False
        collected_rois = []

        # --- Loop for collecting one or more crops on the current image ---
        while True:
            # --- PREVIEW ---
            x, y, s, _ = get_max_square_crop(img.shape)
            preview_img = img.copy()
            cv2.rectangle(preview_img, (x, y), (x + s, y + s), (0, 255, 0), 3)
            
            mode_status = "ON" if multi_crop_enabled else "OFF"
            draw_text(preview_img, f"Y:Save Max | D:Draw | M:Multi-Crop({mode_status}) | S:Next Image | Q:Quit")
            if multi_crop_enabled:
                 draw_text(preview_img, f"Multi-Crop ENABLED ({len(collected_rois)} collected)", y_pos=70, color=(0, 255, 255))

            cv2.imshow("Preview", preview_img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow("Preview")

            roi_to_add = None
            # --- DECISION POINT ---
            if key == ord('m'): # Toggle multi-crop mode
                multi_crop_enabled = not multi_crop_enabled
                status_text = "ENABLED" if multi_crop_enabled else "DISABLED"
                print(f"ℹ️ Multi-Crop mode for this image is now {status_text}.")
                continue 

            elif key == ord('y'): # Select Max Crop
                roi_to_add = (x, y, s, s)

            elif key == ord('d'): # Draw Custom Crop
                selection_window_name = "Draw ROI, then press ENTER or SPACE. 'C' to cancel."
                custom_roi = cv2.selectROI(selection_window_name, img, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow(selection_window_name)

                if sum(custom_roi) == 0:
                    print(" Custom selection cancelled.")
                    continue 
                if min(custom_roi[2:]) < MIN_SIZE:
                    print(f"❌ Selection smaller than {MIN_SIZE}x{MIN_SIZE}. Please try again.")
                    continue
                
                x_c, y_c, w_c, h_c = custom_roi
                side = min(w_c, h_c)
                cx, cy = x_c + w_c // 2, y_c + h_c // 2
                roi_to_add = (cx - side // 2, cy - side // 2, side, side)

            elif key == ord('s'): # Finish with this image and move to the next
                print(" Moving to next image...")
                break 

            elif key == ord('q'): # Quit program
                print("\nQuitting program.")
                # Process any collected crops before quitting
                break
            
            else: # Any other key, re-show the preview
                continue

            # --- COLLECT THE CHOSEN CROP (if any) ---
            if roi_to_add:
                collected_rois.append(roi_to_add)
                print(f"✅ ROI #{len(collected_rois)} collected.")

                if not multi_crop_enabled:
                    break # In single mode, break to save and move to the next image

        # --- SAVE COLLECTED CROPS AFTER FINISHING WITH AN IMAGE ---
        num_crops = len(collected_rois)
        if num_crops == 1:
            print("\nSaving 1 crop and overwriting original...")
            rx, ry, rside, _ = map(int, collected_rois[0])
            cropped_img = img[ry:ry+rside, rx:rx+rside]
            
            # Overwrite original image
            params = [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_PNG_COMPRESSION, 0]
            cv2.imwrite(fpath, cropped_img, params)
            print(f"✅ Overwrote: {os.path.basename(fpath)}")
            print(f"ℹ️ JSON file (if any) remains untouched.")
            
        elif num_crops > 1:
            print(f"\nSaving {num_crops} new multi-crop images to: {MULTI_CROP_FOLDER}")
            base_name, ext = os.path.splitext(os.path.basename(fpath))
            
            for i, roi in enumerate(collected_rois, 1):
                rx, ry, rside, _ = map(int, roi)
                cropped_img = img[ry:ry+rside, rx:rx+rside]
                
                # --- SAVE NEW IMAGE ---
                new_filename = f"{base_name}_{i}{ext}"
                out_path = os.path.join(MULTI_CROP_FOLDER, new_filename)
                params = [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_PNG_COMPRESSION, 0]
                cv2.imwrite(out_path, cropped_img, params)
                print(f"✅ Saved Image: {os.path.basename(out_path)}")

                # --- COPY AND RENAME JSON ---
                original_json_path = os.path.splitext(fpath)[0] + ".json"
                if os.path.exists(original_json_path):
                    new_json_filename = f"{base_name}_{i}.json"
                    new_json_path = os.path.join(MULTI_CROP_FOLDER, new_json_filename)
                    shutil.copy2(original_json_path, new_json_path)
                    print(f"✅ Copied JSON:  {new_json_filename}")

        if key == ord('q'): # Exit outer loop if quit was pressed
            break
            
        idx += 1

    print("\nAll done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
    except cv2.error as e:
        print(f"\nAn OpenCV error occurred: {e}")
        sys.exit("Exiting.")