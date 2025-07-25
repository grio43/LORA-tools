import cv2
import os
import glob
import sys

# ---------- USER SETTINGS ---------------------------------
FOLDER = r"C:\path\to\your\images" # ↖ change this
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"]
OVERWRITE = True # False ➜ writes to /cropped sub‑folder
# ----------------------------------------------------------

def list_files(folder, patterns):
 """Returns a sorted list of files matching the patterns in the given folder."""
 files = []
 for p in patterns:
  files.extend(glob.glob(os.path.join(folder, p)))
 files.sort()
 return files

def ensure_square(r):
 """Resizes the ROI (x,y,w,h) to the largest possible square anchored on its centre."""
 x, y, w, h = r
 side = min(w, h)
 cx, cy = x + w // 2, y + h // 2
 x_new = max(cx - side // 2, 0)
 y_new = max(cy - side // 2, 0)
 return int(x_new), int(y_new), int(side), int(side)

def main():
 """Main function to iterate through images, select ROI, preview, and save."""
 files = list_files(FOLDER, EXTENSIONS)
 if not files:
  print("No images found – check your FOLDER path or EXTENSIONS list.")
  return

 idx = 0
 while idx < len(files):
  fpath = files[idx]
  img = cv2.imread(fpath)
  if img is None:
   print(f"Skipping unreadable file: {fpath}")
   idx += 1
   continue

  # Skip images that are already square
  h, w, _ = img.shape
  if h == w:
   print(f"Skipping already square image: {os.path.basename(fpath)}")
   idx += 1
   continue

  print(f"\n[{idx + 1}/{len(files)}] {os.path.basename(fpath)}")
  print(" Draw a rectangle to select crop area, then press ENTER or SPACE.")
  print(" Press 'c' to cancel selection for this image.")

  # Allow user to select a Region of Interest (ROI)
  roi = cv2.selectROI("Crop (draw, then press Enter)", img, fromCenter=False, showCrosshair=False)

  # roi is (x, y, w, h). If 'c' is pressed or window is closed, all are 0.
  if sum(roi) == 0:
   cv2.destroyWindow("Crop (draw, then press Enter)")
   key = input("❓ Selection cancelled. (s)kip this image / (r)edo / (q)uit program: ").lower()
   if key.startswith("q"):
    break # exit the main while loop
   elif key.startswith("r"):
    continue # restart the loop for the same image (same idx)
   else:
    idx += 1 # skip to the next image
    continue

  # Convert selection to a square and get the cropped image
  x, y, w, h = ensure_square(roi)
  cropped = img[y:y+h, x:x+w]

  # Destroy the selection window as we are done with it
  cv2.destroyWindow("Crop (draw, then press Enter)")

  # --- PREVIEW AND CONFIRMATION ---
  # NEW: Defined a constant for the window name and added the new hotkey
  PREVIEW_WINDOW_NAME = "Preview – Y:Save | R:Redo | Shift+T:Delete | Q:Quit"
  cv2.imshow(PREVIEW_WINDOW_NAME, cropped)

  while True:
   # waitKey returns the ASCII value of the key pressed.
   # For Shift+T, the value is for the uppercase 'T'.
   key = cv2.waitKey(0)

   # --- NEW: Hotkey to delete the image and its JSON file ---
   # Note: We check for ord('T') which corresponds to Shift + 't'
   if key == ord('T'):
    print(" Deleting files...")
    # Construct the corresponding JSON file path
    json_path = os.path.splitext(fpath)[0] + ".json"

    # Delete the image file
    try:
     os.remove(fpath)
     print(f"🗑️ Deleted image: {os.path.basename(fpath)}")
    except OSError as e:
     print(f"❌ Error deleting image {fpath}: {e}")

    # Delete the corresponding JSON file, if it exists
    try:
     os.remove(json_path)
     print(f"🗑️ Deleted json:  {os.path.basename(json_path)}")
    except FileNotFoundError:
     pass # This is expected if there is no JSON file
    except OSError as e:
     print(f"❌ Error deleting json {json_path}: {e}")

    cv2.destroyWindow(PREVIEW_WINDOW_NAME)
    idx += 1 # Move to the next image
    break # Exit preview loop

   # For other keys, use a mask for better cross-platform compatibility
   key_masked = key & 0xFF

   if key_masked in (ord('y'), 13): # 'y' or Enter key
    if OVERWRITE:
     out_path = fpath
    else:
     out_dir = os.path.join(FOLDER, "cropped")
     os.makedirs(out_dir, exist_ok=True)
     out_path = os.path.join(out_dir, os.path.basename(fpath))

    # Set save parameters for maximum quality
    ext = os.path.splitext(out_path)[1].lower()
    params = []
    if ext in ['.jpg', '.jpeg']:
     params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif ext == '.png':
     params = [cv2.IMWRITE_PNG_COMPRESSION, 0]

    cv2.imwrite(out_path, cropped, params)
    print(f"✅ Saved: {out_path}")
    cv2.destroyWindow(PREVIEW_WINDOW_NAME)
    idx += 1 # Move to the next image
    break # Exit preview loop

   elif key_masked == ord('r'): # 'r' key
    print(" Redoing crop for this image.")
    cv2.destroyWindow(PREVIEW_WINDOW_NAME)
    # Do not increment idx, just break the inner loop to restart on the same image
    break

   elif key_masked == ord('q'): # 'q' key
    cv2.destroyAllWindows()
    print("\nQuitting program.")
    return # Exit the main function entirely

  # If user presses 'r', the outer 'while idx < len(files)' loop will repeat for the same idx
  # If user presses 'y' or 'T', idx is incremented, and the loop will proceed to the next file

 print("\nAll done.")

if __name__ == "__main__":
 try:
  main()
 except KeyboardInterrupt:
  sys.exit("\nInterrupted by user.")
 except cv2.error as e:
  print(f"\nAn OpenCV error occurred: {e}")
  print("This can happen if you close a window while the script expects input.")
  sys.exit("Exiting.")