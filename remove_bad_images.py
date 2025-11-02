from PIL import Image
import os

# Absolute path to your dataset
dataset_root = r"C:\uni\demo\forest-fire-project\data\forest_fire_images"

# Walk through all subfolders
for root, _, files in os.walk(dataset_root):
    for file in files:
        # Check only image files
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify that the image is not corrupted
            except Exception:
                print(f"Bad image detected and removed: {file_path}")
                os.remove(file_path)

print("Scan complete. All corrupted images removed.")
