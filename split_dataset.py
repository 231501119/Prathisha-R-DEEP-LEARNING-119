import os
import shutil
from pathlib import Path
import random

# Source folders
train_source = Path("data/Data/Train_Data")
test_source = Path("data/Data/Test_Data")

# Destination folders
train_dest = Path("data/forest_fire_images/train")
val_dest = Path("data/forest_fire_images/val")

split_ratio = 0.2

train_dest.mkdir(parents=True, exist_ok=True)
val_dest.mkdir(parents=True, exist_ok=True)

# Function to copy all images recursively
def copy_images(src_folder, dest_folder):
    for img_file in src_folder.rglob("*.*"):  # rglob goes into subfolders
        shutil.copy(img_file, dest_folder)

# Step 1 & 2: Copy images
copy_images(train_source, train_dest)
copy_images(test_source, val_dest)

# Step 3: Shuffle train and move fraction to val
all_train_images = list(train_dest.glob("*.*"))
random.shuffle(all_train_images)
num_val = int(len(all_train_images) * split_ratio)

for img_file in all_train_images[:num_val]:
    shutil.move(str(img_file), val_dest)

print(f"Training images: {len(list(train_dest.glob('*.*')))}")
print(f"Validation images: {len(list(val_dest.glob('*.*')))}")
print("Dataset merged and split successfully!")
