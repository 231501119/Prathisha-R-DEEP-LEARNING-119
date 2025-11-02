import os
import shutil
from pathlib import Path
import random

# Source dataset folders
source_train = Path("data/forest_fire_images/train")
source_val = Path("data/forest_fire_images/val")

# Destination folders
train_dest = Path("data/forest_fire_images/train_final")
val_dest = Path("data/forest_fire_images/val_final")

# Train/validation split ratio
split_ratio = 0.2

# Create destination folders
train_dest.mkdir(parents=True, exist_ok=True)
val_dest.mkdir(parents=True, exist_ok=True)

# Get class folders from both train and val
classes = set()
for folder in [source_train, source_val]:
    for d in folder.iterdir():
        if d.is_dir():
            classes.add(d.name)

# Process each class
for class_name in classes:
    # Gather all images from train and val folders
    images = list((source_train / class_name).glob("*.*")) + list((source_val / class_name).glob("*.*"))
    random.shuffle(images)
    
    # Determine split
    num_val = int(len(images) * split_ratio)
    val_images = images[:num_val]
    train_images = images[num_val:]
    
    # Create class subfolders in destination
    (train_dest / class_name).mkdir(parents=True, exist_ok=True)
    (val_dest / class_name).mkdir(parents=True, exist_ok=True)
    
    # Copy images to destination
    for img in train_images:
        shutil.copy(img, train_dest / class_name)
    for img in val_images:
        shutil.copy(img, val_dest / class_name)

# Print summary
print("Dataset merged and split successfully!")
for dest in [train_dest, val_dest]:
    print(f"\n{dest.name}:")
    for class_name in (dest).iterdir():
        if class_name.is_dir():
            print(f"  {class_name.name}: {len(list(class_name.glob('*.*')))} images")
