import os
import shutil

# Paths
dataset_root = "./data/Data"  # original dataset folder
new_root = "./data/forest_fire_images"  # new structure

# Define new folders
train_new = os.path.join(new_root, "train")
val_new = os.path.join(new_root, "val")

# Original folders
train_orig = os.path.join(dataset_root, "Train_Data")
test_orig = os.path.join(dataset_root, "Test_Data")

# Create new folders
for folder in [train_new, val_new]:
    for cls in ["Fire", "Non_Fire"]:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

# Function to move images
def move_images(src, dst):
    for cls in ["Fire", "Non_Fire"]:
        src_cls = os.path.join(src, cls)
        dst_cls = os.path.join(dst, cls)
        for file in os.listdir(src_cls):
            shutil.move(os.path.join(src_cls, file), dst_cls)

# Move train images
move_images(train_orig, train_new)

# Move test images → val
move_images(test_orig, val_new)

print("Dataset restructuring complete!")
print(f"New structure is under: {new_root}")
