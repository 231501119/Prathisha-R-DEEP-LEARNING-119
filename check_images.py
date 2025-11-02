from PIL import Image
import os

dataset_dir = "./data/forest_fire_images/train"

for label in ["Fire", "Non_Fire"]:
    folder = os.path.join(dataset_dir, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = Image.open(img_path)
            img.verify()  # verify that it is an image
        except Exception as e:
            print("Bad image:", img_path, e)
