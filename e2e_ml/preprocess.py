import os
from PIL import Image

def resize_images(folder: str, size=(224, 224)) -> None:
    """Resize images in a folder to a given size and save with a prefix."""
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(folder, fname)
            image = Image.open(path)
            image = image.resize(size)
            out_path = os.path.join(folder, f"resized_{fname}")
            image.save(out_path)
