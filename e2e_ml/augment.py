import os
import random
from skimage import io, transform, util, filters, img_as_ubyte


def augment_images(folder: str) -> None:
    """Apply basic augmentation to images in a folder."""
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(folder, fname)
            image = io.imread(path)
            if random.random() < 0.5:
                image = util.random_noise(image)
            if random.random() < 0.5:
                image = filters.gaussian(image)
            direction = random.choice([-90, 90, 45])
            augmented = transform.rotate(image, direction)
            augmented = img_as_ubyte(augmented)
            out_path = os.path.join(folder, f"augmented_{fname}")
            io.imsave(out_path, augmented)
