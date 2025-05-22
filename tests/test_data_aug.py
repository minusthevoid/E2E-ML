import os
import numpy as np
from skimage import io
from e2e_ml import augment


def test_augment_image_shape_dtype(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img_path = tmp_path / "img.png"
    io.imsave(img_path.as_posix(), img)
    augment.augment_images(tmp_path.as_posix())
    out_path = tmp_path / "augmented_img.png"
    assert out_path.exists()
    aug_img = io.imread(out_path.as_posix())
    assert aug_img.shape == img.shape
    assert aug_img.dtype == img.dtype
