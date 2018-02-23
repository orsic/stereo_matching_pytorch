import numpy as np
from PIL import Image as pimg
from pathlib import Path


def open_image(path: Path, *, gray=False):
    image = pimg.open(path)
    if gray:
        image = image.convert(mode='L')
    return np.array(image)


def load_disparity(path: Path):
    disparity = np.array(pimg.open(path)) / 256.0
    assert disparity is not None
    if len(disparity.shape) > 2:
        disparity = disparity[..., 0]
    return disparity.astype('float32')
