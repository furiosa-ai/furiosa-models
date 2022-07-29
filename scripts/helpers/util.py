from typing import Any, Callable, ForwardRef, List, Optional, Tuple

from PIL import Image, ImageOps
import numpy as np

from furiosa.registry import Model
from furiosa.runtime import session


def load_image(image_path: str, seq_channel: str="RGB") -> np.array:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert(seq_channel)
    image = np.asarray(image)
    return image
