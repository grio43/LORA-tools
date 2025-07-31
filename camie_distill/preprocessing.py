"""
Image → tensor preprocessing with configurable padding colour.
Transparent RGBA / LA images are flattened on a white background.
"""
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def load_and_preprocess(path: Path,
                        size: int = 512,
                        pad_colour: Tuple[int, int, int] = (0, 0, 0),
                        fp16: bool = False) -> np.ndarray:
    img = Image.open(path)

    # Handle transparency → white composite
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.alpha_composite(img.convert("RGBA"))
        img = background.convert("RGB")
    else:
        img = img.convert("RGB")

    # Resize with aspect‑ratio, then letter‑box
    w, h = img.size
    r = w / h
    nw, nh = (size, int(size / r)) if r > 1 else (int(size * r), size)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (size, size), pad_colour)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))

    arr = np.asarray(canvas,
                     dtype=np.float16 if fp16 else np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))   # CHW
