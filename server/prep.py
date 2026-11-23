"""Optional preparation of an uploaded image before classification.

MNIST digits are white strokes on a black background, centred and filling about 20 of 28 pixels.
A photo or a drawing of a digit is usually dark on light paper, small and off-centre, and the
frozen YOLO11 features of such an image are far from those of the training digits. `digit_like`
turns it into the MNIST look: grayscale, light background made black, digit cropped by its
bounding box, centred on a square with a margin, 28x28.
"""
import base64
import io

import numpy as np
from PIL import Image, ImageOps


def digit_like(img: Image.Image, size=28, box=20) -> Image.Image:
    g = np.asarray(ImageOps.exif_transpose(img).convert("L"), dtype=np.float32)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if np.median(border) > 127:  # light paper: make the strokes light and the background dark
        g = 255 - g
    bg = np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]))
    g = np.clip((g - bg) / max(1.0, g.max() - bg), 0, 1)  # background to 0, strongest stroke to 1
    g[g < 0.25] = 0  # paper texture and shadows
    ys, xs = np.nonzero(g)
    if len(ys) == 0:
        return Image.fromarray((g * 255).astype(np.uint8)).resize((size, size)).convert("RGB")
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    s = box / max(h, w)  # longest side to `box` pixels, as in MNIST
    digit = Image.fromarray((g * 255).astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))),
                                                                Image.LANCZOS)
    out = Image.new("L", (size, size), 0)
    out.paste(digit, ((size - digit.width) // 2, (size - digit.height) // 2))
    return out.convert("RGB")


PREPS = {"digit": digit_like}


def apply(img: Image.Image, prep: str):
    """(image to classify, PNG data URL of it for the page, or None when nothing was changed)."""
    if not prep:
        return img, None
    out = PREPS[prep](img)
    buf = io.BytesIO()
    out.resize((112, 112), Image.NEAREST).save(buf, "PNG")
    return out, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
