"""In-memory RGB image preparation for OpenAI-compatible vision requests."""

from __future__ import annotations

import base64
from io import BytesIO
from math import ceil

import numpy as np
from PIL import Image


DEFAULT_MAX_SIDE_PX = 1024
DEFAULT_JPEG_QUALITY = 80


def encode_rgb_to_data_url(
    rgb: np.ndarray,
    *,
    max_side_px: int = DEFAULT_MAX_SIDE_PX,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> str:
    """Encode an RGB uint8 array to an in-memory JPEG data URL.

    Images larger than ``max_side_px`` are downscaled while retaining their
    aspect ratio.  Non-uint8 arrays are rejected instead of being implicitly
    clipped or normalized, because those conversions are ambiguous at a model
    input boundary.
    """

    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a numpy.ndarray")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    height, width, _channels = rgb.shape
    if height <= 0 or width <= 0:
        raise ValueError("rgb height and width must be greater than zero")
    if rgb.dtype != np.uint8:
        raise TypeError("rgb dtype must be uint8")
    if isinstance(max_side_px, bool) or not isinstance(max_side_px, int):
        raise TypeError("max_side_px must be an integer")
    if max_side_px <= 0:
        raise ValueError("max_side_px must be greater than zero")
    if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int):
        raise TypeError("jpeg_quality must be an integer")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be between 1 and 95")

    # ``fromarray`` may retain a view of the caller's mutable memory.  The
    # immediately following save/resize occurs synchronously, but copying here
    # also guarantees a contiguous positive-stride representation.
    image = Image.fromarray(np.ascontiguousarray(rgb))
    largest_side = max(width, height)
    if largest_side > max_side_px:
        scale = max_side_px / largest_side
        # ceil avoids collapsing a very thin valid image to zero pixels.
        resized_width = max(1, min(max_side_px, ceil(width * scale)))
        resized_height = max(1, min(max_side_px, ceil(height * scale)))
        image = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.LANCZOS,
        )

    output = BytesIO()
    try:
        image.save(
            output,
            format="JPEG",
            quality=jpeg_quality,
            optimize=False,
        )
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
    except Exception:
        # Never attach the RGB array or encoded bytes to the public diagnostic.
        raise RuntimeError("failed to encode RGB image as JPEG") from None
    finally:
        output.close()
        image.close()

    return f"data:image/jpeg;base64,{encoded}"


__all__ = [
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_MAX_SIDE_PX",
    "encode_rgb_to_data_url",
]
