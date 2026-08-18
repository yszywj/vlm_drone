from __future__ import annotations

import base64
from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from models.image_encoding import encode_rgb_to_data_url


class RGBImageEncodingTest(unittest.TestCase):
    def decode(self, data_url: str) -> Image.Image:
        prefix = "data:image/jpeg;base64,"
        self.assertTrue(data_url.startswith(prefix))
        payload = base64.b64decode(data_url[len(prefix) :], validate=True)
        image = Image.open(BytesIO(payload))
        image.load()
        return image

    def test_encodes_uint8_rgb_without_disk_io(self) -> None:
        rgb = np.zeros((12, 20, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200

        image = self.decode(
            encode_rgb_to_data_url(rgb, max_side_px=1024, jpeg_quality=80)
        )

        self.assertEqual(image.format, "JPEG")
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (20, 12))
        image.close()

    def test_downscales_while_preserving_aspect_ratio(self) -> None:
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)

        image = self.decode(
            encode_rgb_to_data_url(rgb, max_side_px=50, jpeg_quality=75)
        )

        self.assertEqual(image.size, (50, 25))
        image.close()

    def test_validates_shape_dtype_and_options(self) -> None:
        invalid_arrays = (
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 4), dtype=np.uint8),
            np.zeros((0, 10, 3), dtype=np.uint8),
        )
        for rgb in invalid_arrays:
            with self.subTest(shape=rgb.shape), self.assertRaises(ValueError):
                encode_rgb_to_data_url(rgb)

        with self.assertRaises(TypeError):
            encode_rgb_to_data_url(np.zeros((2, 2, 3), dtype=np.float32))
        with self.assertRaises(TypeError):
            encode_rgb_to_data_url([[[]]])  # type: ignore[arg-type]
        for max_side in (True, 0, -1, 1.5):
            with self.subTest(max_side=max_side), self.assertRaises(
                (TypeError, ValueError)
            ):
                encode_rgb_to_data_url(
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    max_side_px=max_side,  # type: ignore[arg-type]
                )
        for quality in (True, 0, 96, 80.5):
            with self.subTest(quality=quality), self.assertRaises(
                (TypeError, ValueError)
            ):
                encode_rgb_to_data_url(
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    jpeg_quality=quality,  # type: ignore[arg-type]
                )


if __name__ == "__main__":
    unittest.main()
