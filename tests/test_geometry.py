"""Image coordinates, overlap slices, and canvas geometry."""

import numpy as np

from multiband_blending import ImageOrigin
from multiband_blending.geometry import calculate_overlap


def test_numpy_integer_origin_is_supported():
    assert ImageOrigin(np.int64(-2), np.int32(3)) == ImageOrigin(-2, 3)


def test_overlap_with_negative_origins_has_exact_coordinates():
    result = calculate_overlap(
        image_a_shape=(5, 7),
        image_b_shape=(4, 6),
        origin_a=ImageOrigin(-2, -3),
        origin_b=ImageOrigin(0, 1),
    )
    assert result.canvas_origin == ImageOrigin(-2, -3)
    assert result.canvas_shape == (6, 10)
    assert result.overlap_origin == ImageOrigin(0, 1)
    assert result.overlap_shape == (3, 3)
    assert result.image_a_slice == (slice(2, 5), slice(4, 7))
    assert result.image_b_slice == (slice(0, 3), slice(0, 3))
    assert result.canvas_slice == (slice(2, 5), slice(4, 7))
