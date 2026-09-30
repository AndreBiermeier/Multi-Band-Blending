"""Collar bounds, edge extension and source selection around the overlap."""

import numpy as np

from multiband_blending import ImageOrigin
from multiband_blending._collar import (
    CollaredOverlap,
    calculate_collared_overlap,
    extend_image_to_collar,
    extend_mask_to_collar,
)
from multiband_blending.geometry import calculate_overlap


def test_one_level_needs_only_the_overlap():
    overlap = calculate_overlap(
        (20, 40), (20, 40), ImageOrigin(-2, -4), ImageOrigin(-2, 26)
    )
    region = calculate_collared_overlap(overlap, levels=1, kernel_size=5)
    assert region.origin == overlap.overlap_origin
    assert region.shape == overlap.overlap_shape
    assert region.overlap_slice == (slice(0, 20), slice(0, 10))


def test_multilevel_collar_is_clipped_and_aligned_to_canvas():
    overlap = calculate_overlap(
        (20, 40), (20, 40), ImageOrigin(-2, -4), ImageOrigin(-2, 26)
    )
    region = calculate_collared_overlap(overlap, levels=2, kernel_size=5)
    assert region.origin == ImageOrigin(-2, 20)
    assert region.shape == (20, 22)
    assert region.overlap_slice == (slice(0, 20), slice(6, 16))


def test_collar_image_replicates_edges_without_zero_padding():
    image = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    region = CollaredOverlap(ImageOrigin(-1, -1), (4, 4), (slice(1, 3), slice(1, 3)))
    actual = extend_image_to_collar(image, ImageOrigin(), region)
    expected = np.array(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.1, 0.1, 0.2, 0.2],
            [0.3, 0.3, 0.4, 0.4],
            [0.3, 0.3, 0.4, 0.4],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_collar_mask_selects_source_regions_and_keeps_overlap_weights():
    origin_a, origin_b = ImageOrigin(), ImageOrigin(2, 2)
    overlap = calculate_overlap((3, 3), (3, 3), origin_a, origin_b)
    region = calculate_collared_overlap(overlap, levels=2, kernel_size=3)
    mask = np.array([[0.25]], dtype=np.float32)
    actual = extend_mask_to_collar(
        mask, (3, 3), (3, 3), origin_a, origin_b, overlap, region
    )
    expected = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0.25, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)
