"""Directional feather weights, graph-cut terminals and seam smoothing."""

import numpy as np
import pytest

from multiband_blending import ImageOrigin, blend
from multiband_blending._graphcut import build_terminal_labels
from multiband_blending.geometry import calculate_overlap
from multiband_blending.masks import create_mask


@pytest.mark.parametrize("method", ["linear", "cosine"])
@pytest.mark.parametrize("offset", [(0, 2), (0, -2), (2, 0), (-2, 0), (2, 2), (-2, -2)])
def test_feather_mask_matches_directional_projection(method, offset):
    image_a = np.zeros((5, 7), dtype=np.float32)
    origin_a = ImageOrigin()
    origin_b = ImageOrigin(*offset)
    overlap = calculate_overlap(image_a.shape, image_a.shape, origin_a, origin_b)
    actual = create_mask(image_a, image_a, origin_a, origin_b, overlap, method, 5)
    height, width = overlap.overlap_shape
    expected = np.empty((height, width), dtype=np.float32)
    corners = [
        0,
        (height - 1) * offset[0],
        (width - 1) * offset[1],
        (height - 1) * offset[0] + (width - 1) * offset[1],
    ]
    low, high = min(corners), max(corners)
    for row in range(height):
        for column in range(width):
            position = (row * offset[0] + column * offset[1] - low) / (high - low)
            expected[row, column] = (
                1 - position
                if method == "linear"
                else (1 + np.cos(np.pi * position)) / 2
            )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("method", ["linear", "cosine"])
def test_single_pixel_transition_uses_equal_weights(method):
    image = np.zeros((5, 7), dtype=np.float32)
    image_a, image_b = ImageOrigin(), ImageOrigin(x=6)
    overlap = calculate_overlap(image.shape, image.shape, image_a, image_b)
    actual = create_mask(image, image, image_a, image_b, overlap, method, 5)
    np.testing.assert_allclose(actual, 0.5, atol=1e-7)


@pytest.mark.parametrize(
    "offset,expected",
    [
        ((0, 2), [[1, 0, -1]] * 5),
        ((2, 0), [[1] * 5, [0] * 5, [-1] * 5]),
        ((2, 2), [[1, 1, 0], [1, 0, -1], [0, -1, -1]]),
        ((-2, -2), [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
    ],
)
def test_terminal_labels_match_exact_geometry(offset, expected):
    origin_a, origin_b = ImageOrigin(), ImageOrigin(*offset)
    overlap = calculate_overlap((5, 5), (5, 5), origin_a, origin_b)
    labels = build_terminal_labels((5, 5), (5, 5), origin_a, origin_b, overlap)
    assert labels.dtype == np.int8
    assert labels.flags.c_contiguous
    np.testing.assert_array_equal(labels, expected)


def test_graphcut_smoothing_matches_separable_binomial_kernel():
    image_a = np.zeros((5, 7), dtype=np.float32)
    image_b = np.ones_like(image_a)
    origins = ImageOrigin(), ImageOrigin(x=2)
    overlap = calculate_overlap(image_a.shape, image_b.shape, *origins)
    binary = create_mask(image_a, image_b, *origins, overlap, "graphcut", 1)
    actual = create_mask(image_a, image_b, *origins, overlap, "graphcut", 3)
    padded = np.pad(binary, 1, mode="edge")
    expected = np.zeros_like(binary)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    for row in range(binary.shape[0]):
        for column in range(binary.shape[1]):
            expected[row, column] = np.sum(
                padded[row : row + 3, column : column + 3] * kernel
            )
    np.testing.assert_array_equal(actual, expected)


def test_graphcut_kernel_one_returns_binary_mask():
    image_a = np.zeros((5, 7), dtype=np.float32)
    image_b = np.ones((5, 7), dtype=np.float32)
    result = blend(image_a, image_b, origin_b=ImageOrigin(x=2), mask_kernel_size=1)
    assert set(np.unique(result.overlap_mask)) <= {0, 1}
