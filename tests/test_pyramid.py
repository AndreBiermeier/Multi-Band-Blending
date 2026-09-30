"""Numerical regressions for the binomial Gaussian/Laplacian pyramid."""

from math import comb

import numpy as np
import pytest

from multiband_blending._blending import blend_opaque_images
from multiband_blending._pyramid import maximum_levels


def reference_blur(image, kernel_size):
    """Apply the binomial filter with explicit edge-padded convolution."""
    weights = np.array([comb(kernel_size - 1, i) for i in range(kernel_size)])
    weights = weights / weights.sum()
    radius = kernel_size // 2
    result = image.astype(np.float64)
    for axis in (0, 1):
        padding = [(0, 0)] * image.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="edge")
        filtered = np.zeros_like(result)
        for index, weight in enumerate(weights):
            selection = [slice(None)] * image.ndim
            selection[axis] = slice(index, index + image.shape[axis])
            filtered += weight * padded[tuple(selection)]
        result = filtered
    return result


def reference_blend(image_a, image_b, mask, levels, kernel_size):
    """Build and blend pyramids independently of the package helpers."""

    def expand(image, target):
        repeated = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
        return reference_blur(
            repeated[: target.shape[0], : target.shape[1]], kernel_size
        )

    def gaussian(image):
        result = [image.astype(np.float64)]
        for _ in range(levels - 1):
            result.append(reference_blur(result[-1], kernel_size)[::2, ::2])
        return result

    gaussian_a = gaussian(image_a)
    gaussian_b = gaussian(image_b)
    masks = gaussian(mask)
    blended = []
    for index, (band_a, band_b, weight) in enumerate(
        zip(gaussian_a, gaussian_b, masks, strict=True)
    ):
        if index < levels - 1:
            band_a = band_a - expand(gaussian_a[index + 1], band_a)
            band_b = band_b - expand(gaussian_b[index + 1], band_b)
        if image_a.ndim == 3:
            weight = weight[..., None]
        blended.append(weight * band_a + (1 - weight) * band_b)
    result = blended[-1]
    for detail in reversed(blended[:-1]):
        result = expand(result, detail) + detail
    return np.clip(result, 0, 1).astype(np.float32)


def test_fixed_two_level_regression():
    image_a = np.array([[0, 0.2, 0.4], [0.6, 0.8, 1], [0.1, 0.3, 0.5]], np.float32)
    image_b = np.flip(image_a, axis=(0, 1)).copy()
    mask = np.array([[1, 0.75, 0.5], [0.75, 0.5, 0.25], [0.5, 0.25, 0]], np.float32)
    expected = np.array(
        [
            [0.050506562, 0.19725953, 0.2171936],
            [0.695697, 0.74430543, 0.6756165],
            [0.21719359, 0.16546021, 0.00289917],
        ],
        dtype=np.float32,
    )
    actual = blend_opaque_images(image_a, image_b, mask, levels=2, kernel_size=5)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 11])
@pytest.mark.parametrize("channels", [None, 1, 3])
def test_odd_sized_multilevel_result_matches_independent_reference(
    kernel_size, channels
):
    rng = np.random.default_rng(2026)
    shape = (5, 7) if channels is None else (5, 7, channels)
    image_a = rng.random(shape, dtype=np.float32)
    image_b = rng.random(shape, dtype=np.float32)
    mask = rng.random((5, 7), dtype=np.float32)
    actual = blend_opaque_images(
        image_a, image_b, mask, levels=3, kernel_size=kernel_size
    )
    expected = reference_blend(
        image_a, image_b, mask, levels=3, kernel_size=kernel_size
    )
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-7)


@pytest.mark.parametrize("selection", [0, 1])
def test_constant_mask_reconstructs_selected_image(selection):
    rng = np.random.default_rng(12)
    image_a = rng.random((5, 7, 3), dtype=np.float32)
    image_b = rng.random((5, 7, 3), dtype=np.float32)
    mask = np.full((5, 7), selection, dtype=np.float32)
    actual = blend_opaque_images(image_a, image_b, mask, levels=4, kernel_size=5)
    np.testing.assert_allclose(actual, image_a if selection else image_b, atol=2e-7)


def test_one_level_is_pointwise_weighted_blend():
    image_a = np.full((2, 5, 3), [0.9, 0.6, 0.3], dtype=np.float32)
    image_b = np.full((2, 5, 3), [0.1, 0.2, 0.7], dtype=np.float32)
    mask = np.broadcast_to(np.linspace(0, 1, 5, dtype=np.float32), (2, 5))
    expected = mask[..., None] * image_a + (1 - mask[..., None]) * image_b
    actual = blend_opaque_images(image_a, image_b, mask, levels=1, kernel_size=5)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    "shape,expected",
    [((1, 9), 1), ((2, 20), 2), ((3, 8), 3), ((5, 7), 4), ((16, 16), 5), ((17, 24), 6)],
)
def test_maximum_levels_uses_ceil_halving(shape, expected):
    assert maximum_levels(shape) == expected
