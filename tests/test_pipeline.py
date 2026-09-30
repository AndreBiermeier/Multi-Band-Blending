"""Public blending results, image layouts, placement, and input preservation."""

import numpy as np
import pytest

from multiband_blending import BlendResult, ImageOrigin, blend


@pytest.mark.parametrize("mask", ["linear", "cosine", "graphcut"])
@pytest.mark.parametrize("channels", [None, 1, 2, 3, 4])
def test_supported_image_layouts(mask, channels):
    shape = (5, 7) if channels is None else (5, 7, channels)
    image = np.full(shape, 128, dtype=np.uint8)
    if channels in (2, 4):
        image[..., -1] = 255
    result = blend(image, image, origin_b=ImageOrigin(x=2), mask=mask)
    expected_channels = 2 if channels in (None, 1, 2) else 4
    assert isinstance(result, BlendResult)
    assert result.blended_image.shape == (5, 9, expected_channels)
    assert result.blended_image.dtype == np.float32
    assert result.overlap_mask.shape == (5, 5)
    assert result.overlap_mask.dtype == np.float32
    assert result.canvas_origin == ImageOrigin()
    np.testing.assert_allclose(result.blended_image[..., :-1], 128 / 255, atol=2e-6)
    np.testing.assert_allclose(result.blended_image[..., -1], 1.0, atol=2e-6)
    assert np.all((result.overlap_mask >= 0) & (result.overlap_mask <= 1))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_unsigned_normalization_is_consistent_across_the_whole_pipeline(dtype):
    value = int(np.iinfo(dtype).max) // 2
    image = np.full((5, 7, 3), value, dtype=dtype)
    result = blend(image, image, origin_b=ImageOrigin(x=2), mask="linear")
    np.testing.assert_allclose(
        result.blended_image[..., :3], value / int(np.iinfo(dtype).max), atol=2e-6
    )


def test_negative_origins_and_diagonal_canvas_holes():
    image_a = np.ones((4, 5), dtype=np.float32)
    result = blend(
        image_a, image_a, ImageOrigin(-2, -3), ImageOrigin(0, 0), mask="linear"
    )
    assert result.canvas_origin == ImageOrigin(-2, -3)
    assert result.blended_image.shape == (6, 8, 2)
    np.testing.assert_array_equal(result.blended_image[:2, 5:], 0)
    np.testing.assert_array_equal(result.blended_image[4:, :3], 0)


def test_strided_inputs_remain_unchanged():
    image_a = np.linspace(0, 1, 12 * 18 * 3, dtype=np.float32).reshape(12, 18, 3)
    original = image_a.copy()
    view = image_a[::2, ::2, ::-1]
    result = blend(view, view, origin_b=ImageOrigin(x=3), mask="linear")
    assert np.isfinite(result.blended_image).all()
    np.testing.assert_array_equal(image_a, original)


@pytest.mark.parametrize("mask", ["linear", "cosine"])
def test_automatic_depth_matches_maximum_explicit_depth(mask):
    image_a = np.linspace(0, 1, 5 * 7, dtype=np.float32).reshape(5, 7)
    image_b = np.flip(image_a, axis=1).copy()
    automatic = blend(
        image_a, image_b, origin_b=ImageOrigin(x=2), mask=mask, levels=None
    )
    explicit = blend(image_a, image_b, origin_b=ImageOrigin(x=2), mask=mask, levels=4)
    np.testing.assert_array_equal(automatic.blended_image, explicit.blended_image)


@pytest.mark.parametrize("mask", ["linear", "cosine", "graphcut"])
@pytest.mark.parametrize("levels", [5, 20])
def test_requested_depth_stops_at_the_canvas_limit(mask, levels):
    rng = np.random.default_rng(15)
    image_a = rng.random((5, 7), dtype=np.float32)
    image_b = rng.random((5, 7), dtype=np.float32)
    limited = blend(
        image_a, image_b, origin_b=ImageOrigin(x=2), mask=mask, levels=levels
    )
    maximum = blend(image_a, image_b, origin_b=ImageOrigin(x=2), mask=mask, levels=4)
    np.testing.assert_array_equal(limited.blended_image, maximum.blended_image)
    np.testing.assert_array_equal(limited.overlap_mask, maximum.overlap_mask)


def test_default_depth_uses_five_levels_for_larger_images():
    rng = np.random.default_rng(16)
    image_a = rng.random((33, 49), dtype=np.float32)
    image_b = rng.random((33, 49), dtype=np.float32)
    default = blend(image_a, image_b, origin_b=ImageOrigin(x=12), mask="linear")
    five_levels = blend(
        image_a, image_b, origin_b=ImageOrigin(x=12), mask="linear", levels=5
    )
    np.testing.assert_array_equal(default.blended_image, five_levels.blended_image)
    np.testing.assert_array_equal(default.overlap_mask, five_levels.overlap_mask)


@pytest.mark.parametrize("mask", ["linear", "cosine", "graphcut"])
def test_translating_both_origins_only_changes_canvas_origin(mask):
    rng = np.random.default_rng(27)
    image_a = rng.random((7, 9, 3), dtype=np.float32)
    image_b = rng.random((7, 9, 3), dtype=np.float32)
    original = blend(
        image_a, image_b, ImageOrigin(), ImageOrigin(1, 3), mask=mask, levels=2
    )
    translated = blend(
        image_a, image_b, ImageOrigin(-5, 4), ImageOrigin(-4, 7), mask=mask, levels=2
    )
    assert translated.canvas_origin == ImageOrigin(-5, 4)
    np.testing.assert_array_equal(translated.blended_image, original.blended_image)
    np.testing.assert_array_equal(translated.overlap_mask, original.overlap_mask)
