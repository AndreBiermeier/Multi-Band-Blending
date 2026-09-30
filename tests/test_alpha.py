"""Opacity handling, hidden colors, and regressions for transparent image edges."""

import numpy as np
import pytest

from multiband_blending import ImageOrigin, blend


@pytest.mark.parametrize("mask", ["linear", "cosine", "graphcut"])
def test_opaque_alpha_matches_color_input(mask):
    rng = np.random.default_rng(42)
    image_a = rng.random((7, 9, 3), dtype=np.float32)
    image_b = rng.random((7, 9, 3), dtype=np.float32)
    alpha = np.ones((7, 9, 1), dtype=np.float32)
    expected = blend(image_a, image_b, origin_b=ImageOrigin(x=3), mask=mask)
    actual = blend(
        np.concatenate((image_a, alpha), axis=2),
        np.concatenate((image_b, alpha), axis=2),
        origin_b=ImageOrigin(x=3),
        mask=mask,
    )
    np.testing.assert_allclose(actual.blended_image, expected.blended_image, atol=2e-6)
    np.testing.assert_array_equal(actual.overlap_mask, expected.overlap_mask)


@pytest.mark.parametrize("mask", ["linear", "cosine", "graphcut"])
def test_hidden_colors_do_not_affect_visible_output_or_seam(mask):
    image_a = np.full((7, 9, 4), 0.2, dtype=np.float32)
    image_b = np.full((7, 9, 4), 0.8, dtype=np.float32)
    image_a[..., 3] = 1
    image_b[..., 3] = 1
    image_a[2:5, 3:7, 3] = 0
    image_b[1:4, 1:4, 3] = 0
    expected = blend(image_a, image_b, origin_b=ImageOrigin(x=3), mask=mask)
    image_a[image_a[..., 3] == 0, :3] = [1, 0, 1]
    image_b[image_b[..., 3] == 0, :3] = [0, 1, 0]
    actual = blend(image_a, image_b, origin_b=ImageOrigin(x=3), mask=mask)
    np.testing.assert_allclose(actual.blended_image, expected.blended_image, atol=2e-6)
    np.testing.assert_array_equal(actual.overlap_mask, expected.overlap_mask)


@pytest.mark.parametrize("mask", ["linear", "cosine", "graphcut"])
def test_fully_transparent_sources_return_zero(mask):
    image_a = np.ones((5, 7, 4), dtype=np.float32)
    image_b = np.full_like(image_a, 0.7)
    image_a[..., 3] = image_b[..., 3] = 0
    result = blend(image_a, image_b, origin_b=ImageOrigin(x=2), mask=mask)
    np.testing.assert_array_equal(result.blended_image, 0)


@pytest.mark.parametrize("mask", ["linear", "cosine", "graphcut"])
def test_mixed_alpha_presence_preserves_only_visible_source(mask):
    image_a = np.full((5, 7, 3), [0.2, 0.4, 0.6], dtype=np.float32)
    image_b = np.zeros((5, 7, 4), dtype=np.float32)
    image_b[..., :3] = [1, 0, 1]
    result = blend(image_a, image_b, origin_b=ImageOrigin(x=2), mask=mask)
    np.testing.assert_allclose(result.blended_image[:, :7, :3], image_a, atol=2e-6)
    np.testing.assert_allclose(result.blended_image[:, :7, 3], 1, atol=2e-6)
    np.testing.assert_array_equal(result.blended_image[:, 7:], 0)


@pytest.mark.parametrize("mask", ["linear", "cosine"])
def test_one_level_partial_alpha_is_premultiplied_weighted_blend(mask):
    image_a = np.full((5, 7, 4), [0.2, 0.4, 0.6, 0.25], dtype=np.float32)
    image_b = np.full((5, 7, 4), [0.8, 0.6, 0.4, 0.75], dtype=np.float32)
    result = blend(image_a, image_b, origin_b=ImageOrigin(x=2), mask=mask, levels=1)
    weight = result.overlap_mask
    alpha = weight * 0.25 + (1 - weight) * 0.75
    premultiplied = (
        weight[..., None] * image_a[:, 2:, :3] * 0.25
        + (1 - weight[..., None]) * image_b[:, :5, :3] * 0.75
    )
    np.testing.assert_allclose(
        result.blended_image[:, 2:7, :3], premultiplied / alpha[..., None], atol=2e-6
    )
    np.testing.assert_allclose(result.blended_image[:, 2:7, 3], alpha, atol=2e-6)
    np.testing.assert_allclose(result.blended_image[:, :2], image_a[:, :2], atol=2e-6)
    np.testing.assert_allclose(result.blended_image[:, 7:], image_b[:, 5:], atol=2e-6)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("method", ["linear", "cosine", "graphcut"])
def test_constant_visible_color_survives_alpha_holes(channels, method):
    color = np.array([0.2, 0.4, 0.8][:channels], dtype=np.float32)
    rng = np.random.default_rng(18)
    image_a = np.empty((9, 13, channels + 1), dtype=np.float32)
    image_b = np.empty((11, 10, channels + 1), dtype=np.float32)
    for image in (image_a, image_b):
        image[..., :-1] = color
        image[..., -1] = rng.choice([0.0, 0.25, 0.5, 1.0], size=image.shape[:2])
        image[..., :-1][image[..., -1] == 0] = 1.0
        image.setflags(write=False)
    result = blend(
        image_a, image_b, ImageOrigin(-2, -1), ImageOrigin(1, 3), mask=method
    )
    visible = result.blended_image[..., -1] > 0
    np.testing.assert_allclose(
        result.blended_image[..., :-1][visible],
        np.broadcast_to(color, (np.count_nonzero(visible), channels)),
        rtol=0,
        atol=2e-6,
    )
    np.testing.assert_array_equal(result.blended_image[~visible], 0)


@pytest.mark.parametrize("method", ["linear", "cosine", "graphcut"])
def test_opposing_transparent_bands_do_not_brighten_source_only_pixels(method):
    color = np.array([0.2, 0.4, 0.7], dtype=np.float32)
    image_a = np.full((5, 7, 4), [*color, 1], dtype=np.float32)
    image_b = image_a.copy()
    image_a[:3, :, 3] = 0
    image_b[-3:, :, 3] = 0
    result = blend(image_a, image_b, origin_b=ImageOrigin(1, 2), mask=method)
    visible = result.blended_image[..., 3] > 0
    np.testing.assert_allclose(
        result.blended_image[..., :3][visible],
        np.broadcast_to(color, (np.count_nonzero(visible), 3)),
        rtol=0,
        atol=2e-6,
    )


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("method", ["linear", "cosine", "graphcut"])
def test_sparse_alpha_preserves_constant_color_in_shared_visible_pixels(
    kernel_size, method
):
    rng = np.random.default_rng(434)
    color = np.array([0.2, 0.4, 0.7], dtype=np.float32)
    image_a = np.full((13, 17, 4), [*color, 1], dtype=np.float32)
    image_b = image_a.copy()
    image_a[..., 3] = rng.random(image_a.shape[:2]) < 0.15
    image_b[..., 3] = rng.random(image_b.shape[:2]) < 0.15
    result = blend(
        image_a,
        image_b,
        origin_b=ImageOrigin(2, 3),
        mask=method,
        kernel_size=kernel_size,
    )
    visible = result.blended_image[..., 3] > 0
    np.testing.assert_allclose(
        result.blended_image[..., :3][visible],
        np.broadcast_to(color, (np.count_nonzero(visible), 3)),
        rtol=0,
        atol=2e-6,
    )


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_sparse_single_source_is_not_dimmed_by_empty_coarse_weights(kernel_size):
    image_a = np.zeros((9, 13, 4), dtype=np.float32)
    image_a[..., :3] = [0.2, 0.4, 0.7]
    image_a[1::2, 1::2, 3] = 0.5
    image_b = np.zeros_like(image_a)
    result = blend(
        image_a,
        image_b,
        origin_b=ImageOrigin(2, 3),
        mask="linear",
        kernel_size=kernel_size,
    )
    source_result = result.blended_image[:9, :13]
    visible = image_a[..., 3] > 0
    np.testing.assert_allclose(
        source_result[visible], image_a[visible], rtol=0, atol=2e-6
    )
