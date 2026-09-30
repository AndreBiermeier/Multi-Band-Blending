"""Public input validation for images, placement, mask options and pyramid depth."""

import numpy as np
import pytest

from multiband_blending import ImageOrigin, blend


@pytest.mark.parametrize("option", ["kernel_size", "mask_kernel_size"])
@pytest.mark.parametrize("value", [0, -1, 2, 2.5, True, "5"])
def test_invalid_kernels_are_rejected(option, value):
    image = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        blend(image, image, origin_b=ImageOrigin(x=2), **{option: value})


@pytest.mark.parametrize("levels", [0, -1, 1.5, True, "2"])
def test_invalid_levels_are_rejected(levels):
    image = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        blend(image, image, origin_b=ImageOrigin(x=2), mask="linear", levels=levels)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_float_image_values_are_rejected(value):
    image = np.zeros((5, 7), dtype=np.float32)
    invalid = image.copy()
    invalid[0, 0] = value
    with pytest.raises(ValueError):
        blend(invalid, image, origin_b=ImageOrigin(x=2), mask="linear")


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.bool_, np.complex64, object])
def test_unsupported_image_dtypes_are_rejected(dtype):
    image = np.zeros((5, 7), dtype=dtype)
    with pytest.raises((TypeError, ValueError)):
        blend(image, image, origin_b=ImageOrigin(x=2), mask="linear")


@pytest.mark.parametrize(
    "shape", [(0, 7), (5, 0), (5,), (5, 7, 0), (5, 7, 5), (2, 3, 4, 5)]
)
def test_invalid_image_shapes_are_rejected(shape):
    image = np.zeros(shape, dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        blend(image, image, origin_b=ImageOrigin(x=2), mask="linear")


def test_grayscale_color_mismatch_is_rejected():
    with pytest.raises(ValueError):
        blend(
            np.zeros((5, 7)),
            np.zeros((5, 7, 3)),
            origin_b=ImageOrigin(x=2),
            mask="linear",
        )


def test_non_array_image_is_rejected():
    with pytest.raises(TypeError):
        blend([[0, 1]], np.zeros((1, 2)), origin_b=ImageOrigin(x=1), mask="linear")


@pytest.mark.parametrize("method", ["unknown", None, [], 1])
def test_unknown_mask_is_rejected(method):
    image = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises(ValueError):
        blend(image, image, mask=method)


def test_disjoint_images_are_rejected():
    image = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises(ValueError):
        blend(image, image, origin_b=ImageOrigin(x=7), mask="linear")


@pytest.mark.parametrize("mask", ["linear", "cosine", "graphcut"])
def test_equal_origins_are_rejected(mask):
    image = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="distinct image origins"):
        blend(image, image, mask=mask)


def test_plain_tuple_origins_are_rejected():
    image = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises(TypeError, match="ImageOrigin"):
        blend(image, image, origin_b=(0, 2), mask="linear")


def test_graphcut_rejects_containment():
    image_a = np.zeros((7, 9), dtype=np.float32)
    image_b = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        blend(image_a, image_b, origin_b=ImageOrigin(1, 1))


def test_graphcut_requires_two_pixel_overlap_width():
    image = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises(ValueError):
        blend(image, image, origin_b=ImageOrigin(x=6))
