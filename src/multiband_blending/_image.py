"""Validate and normalize input images and select available sources by alpha."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .geometry import ImageOrigin


@dataclass(frozen=True, eq=False)
class PreparedImage:
    """Normalized image data passed between pipeline steps.

    Attributes:
        colors (np.ndarray): Float32 colors with shape (H, W, 1) or (H, W, 3).
        alpha (np.ndarray): Float32 opacity with shape (H, W).
        origin (ImageOrigin): Top-left image position in world coordinates.
        is_opaque (bool): Whether every source pixel has alpha equal to one.
    """

    colors: np.ndarray
    alpha: np.ndarray
    origin: ImageOrigin
    is_opaque: bool


def normalize_image_values(array: np.ndarray, name: str) -> np.ndarray:
    """Convert supported image values to float32 without changing the input.

    Args:
        array (np.ndarray): Nonempty unsigned integer or normalized floating-point
            array.
        name (str): Parameter name used in validation errors.

    Returns:
        np.ndarray: Float32 values in [0, 1], with the original shape.

    Raises:
        TypeError: The array or its dtype is unsupported.
        ValueError: Values are empty, nonfinite, or outside the float range.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(array.dtype, np.unsignedinteger):
        return array.astype(np.float32) / np.float32(np.iinfo(array.dtype).max)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have an unsigned integer or floating-point dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    if array.min() < 0 or array.max() > 1:
        raise ValueError(f"{name} floating-point values must be in [0, 1]")
    return array.astype(np.float32, copy=False)


def prepare_image(image: np.ndarray, origin: ImageOrigin, name: str) -> PreparedImage:
    """Normalize an image and separate its color channels from straight alpha.

    Args:
        image (np.ndarray): Image with shape (H, W), (H, W, 1/2), or (H, W, 3/4).
        origin (ImageOrigin): Top-left image position in world coordinates.
        name (str): Parameter name used in validation errors.

    Returns:
        PreparedImage: Normalized colors, alpha, and placement. Missing alpha is
            filled with ones. Channel order is preserved.
    """
    values = normalize_image_values(image, name)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3 or values.shape[2] not in (1, 2, 3, 4):
        raise ValueError(f"{name} must be grayscale, grayscale-alpha, RGB, or RGBA")
    if values.shape[2] in (2, 4):
        colors, alpha = values[..., :-1], values[..., -1]
    else:
        colors = values
        alpha = np.ones(values.shape[:2], dtype=np.float32)
    return PreparedImage(colors, alpha, origin, bool(np.all(alpha == 1)))


def validate_positive_integer(value: int, name: str) -> int:
    """Validate a positive integer.

    Args:
        value (int): Candidate integer.
        name (str): Parameter name used in validation errors.

    Returns:
        int: The value as a Python integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_kernel_size(kernel_size: int, name: str = "kernel_size") -> int:
    """Validate the width of an odd binomial filter.

    Args:
        kernel_size (int): Positive odd filter width. 1 disables filtering.
        name (str): Parameter name used in validation errors.

    Returns:
        int: The validated width.
    """
    size = validate_positive_integer(kernel_size, name)
    if size % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return size


def adjust_mask_for_transparency(
    mask: np.ndarray, alpha_a: np.ndarray, alpha_b: np.ndarray
) -> np.ndarray:
    """Select the available source when the other is fully transparent.

    Args:
        mask (np.ndarray): Float32 mask for image A, with shape (H, W).
        alpha_a (np.ndarray): Image A's float32 opacity with shape (H, W).
        alpha_b (np.ndarray): Image B's float32 opacity with shape (H, W).

    Returns:
        np.ndarray: A new mask with weights corrected for unavailable pixels.
    """
    result = mask.copy()
    result[(alpha_a > 0) & (alpha_b == 0)] = 1.0
    result[(alpha_a == 0) & (alpha_b > 0)] = 0.0
    return result
