"""Binomial filtering and Gaussian/Laplacian pyramids after Burt and Adelson (1983).
"""

from functools import lru_cache

import numpy as np


def maximum_levels(image_shape: tuple[int, ...]) -> int:
    """Count levels until the shorter dimension reaches one pixel.

    Args:
        image_shape (tuple[int, ...]): Shape containing positive height and width.

    Returns:
        int: The count including the original resolution.
    """
    remaining_side = min(image_shape[:2])
    levels = 1
    while remaining_side > 1:
        remaining_side = (remaining_side + 1) // 2
        levels += 1
    return levels


def blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a separable binomial filter with replicated boundaries.

    Args:
        image (np.ndarray): A two-dimensional or channel-last float32 array.
        kernel_size (int): Positive odd filter width.

    Returns:
        np.ndarray: A filtered array with the same shape.
    """
    weights = _binomial_kernel(kernel_size)
    result = image
    for axis in (0, 1):
        padding = [(0, 0)] * image.ndim
        padding[axis] = (kernel_size // 2, kernel_size // 2)
        padded = np.pad(result, padding, mode="edge")
        result = np.zeros_like(image, dtype=np.float32)
        for offset, weight in enumerate(weights):
            section = [slice(None)] * image.ndim
            section[axis] = slice(offset, offset + image.shape[axis])
            result += weight * padded[tuple(section)]
    return result


def gaussian_pyramid(
    image: np.ndarray, levels: int, kernel_size: int
) -> list[np.ndarray]:
    """Build progressively filtered and downsampled levels.

    Args:
        image (np.ndarray): Finest level.
        levels (int): Number of levels including the original.
        kernel_size (int): Binomial filter width.

    Returns:
        list[np.ndarray]: Levels ordered from finest to coarsest.
    """
    pyramid = [image]
    for _ in range(1, levels):
        pyramid.append(blur(pyramid[-1], kernel_size)[::2, ::2])
    return pyramid


def expand(image: np.ndarray, shape: tuple[int, ...], kernel_size: int) -> np.ndarray:
    """Repeat, crop and filter a coarse level.

    Args:
        image (np.ndarray): Coarse level to expand.
        shape (tuple[int, ...]): Target shape.
        kernel_size (int): Binomial filter width.

    Returns:
        np.ndarray: The expanded level.
    """
    expanded = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    return blur(expanded[: shape[0], : shape[1]], kernel_size)


def laplacian_pyramid(gaussian: list[np.ndarray], kernel_size: int) -> list[np.ndarray]:
    """Subtract expanded coarse levels to form frequency bands.

    Args:
        gaussian (list[np.ndarray]): Gaussian levels ordered finest first.
        kernel_size (int): Expansion filter width.

    Returns:
        list[np.ndarray]: Detail bands followed by the coarsest residual.
    """
    return [
        fine - expand(coarse, fine.shape, kernel_size)
        for fine, coarse in zip(gaussian[:-1], gaussian[1:], strict=True)
    ] + [gaussian[-1]]


@lru_cache(maxsize=16)
def _binomial_kernel(kernel_size: int) -> np.ndarray:
    """Construct immutable binomial coefficients.

    Args:
        kernel_size (int): Validated odd number of coefficients.

    Returns:
        np.ndarray: Normalized float32 coefficients.
    """
    weights = np.array([1.0], dtype=np.float32)
    half_box = np.array([0.5, 0.5], dtype=np.float32)
    for _ in range(kernel_size - 1):
        weights = np.convolve(weights, half_box).astype(np.float32, copy=False)
    weights.setflags(write=False)
    return weights
