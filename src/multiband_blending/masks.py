"""Directional feathering and optionally smoothed graph-cut seam masks."""

import numpy as np

from ._graphcut import graphcut_mask
from ._pyramid import blur
from .geometry import ImageOrigin, OverlapRegion


def create_mask(
    image_a: np.ndarray,
    image_b: np.ndarray,
    origin_a: ImageOrigin,
    origin_b: ImageOrigin,
    overlap: OverlapRegion,
    method: str,
    mask_kernel_size: int,
) -> np.ndarray:
    """Generate the requested image-A weight in the overlap.

    Args:
        image_a (np.ndarray): Normalized float32 image A colors without alpha.
        image_b (np.ndarray): Normalized float32 image B colors without alpha.
        origin_a (ImageOrigin): Image A top-left world coordinate.
        origin_b (ImageOrigin): Image B top-left world coordinate.
        overlap (OverlapRegion): Intersection geometry for the two images.
        method (str): ``linear``, ``cosine`` or ``graphcut``.
        mask_kernel_size (int): Positive odd graph-cut smoothing width. 1 disables it.

    Returns:
        np.ndarray: Float32 overlap mask in [0, 1]. 1 selects image A.

    Raises:
        ValueError: The overlap cannot support a graph-cut seam.
    """
    if method == "graphcut":
        mask = graphcut_mask(image_a, image_b, origin_a, origin_b, overlap)
        mask = blur(mask, mask_kernel_size)
    else:
        mask = _feather_mask(overlap.overlap_shape, origin_a, origin_b, method)
    return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)


def _feather_mask(
    shape: tuple[int, int],
    origin_a: ImageOrigin,
    origin_b: ImageOrigin,
    method: str,
) -> np.ndarray:
    """Project overlap coordinates onto the image-offset direction.

    Args:
        shape (tuple[int, int]): Overlap height and width.
        origin_a (ImageOrigin): Image A's world origin.
        origin_b (ImageOrigin): Image B's world origin.
        method (str): Feathering method: 'linear' or 'cosine'.

    Returns:
        np.ndarray: Float32 image A weights with the overlap shape.
    """
    offset_y = origin_b.y - origin_a.y
    offset_x = origin_b.x - origin_a.x
    rows, columns = np.indices(shape, dtype=np.float32)
    projection = rows * offset_y + columns * offset_x
    minimum, maximum = float(projection.min()), float(projection.max())
    if maximum == minimum:
        projection = np.full(shape, 0.5, dtype=np.float32)
    else:
        projection = (projection - minimum) / (maximum - minimum)
    if method == "linear":
        return 1.0 - projection
    return 0.5 * (1.0 + np.cos(np.pi * projection))
