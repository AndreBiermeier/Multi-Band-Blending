"""Calculate the overlap with its surrounding collar and extend source data into it.

Burt and Adelson, A Multiresolution Spline with Application to Image Mosaics
(1983), discuss increasing filter support and boundary extrapolation. The
collar width k*(2**(levels-1)-1)+1 is based on their discussion.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import ImageOrigin, OverlapRegion, placement_slice


@dataclass(frozen=True)
class CollaredOverlap:
    """Rectangle containing the overlap and its surrounding collar.

    Attributes:
        origin (ImageOrigin): World position of the first sample.
        shape (tuple[int, int]): Region height and width.
        overlap_slice (tuple[slice, slice]): Overlap's position within this region.
    """

    origin: ImageOrigin
    shape: tuple[int, int]
    overlap_slice: tuple[slice, slice]


def calculate_collared_overlap(
    overlap: OverlapRegion, levels: int, kernel_size: int
) -> CollaredOverlap:
    """Construct the collared overlap region.

    Args:
        overlap (OverlapRegion): Overlap and canvas geometry.
        levels (int): Pyramid depth including the original resolution.
        kernel_size (int): Binomial filter width.

    Returns:
        CollaredOverlap: The collared overlap.
    """
    grid_spacing = 2 ** (levels - 1)
    padding = kernel_size * (grid_spacing - 1) + 1 if levels > 1 else 0
    top, bottom = _expand_axis_to_grid(
        overlap.overlap_origin.y,
        overlap.overlap_shape[0],
        overlap.canvas_origin.y,
        overlap.canvas_shape[0],
        padding,
        grid_spacing,
    )
    left, right = _expand_axis_to_grid(
        overlap.overlap_origin.x,
        overlap.overlap_shape[1],
        overlap.canvas_origin.x,
        overlap.canvas_shape[1],
        padding,
        grid_spacing,
    )
    origin = ImageOrigin(top, left)
    shape = (bottom - top, right - left)
    return CollaredOverlap(
        origin,
        shape,
        placement_slice(overlap.overlap_origin, overlap.overlap_shape, origin),
    )


def _expand_axis_to_grid(
    start: int,
    size: int,
    canvas_start: int,
    canvas_size: int,
    padding: int,
    grid_spacing: int,
) -> tuple[int, int]:
    """Pad one axis, round outward to the canvas grid, and clip to the canvas.

    Args:
        start (int): Overlap's world start along this axis.
        size (int): Overlap's length along this axis.
        canvas_start (int): Canvas's world start along this axis.
        canvas_size (int): Canvas's length along this axis.
        padding (int): Extra pixels to include at each end.
        grid_spacing (int): Required sampling alignment in original pixels.

    Returns:
        tuple[int, int]: World start and exclusive end of the expanded interval.
    """
    local_start = start - canvas_start
    padded_start = max(0, local_start - padding)
    padded_end = min(canvas_size, local_start + size + padding)

    # Round the start down and the end up to include the entire padded interval.
    aligned_start = padded_start // grid_spacing * grid_spacing
    aligned_end = (padded_end + grid_spacing - 1) // grid_spacing * grid_spacing
    clipped_end = min(canvas_size, aligned_end)
    return canvas_start + aligned_start, canvas_start + clipped_end


def extend_image_to_collar(
    image: np.ndarray, origin: ImageOrigin, region: CollaredOverlap
) -> np.ndarray:
    """Sample the region replicating the nearest source edge as needed.

    Args:
        image (np.ndarray): Source color image or alpha plane.
        origin (ImageOrigin): Source image's world origin.
        region (CollaredOverlap): Rectangle to sample.

    Returns:
        np.ndarray: An array covering the region.
    """
    rows = np.clip(
        np.arange(region.shape[0]) + region.origin.y - origin.y, 0, image.shape[0] - 1
    )
    columns = np.clip(
        np.arange(region.shape[1]) + region.origin.x - origin.x, 0, image.shape[1] - 1
    )
    return image[rows[:, None], columns[None, :]]


def _squared_distance(
    rows: np.ndarray, columns: np.ndarray, origin: ImageOrigin, shape: tuple[int, ...]
) -> np.ndarray:
    """Measure squared distance from each sample to the nearest source pixel.

    Args:
        rows (np.ndarray): World row coordinates.
        columns (np.ndarray): World column coordinates.
        origin (ImageOrigin): Source rectangle origin.
        shape (tuple[int, ...]): Source rectangle shape.

    Returns:
        np.ndarray: Squared Euclidean distances, with zero inside the source rectangle.
    """
    row_distance = rows - np.clip(rows, origin.y, origin.y + shape[0] - 1)
    column_distance = columns - np.clip(columns, origin.x, origin.x + shape[1] - 1)
    return row_distance[:, None] ** 2 + column_distance[None, :] ** 2


def extend_mask_to_collar(
    mask: np.ndarray,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    origin_a: ImageOrigin,
    origin_b: ImageOrigin,
    overlap: OverlapRegion,
    region: CollaredOverlap,
) -> np.ndarray:
    """Extend overlap selection through the surrounding region.

    Args:
        mask (np.ndarray): Image A weights in the overlap.
        shape_a (tuple[int, ...]): Shape of image A.
        shape_b (tuple[int, ...]): Shape of image B.
        origin_a (ImageOrigin): Image A's origin.
        origin_b (ImageOrigin): Image B's origin.
        overlap (OverlapRegion): Overlap geometry.
        region (CollaredOverlap): Region to fill.

    Returns:
        np.ndarray: Weights selecting the only supported image outside the overlap. Gaps
            select the nearest rectangle, breaking ties with the nearest mask value.
    """
    rows = np.arange(region.shape[0]) + region.origin.y
    columns = np.arange(region.shape[1]) + region.origin.x
    distance_a = _squared_distance(rows, columns, origin_a, shape_a)
    distance_b = _squared_distance(rows, columns, origin_b, shape_b)
    result = (distance_a < distance_b).astype(np.float32)
    overlap_rows = np.clip(rows - overlap.overlap_origin.y, 0, mask.shape[0] - 1)
    overlap_columns = np.clip(columns - overlap.overlap_origin.x, 0, mask.shape[1] - 1)
    nearest = mask[overlap_rows[:, None], overlap_columns[None, :]]
    ties = distance_a == distance_b
    result[ties] = nearest[ties] >= 0.5
    result[region.overlap_slice] = mask
    return result
