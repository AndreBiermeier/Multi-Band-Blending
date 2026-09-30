"""Prepare rectangular graph cuts for the Dijkstra solver.

The seam energy follows Kwatra et al., "Graphcut Textures: Image and Video
Synthesis Using Graph Cuts" (2003), Equation (1). The native implementation
uses Hassin's "Maximum Flow in (s,t) Planar Networks" (1981) exterior-face
split. Each terminal must form one connected overlap-boundary arc.
"""

import numpy as np

from .geometry import ImageOrigin, OverlapRegion


def build_terminal_labels(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    origin_a: ImageOrigin,
    origin_b: ImageOrigin,
    overlap: OverlapRegion,
) -> np.ndarray:
    """Label boundaries by which source continues beyond the overlap.

    Args:
        shape_a (tuple[int, ...]): Image A dimensions, starting with height and width.
        shape_b (tuple[int, ...]): Image B dimensions, starting with height and width.
        origin_a (ImageOrigin): Image A top-left world coordinate.
        origin_b (ImageOrigin): Image B top-left world coordinate.
        overlap (OverlapRegion): Intersection geometry for the two images.

    Returns:
        np.ndarray: Contiguous int8 labels: A is 1, B is -1, and free pixels are 0.

    Raises:
        ValueError: The intersection cannot support a single separating seam.
    """
    height, width = overlap.overlap_shape
    if height < 2 or width < 2:
        raise ValueError("graph cut requires overlap height and width at least 2")
    if (
        overlap.overlap_origin == origin_a
        and overlap.overlap_shape == tuple(shape_a[:2])
    ) or (
        overlap.overlap_origin == origin_b
        and overlap.overlap_shape == tuple(shape_b[:2])
    ):
        raise ValueError("graph cut requires a partial overlap")

    top = overlap.overlap_origin.y
    left = overlap.overlap_origin.x
    bottom = top + height
    right = left + width
    labels = np.zeros((height, width), dtype=np.int8)
    _label_boundary(labels[0, :], origin_a.y < top, origin_b.y < top)
    _label_boundary(
        labels[-1, :],
        origin_a.y + shape_a[0] > bottom,
        origin_b.y + shape_b[0] > bottom,
    )
    _label_boundary(labels[:, 0], origin_a.x < left, origin_b.x < left)
    _label_boundary(
        labels[:, -1],
        origin_a.x + shape_a[1] > right,
        origin_b.x + shape_b[1] > right,
    )
    return labels


def _label_boundary(boundary: np.ndarray, continues_a: bool, continues_b: bool) -> None:
    """Assign a boundary to its continuing source, leaving conflicting corners free.

    Args:
        boundary (np.ndarray): Row or column view into the labels, modified in place.
        continues_a (bool): Whether image A extends beyond this overlap boundary.
        continues_b (bool): Whether image B extends beyond this overlap boundary.

    Returns:
        None: The boundary labels are updated in place.
    """
    if not continues_a and not continues_b:
        return
    label = 1 if continues_a else -1
    conflicts = (boundary != 0) & (boundary != label)
    boundary[:] = label
    boundary[conflicts] = 0


def graphcut_mask(
    image_a: np.ndarray,
    image_b: np.ndarray,
    origin_a: ImageOrigin,
    origin_b: ImageOrigin,
    overlap: OverlapRegion,
) -> np.ndarray:
    """Compute an exact binary seam mask from normalized image colors.

    Args:
        image_a (np.ndarray): Normalized float32 image A with no alpha channel.
        image_b (np.ndarray): Normalized float32 image B with matching channels.
        origin_a (ImageOrigin): Image A top-left world coordinate.
        origin_b (ImageOrigin): Image B top-left world coordinate.
        overlap (OverlapRegion): Intersection geometry for the two images.

    Returns:
        np.ndarray: Float32 overlap mask whose value one selects image A.

    Raises:
        ValueError: Geometry or normalized input values are invalid.
        TypeError: Input dtypes are not float32.
    """
    from . import _native

    labels = build_terminal_labels(
        image_a.shape, image_b.shape, origin_a, origin_b, overlap
    )
    patch_a = np.ascontiguousarray(image_a[overlap.image_a_slice])
    patch_b = np.ascontiguousarray(image_b[overlap.image_b_slice])
    return _native.graphcut_mask_dijkstra(patch_a, patch_b, labels)
