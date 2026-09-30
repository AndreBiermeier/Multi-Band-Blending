"""Integer image placement and rectangular overlap geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageOrigin:
    """Top-left image position in shared coordinates.

    Attributes:
        y (int): Row coordinate. Negative positions are allowed.
        x (int): Column coordinate. Negative positions are allowed.
    """

    y: int = 0
    x: int = 0


@dataclass(frozen=True)
class OverlapRegion:
    """Describe the shared rectangle and its placement.

    Attributes:
        canvas_origin (ImageOrigin): Top-left corner of the bounding output canvas.
        canvas_shape (tuple[int, int]): Canvas height and width.
        overlap_origin (ImageOrigin): Top-left corner of the shared rectangle.
        overlap_shape (tuple[int, int]): Overlap height and width.
        image_a_slice (tuple[slice, slice]): Overlap coordinates within image A.
        image_b_slice (tuple[slice, slice]): Overlap coordinates within image B.
        canvas_slice (tuple[slice, slice]): Overlap coordinates within the output
            canvas.
    """

    canvas_origin: ImageOrigin
    canvas_shape: tuple[int, int]
    overlap_origin: ImageOrigin
    overlap_shape: tuple[int, int]
    image_a_slice: tuple[slice, slice]
    image_b_slice: tuple[slice, slice]
    canvas_slice: tuple[slice, slice]


def placement_slice(
    origin: ImageOrigin, shape: tuple[int, ...], canvas_origin: ImageOrigin
) -> tuple[slice, slice]:
    """Get the slice of a rectangle in the containing canvas.

    Args:
        origin (ImageOrigin): Rectangle's world origin.
        shape (tuple[int, ...]): Rectangle shape (height, width).
        canvas_origin (ImageOrigin): Containing canvas's world origin.

    Returns:
        tuple[slice, slice]: Row and column slices.
    """
    row = origin.y - canvas_origin.y
    column = origin.x - canvas_origin.x
    return slice(row, row + shape[0]), slice(column, column + shape[1])


def calculate_overlap(
    image_a_shape: tuple[int, ...],
    image_b_shape: tuple[int, ...],
    origin_a: ImageOrigin = ImageOrigin(),
    origin_b: ImageOrigin = ImageOrigin(),
) -> OverlapRegion:
    """Calculate the overlap and bounding canvas of two positioned images.

    Args:
        image_a_shape (tuple[int, ...]): Shape of image A.
        image_b_shape (tuple[int, ...]): Shape of image B.
        origin_a (ImageOrigin): Image A's position, in (y, x) order.
        origin_b (ImageOrigin): Image B's position, in (y, x) order.

    Returns:
        OverlapRegion: World coordinates, dimensions and overlap-local array slices.

    Raises:
        ValueError: The images do not overlap.
    """
    height_a, width_a = image_a_shape[:2]
    height_b, width_b = image_b_shape[:2]
    overlap_top = max(origin_a.y, origin_b.y)
    overlap_left = max(origin_a.x, origin_b.x)
    overlap_bottom = min(origin_a.y + height_a, origin_b.y + height_b)
    overlap_right = min(origin_a.x + width_a, origin_b.x + width_b)
    if overlap_top >= overlap_bottom or overlap_left >= overlap_right:
        raise ValueError("images do not overlap")
    canvas_origin = ImageOrigin(
        min(origin_a.y, origin_b.y), min(origin_a.x, origin_b.x)
    )
    canvas_shape = (
        max(origin_a.y + height_a, origin_b.y + height_b) - canvas_origin.y,
        max(origin_a.x + width_a, origin_b.x + width_b) - canvas_origin.x,
    )
    overlap_origin = ImageOrigin(overlap_top, overlap_left)
    overlap_shape = (overlap_bottom - overlap_top, overlap_right - overlap_left)
    return OverlapRegion(
        canvas_origin,
        canvas_shape,
        overlap_origin,
        overlap_shape,
        placement_slice(overlap_origin, overlap_shape, origin_a),
        placement_slice(overlap_origin, overlap_shape, origin_b),
        placement_slice(overlap_origin, overlap_shape, canvas_origin),
    )
