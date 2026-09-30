"""Coordinate input preparation, seam selection, pyramid blending and canvas assembly.

Pyramid blending follows Burt and Adelson (1983).
Edge costs for seam selection follow Kwatra et al. (2003).
The minimum cut calculation follows Hassin (1981).
"""

from dataclasses import dataclass

import numpy as np

from ._blending import blend_images_with_alpha, blend_opaque_images
from ._canvas import assemble_canvas
from ._collar import (
    CollaredOverlap,
    calculate_collared_overlap,
    extend_image_to_collar,
    extend_mask_to_collar,
)
from ._image import (
    PreparedImage,
    adjust_mask_for_transparency,
    prepare_image,
    validate_kernel_size,
    validate_positive_integer,
)
from ._pyramid import maximum_levels
from .geometry import ImageOrigin, OverlapRegion, calculate_overlap
from .masks import create_mask


@dataclass(frozen=True, eq=False)
class BlendResult:
    """Blended output and the mask for image A.

    Attributes:
        blended_image (np.ndarray): Float32 blended canvas with alpha channel: (H, W, 2)
            for grayscale or (H, W, 4) for color. Alpha values in [0, 1].
        overlap_mask (np.ndarray): The mask for image A. Where only one source has
            positive alpha, that source is selected.
        canvas_origin (ImageOrigin): World position corresponding to output pixel (0,
            0).
    """

    blended_image: np.ndarray
    overlap_mask: np.ndarray
    canvas_origin: ImageOrigin


def blend(
    image_a: np.ndarray,
    image_b: np.ndarray,
    origin_a: ImageOrigin = ImageOrigin(),
    origin_b: ImageOrigin = ImageOrigin(),
    *,
    mask: str = "graphcut",
    kernel_size: int = 5,
    mask_kernel_size: int = 5,
    levels: int | None = 5,
) -> BlendResult:
    """Blend two positioned images using multi-band blending.

    Args:
        image_a (np.ndarray): Grayscale, grayscale-alpha, RGB, or RGBA NumPy image.
            Unsigned integers use their dtype range. Floats must be finite and in [0,
            1].
        image_b (np.ndarray): The second image of the same type.
        origin_a (ImageOrigin): Top-left position of image A, in (y, x) order.
        origin_b (ImageOrigin): Top-left position of image B, in (y, x) order.
        mask (str): The used mask type: 'graphcut', 'linear', or 'cosine'.
        kernel_size (int): Positive odd binomial width for the image and mask pyramids.
        mask_kernel_size (int): Positive odd width for smoothing the binary graph cut;
            one disables smoothing. Has no effect on feathering.
        levels (int | None): Maximum pyramid depth including original resolution.
            Defaults to five.
            None uses the maximum supported by the shorter canvas dimension.

    Returns:
        BlendResult: Float32 LA/RGBA canvas, effective overlap mask and canvas origin.
            Output alpha interpolates source alpha with the effective mask.

    Raises:
        TypeError: An input array, dtype, or origin has an unsupported type.
        ValueError: Values, channels, configuration or overlap are unsupported.
            Graph cut requires a partial overlap at least two pixels wide and
            high with connected terminal boundary arcs. All mask methods require
            distinct origins.
    """
    kernel_size, mask_kernel_size = _validate_blend_options(
        origin_a, origin_b, mask, kernel_size, mask_kernel_size
    )
    prepared_a, prepared_b = _prepare_images(image_a, image_b, origin_a, origin_b)

    overlap = calculate_overlap(
        prepared_a.colors.shape, prepared_b.colors.shape, origin_a, origin_b
    )
    overlap_mask = _create_overlap_mask(
        prepared_a, prepared_b, overlap, mask, mask_kernel_size
    )

    levels = _resolve_pyramid_levels(levels, overlap.canvas_shape)
    collared_overlap = calculate_collared_overlap(overlap, levels, kernel_size)
    blended_colors = _blend_overlap_with_collar(
        prepared_a,
        prepared_b,
        overlap,
        overlap_mask,
        collared_overlap,
        levels,
        kernel_size,
    )

    canvas = assemble_canvas(
        prepared_a,
        prepared_b,
        overlap,
        blended_colors,
        collared_overlap.origin,
        overlap_mask,
    )
    return BlendResult(canvas, overlap_mask, overlap.canvas_origin)


def _validate_blend_options(
    origin_a: ImageOrigin,
    origin_b: ImageOrigin,
    mask: str,
    kernel_size: int,
    mask_kernel_size: int,
) -> tuple[int, int]:
    """Check image origins and mask options before preparing source data.

    Args:
        origin_a (ImageOrigin): Image A's world position.
        origin_b (ImageOrigin): Image B's world position.
        mask (str): Requested mask method.
        kernel_size (int): Requested pyramid filter width.
        mask_kernel_size (int): Requested graph-cut smoothing width.

    Returns:
        tuple[int, int]: Validated pyramid and mask filter widths.
    """
    if mask not in ("linear", "cosine", "graphcut"):
        raise ValueError("mask must be 'linear', 'cosine', or 'graphcut'")
    if not isinstance(origin_a, ImageOrigin) or not isinstance(origin_b, ImageOrigin):
        raise TypeError("origins must be ImageOrigin instances")
    if origin_a == origin_b:
        raise ValueError("blending requires distinct image origins")
    return (
        validate_kernel_size(kernel_size),
        validate_kernel_size(mask_kernel_size, "mask_kernel_size"),
    )


def _prepare_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    origin_a: ImageOrigin,
    origin_b: ImageOrigin,
) -> tuple[PreparedImage, PreparedImage]:
    """Normalize both images and check that their color channels match.

    Args:
        image_a (np.ndarray): Original image A, with optional alpha.
        image_b (np.ndarray): Original image B, with optional alpha.
        origin_a (ImageOrigin): Image A's world position.
        origin_b (ImageOrigin): Image B's world position.

    Returns:
        tuple[PreparedImage, PreparedImage]: Prepared image A and image B.
    """
    prepared_a = prepare_image(image_a, origin_a, "image_a")
    prepared_b = prepare_image(image_b, origin_b, "image_b")
    if prepared_a.colors.shape[2] != prepared_b.colors.shape[2]:
        raise ValueError("image_a and image_b must have the same color channel count")
    return prepared_a, prepared_b


def _resolve_pyramid_levels(levels: int | None, canvas_shape: tuple[int, int]) -> int:
    """Limit the requested pyramid depth to what the canvas supports.

    Args:
        levels (int | None): Requested depth limit or None for the maximum.
        canvas_shape (tuple[int, int]): Output canvas height and width.

    Returns:
        int: Validated pyramid depth including the original resolution.
    """
    maximum = maximum_levels(canvas_shape)
    if levels is None:
        return maximum
    return min(validate_positive_integer(levels, "levels"), maximum)


def _create_overlap_mask(
    image_a: PreparedImage,
    image_b: PreparedImage,
    overlap: OverlapRegion,
    method: str,
    mask_kernel_size: int,
) -> np.ndarray:
    """Create the seam mask using visible colors and source availability.

    Args:
        image_a (PreparedImage): Prepared image A.
        image_b (PreparedImage): Prepared image B.
        overlap (OverlapRegion): Shared rectangle and canvas geometry.
        method (str): Validated mask method.
        mask_kernel_size (int): Validated graph-cut smoothing width.

    Returns:
        np.ndarray: Float32 overlap weights; one selects image A.
    """
    seam_colors_a, seam_colors_b = image_a.colors, image_b.colors

    # Transparent pixels should have no edge cost.
    if method == "graphcut" and not (image_a.is_opaque and image_b.is_opaque):
        seam_colors_a = image_a.colors * image_a.alpha[..., None]
        seam_colors_b = image_b.colors * image_b.alpha[..., None]

    mask = create_mask(
        seam_colors_a,
        seam_colors_b,
        image_a.origin,
        image_b.origin,
        overlap,
        method,
        mask_kernel_size,
    )
    return adjust_mask_for_transparency(
        mask, image_a.alpha[overlap.image_a_slice], image_b.alpha[overlap.image_b_slice]
    )


def _blend_overlap_with_collar(
    image_a: PreparedImage,
    image_b: PreparedImage,
    overlap: OverlapRegion,
    overlap_mask: np.ndarray,
    collared_overlap: CollaredOverlap,
    levels: int,
    kernel_size: int,
) -> np.ndarray:
    """Extend source data into the collar and blend the resulting rectangle.

    Args:
        image_a (PreparedImage): Prepared image A.
        image_b (PreparedImage): Prepared image B.
        overlap (OverlapRegion): Shared rectangle and canvas geometry.
        overlap_mask (np.ndarray): Float32 image A weights within the overlap.
        collared_overlap (CollaredOverlap): Overlap plus its surrounding collar.
        levels (int): Validated pyramid depth.
        kernel_size (int): Validated pyramid filter width.

    Returns:
        np.ndarray: Float32 blended colors covering the collared overlap.
    """
    extended_mask = extend_mask_to_collar(
        overlap_mask,
        image_a.colors.shape,
        image_b.colors.shape,
        image_a.origin,
        image_b.origin,
        overlap,
        collared_overlap,
    )
    extended_colors_a = extend_image_to_collar(
        image_a.colors, image_a.origin, collared_overlap
    )
    extended_colors_b = extend_image_to_collar(
        image_b.colors, image_b.origin, collared_overlap
    )
    if image_a.is_opaque and image_b.is_opaque:
        return blend_opaque_images(
            extended_colors_a, extended_colors_b, extended_mask, levels, kernel_size
        )

    extended_alpha_a = extend_image_to_collar(
        image_a.alpha, image_a.origin, collared_overlap
    )
    extended_alpha_b = extend_image_to_collar(
        image_b.alpha, image_b.origin, collared_overlap
    )
    extended_mask = adjust_mask_for_transparency(
        extended_mask, extended_alpha_a, extended_alpha_b
    )
    return blend_images_with_alpha(
        extended_colors_a,
        extended_colors_b,
        extended_alpha_a,
        extended_alpha_b,
        extended_mask,
        levels,
        kernel_size,
    )
