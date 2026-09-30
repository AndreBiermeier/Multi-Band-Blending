"""Assemble source colors, the blended collared overlap and alpha on the canvas."""

import numpy as np

from ._image import PreparedImage
from .geometry import ImageOrigin, OverlapRegion, placement_slice


def assemble_canvas(
    image_a: PreparedImage,
    image_b: PreparedImage,
    overlap: OverlapRegion,
    blended_colors: np.ndarray,
    collared_overlap_origin: ImageOrigin,
    mask: np.ndarray,
) -> np.ndarray:
    """Place sources, the blended collared overlap and alpha on the canvas.

    Args:
        image_a (PreparedImage): Image A's normalized colors, alpha and placement.
        image_b (PreparedImage): Image B's normalized colors, alpha and placement.
        overlap (OverlapRegion): Overlap and canvas geometry.
        blended_colors (np.ndarray): Reconstructed colors for the collared overlap.
        collared_overlap_origin (ImageOrigin): World origin of the collared overlap rectangle.
        mask (np.ndarray): Effective overlap weights.

    Returns:
        np.ndarray: A float32 canvas with one alpha channel and zero color in
            transparent gaps.
    """
    canvas = np.zeros(
        (*overlap.canvas_shape, image_a.colors.shape[2] + 1), dtype=np.float32
    )
    canvas_colors = canvas[..., :-1]
    canvas_alpha = canvas[..., -1]

    # Place original images and collared overlap.
    for image in (image_a, image_b):
        location = placement_slice(
            image.origin, image.colors.shape, overlap.canvas_origin
        )
        canvas_colors[location] = image.colors
        canvas_alpha[location] = image.alpha
    collared_overlap_slice = placement_slice(
        collared_overlap_origin, blended_colors.shape, overlap.canvas_origin
    )
    canvas_colors[collared_overlap_slice] = blended_colors

    # Alpha inside overlap.
    overlap_alpha_a = image_a.alpha[overlap.image_a_slice]
    overlap_alpha_b = image_b.alpha[overlap.image_b_slice]
    canvas_alpha[overlap.canvas_slice] = (
        mask * overlap_alpha_a + (1.0 - mask) * overlap_alpha_b
    )

    # Restore source colors where only one image has positive alpha.
    overlap_colors = canvas_colors[overlap.canvas_slice]
    for color, alpha, other_alpha, source_slice in (
        (image_a.colors, overlap_alpha_a, overlap_alpha_b, overlap.image_a_slice),
        (image_b.colors, overlap_alpha_b, overlap_alpha_a, overlap.image_b_slice),
    ):
        only_source = (alpha > 0) & (other_alpha == 0)
        overlap_colors[only_source] = color[source_slice][only_source]

    canvas_colors[canvas_alpha == 0] = 0.0
    return canvas
