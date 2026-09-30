"""Combine Laplacian bands with opaque or alpha-aware source weights.

Multi-band combination follows Burt and Adelson's A Multiresolution Spline
with Application to Image Mosaics (1983).

Alpha-normalized color and selection pyramids are a project extension: alpha
represents available image data and is never a Laplacian color channel. Empty
samples are extrapolated from coarser levels, seeded by the available-data mean,
so missing data cannot introduce artificial black detail bands.
"""

import numpy as np

from ._pyramid import expand, gaussian_pyramid, laplacian_pyramid


def blend_opaque_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask: np.ndarray,
    levels: int,
    kernel_size: int,
) -> np.ndarray:
    """Blend opaque images using multi-band blending..

    Args:
        image_a (np.ndarray): Normalized float32 grayscale or RGB image A.
        image_b (np.ndarray): Image B with the same shape.
        mask (np.ndarray): Two-dimensional image A weights, in [0, 1].
        levels (int): Level count including the original.
        kernel_size (int): Positive, odd filter width.

    Returns:
        np.ndarray: Normalized float32 colors with the input shape.
    """
    return _blend_pyramids(
        gaussian_pyramid(image_a, levels, kernel_size),
        gaussian_pyramid(image_b, levels, kernel_size),
        gaussian_pyramid(mask, levels, kernel_size),
        kernel_size,
    )


def blend_images_with_alpha(
    color_a: np.ndarray,
    color_b: np.ndarray,
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
    mask: np.ndarray,
    levels: int,
    kernel_size: int,
) -> np.ndarray:
    """Blend alpha-normalized colors with available-data selection weights.

    Args:
        color_a (np.ndarray): Normalized channel-last colors from image A.
        color_b (np.ndarray): Normalized colors from image B with the same shape.
        alpha_a (np.ndarray): Image A's alpha, separate from color.
        alpha_b (np.ndarray): Image B's alpha, separate from color.
        mask (np.ndarray): Effective image A weights, corrected for missing data.
        levels (int): Validated pyramid depth.
        kernel_size (int): Validated binomial filter width.

    Returns:
        np.ndarray: Straight colors; output opacity is calculated separately at full
            resolution.
    """
    pyramid_a = _alpha_color_pyramid(color_a, alpha_a, levels, kernel_size)
    pyramid_b = _alpha_color_pyramid(color_b, alpha_b, levels, kernel_size)

    # Adjust weights based on transparency.
    weights_a = gaussian_pyramid(mask * alpha_a, levels, kernel_size)
    weights_b = gaussian_pyramid((1.0 - mask) * alpha_b, levels, kernel_size)
    weights = []
    total_weights = []
    for weight_a, weight_b in zip(weights_a, weights_b, strict=True):
        total = weight_a + weight_b
        total_weights.append(total)
        weights.append(
            np.divide(weight_a, total, out=np.full_like(total, 0.5), where=total > 0)
        )

    # If a pixel is transparent in both images, we choose the average weight for image A.
    total_weight = total_weights[0].sum(dtype=np.float64)
    mean_weight = (
        weights_a[0].sum(dtype=np.float64) / total_weight if total_weight > 0 else 0.5
    )
    _fill_missing_levels(weights, total_weights, mean_weight, kernel_size)
    return _blend_pyramids(pyramid_a, pyramid_b, weights, kernel_size)


def _blend_pyramids(
    gaussian_a: list[np.ndarray],
    gaussian_b: list[np.ndarray],
    gaussian_mask: list[np.ndarray],
    kernel_size: int,
) -> np.ndarray:
    """Blend Laplacian bands and reconstruct their image.

    Args:
        gaussian_a (list[np.ndarray]): Image A's Gaussian color levels.
        gaussian_b (list[np.ndarray]): Image B's Gaussian color levels.
        gaussian_mask (list[np.ndarray]): Mask A's Gaussian.
        kernel_size (int): Expansion filter width.

    Returns:
        np.ndarray: Reconstructed float32 colors clipped to [0, 1].
    """
    bands_a = laplacian_pyramid(gaussian_a, kernel_size)
    bands_b = laplacian_pyramid(gaussian_b, kernel_size)
    blended = []
    for band_a, band_b, weight in zip(bands_a, bands_b, gaussian_mask, strict=True):
        if band_a.ndim == 3:
            weight = weight[..., None]
        blended.append(weight * band_a + (1.0 - weight) * band_b)
    result = blended[-1]
    for band in reversed(blended[:-1]):
        result = expand(result, band.shape, kernel_size) + band
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def _alpha_color_pyramid(
    color: np.ndarray, alpha: np.ndarray, levels: int, kernel_size: int
) -> list[np.ndarray]:
    """Build Gaussian colors normalized by the available image data.

    # Blur alpha-weighted colors, then divide by blurred alpha so unavailable
    # pixels do not contribute to the average. This preserves constant colors.

    Args:
        color (np.ndarray): Normalized float32 colors with shape (H, W, C).
        alpha (np.ndarray): Float32 opacity with shape (H, W).
        levels (int): Number of pyramid levels.
        kernel_size (int): Binomial filter width.

    Returns:
        list[np.ndarray]: Color levels ordered finest first. Transparent samples
            are filled from coarser levels, using the visible mean as a fallback.
    """
    coverage_pyramid = gaussian_pyramid(alpha, levels, kernel_size)
    premultiplied_pyramid = gaussian_pyramid(
        color * alpha[..., None], levels, kernel_size
    )
    normalized_pyramid = []
    for premultiplied, coverage in zip(
        premultiplied_pyramid, coverage_pyramid, strict=True
    ):
        coverage = coverage[..., None]
        normalized_pyramid.append(
            np.divide(
                premultiplied,
                coverage,
                out=np.zeros_like(premultiplied),
                where=coverage > 0,
            )
        )

    # Fix missing spots.
    total_coverage = alpha.sum(dtype=np.float64)
    mean_color = (
        premultiplied_pyramid[0].sum(axis=(0, 1), dtype=np.float64) / total_coverage
        if total_coverage > 0
        else np.zeros(color.shape[2], dtype=np.float32)
    )
    _fill_missing_levels(normalized_pyramid, coverage_pyramid, mean_color, kernel_size)
    return normalized_pyramid


def _fill_missing_levels(
    pyramid: list[np.ndarray],
    support: list[np.ndarray],
    fallback: np.ndarray | float,
    kernel_size: int,
) -> None:
    """Fill unsupported samples from coarse to fine, preserving supported values.

    The idea is that small unsupported parts at the original resolution get smoothed
    out when going down the pyramid (which includes blur).
    We then use the coarser levels to reconstruct the higher levels.
    The fallback is mostly required when there are large unsupported areas
    that the downscaling doesn't get rid of (for example not enough levels).

    Args:
        pyramid (list[np.ndarray]): Normalized levels to fill in place, finest first.
        support (list[np.ndarray]): Nonnegative available-data weights for each level.
        fallback (np.ndarray | float): Available-data mean used for empty coarsest
            samples.
        kernel_size (int): Filter width used when expanding coarser samples.

    Returns:
        None: The pyramid levels are updated in place.
    """
    pyramid[-1][support[-1] == 0] = fallback
    for index in range(len(pyramid) - 2, -1, -1):
        expanded = expand(pyramid[index + 1], pyramid[index].shape, kernel_size)
        missing = support[index] == 0
        pyramid[index][missing] = expanded[missing]
