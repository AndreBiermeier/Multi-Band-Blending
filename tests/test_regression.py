"""Complete numerical pipeline regressions preserved from the original thesis code.
The regression tests were created manually by hand initially and visually verified.
The regression cases are the stored results.
"""

import numpy as np
import pytest

from multiband_blending import ImageOrigin, blend

from .regression_cases import EXPECTED_IMAGES, EXPECTED_MASKS


@pytest.mark.parametrize("mode", ["gray", "rgb"])
@pytest.mark.parametrize("method", ["linear", "cosine", "graphcut"])
def test_opaque_pipeline_matches_original_snapshot(mode, method):
    rng = np.random.default_rng(1729)
    shape = (5, 9) if mode == "gray" else (5, 9, 3)
    image_a = rng.random(shape, dtype=np.float32)
    image_b = rng.random(shape, dtype=np.float32)
    result = blend(
        image_a,
        image_b,
        ImageOrigin(-2, -1),
        ImageOrigin(-1, 2),
        mask=method,
        levels=2,
    )
    key = f"{mode}_{method}"
    actual_colors = result.blended_image[..., :-1]
    if mode == "gray":
        actual_colors = actual_colors[..., 0]
    np.testing.assert_allclose(actual_colors, EXPECTED_IMAGES[key], rtol=0, atol=2e-7)
    np.testing.assert_array_equal(result.overlap_mask, EXPECTED_MASKS[key])
    expected_alpha = np.zeros((6, 12), dtype=np.float32)
    expected_alpha[:5, :9] = 1
    expected_alpha[1:, 3:] = 1
    np.testing.assert_array_equal(result.blended_image[..., -1], expected_alpha)
    assert result.canvas_origin == ImageOrigin(-2, -1)


def test_collared_pipeline_preserves_original_numerical_result():
    image_a = np.linspace(0.05, 0.95, 30, dtype=np.float32).reshape(3, 10)
    image_b = np.flip(image_a, axis=(0, 1)).copy()
    actual = blend(image_a, image_b, origin_b=ImageOrigin(x=8), mask="linear", levels=2)
    expected = np.array(
        [
            [
                0.05,
                0.081034482,
                0.112068966,
                0.143103451,
                0.174137935,
                0.205172420,
                0.243253291,
                0.302473396,
                0.385105759,
                0.731455386,
                0.799510002,
                0.830483913,
                0.820406795,
                0.793736517,
                0.763793111,
                0.732758641,
                0.701724112,
                0.670689642,
            ],
            [
                0.360344827,
                0.391379297,
                0.422413796,
                0.453448266,
                0.484482765,
                0.515517235,
                0.549809694,
                0.593876302,
                0.646201551,
                0.520366371,
                0.538414299,
                0.536050320,
                0.513850331,
                0.484149396,
                0.453448266,
                0.422413796,
                0.391379297,
                0.360344827,
            ],
            [
                0.670689642,
                0.701724112,
                0.732758641,
                0.763793111,
                0.794827580,
                0.825862050,
                0.854471982,
                0.875808179,
                0.882672668,
                0.358905166,
                0.301943094,
                0.249572396,
                0.209188133,
                0.174941093,
                0.143103451,
                0.112068981,
                0.081034482,
                0.05,
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        actual.blended_image[..., 0], expected, rtol=0, atol=1e-7
    )
    np.testing.assert_array_equal(actual.blended_image[..., 1], 1)
    np.testing.assert_array_equal(
        actual.overlap_mask, np.array([[1, 0]] * 3, np.float32)
    )
