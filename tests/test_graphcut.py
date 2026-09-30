"""Minimum-cut correctness and small-boundary regressions checked by exhaustive cuts.
"""

from itertools import product

import numpy as np
import pytest

from multiband_blending import _native


def cut_energy(differences, mask):
    """Sum costs of horizontal and vertical edges crossing the cut."""
    horizontal = differences[:, :-1] + differences[:, 1:]
    vertical = differences[:-1] + differences[1:]
    return float(
        horizontal[mask[:, :-1] != mask[:, 1:]].sum(dtype=np.float64)
        + vertical[mask[:-1] != mask[1:]].sum(dtype=np.float64)
    )


def exhaustive_cut(differences, labels):
    """Find the minimum energy by enumerating all unconstrained pixels."""
    free = np.argwhere(labels == 0)
    assert len(free) <= 12
    best = float("inf")
    for values in product((0, 1), repeat=len(free)):
        candidate = (labels == 1).astype(np.float32)
        for position, value in zip(free, values, strict=True):
            candidate[tuple(position)] = value
        best = min(best, cut_energy(differences, candidate))
    return best


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 12, 42])
def test_dijkstra_matches_exhaustive_minimum(diagonal, color, seed):
    rng = np.random.default_rng(seed)
    shape = (3, 3, 3) if color else (3, 3)
    image_a = np.zeros(shape, dtype=np.float32)
    image_b = rng.random(shape, dtype=np.float32)
    if diagonal:
        labels = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype=np.int8)
    else:
        labels = np.array([[1, 0, -1]] * 3, dtype=np.int8)
    differences = np.linalg.norm(image_b, axis=2) if color else image_b
    mask = _native.graphcut_mask_dijkstra(image_a, image_b, labels)
    assert mask.dtype == np.float32
    assert mask.flags.c_contiguous
    assert set(np.unique(mask)) <= {0, 1}
    np.testing.assert_array_equal(mask[labels == 1], 1)
    np.testing.assert_array_equal(mask[labels == -1], 0)
    assert cut_energy(differences, mask) == pytest.approx(
        exhaustive_cut(differences, labels)
    )


def test_unique_unbiased_energy_regression():
    image_a = np.zeros((3, 4), dtype=np.float32)
    differences = np.array(
        [
            [0.002, 0.005, 0.006, 0.009],
            [0.009, 0.003, 0.002, 0.009],
            [0.000, 0.004, 0.005, 0.009],
        ],
        dtype=np.float32,
    )
    labels = np.array([[1, 0, 0, -1]] * 3, dtype=np.int8)
    actual = _native.graphcut_mask_dijkstra(image_a, differences, labels)
    expected = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], np.float32)
    assert cut_energy(differences, actual) == pytest.approx(0.023)
    np.testing.assert_array_equal(actual, expected)


def test_known_horizontal_mask_regression():
    image_a = np.zeros((3, 3), dtype=np.float32)
    image_b = np.array([[0, 0.1, 0.3], [0.2, 0.4, 0.8], [0.5, 0.7, 1]], np.float32)
    labels = np.array([[1, 0, -1]] * 3, np.int8)
    actual = _native.graphcut_mask_dijkstra(
        patch_a=image_a, patch_b=image_b, terminal_labels=labels
    )
    np.testing.assert_array_equal(actual, np.array([[1, 0, 0]] * 3, np.float32))


@pytest.mark.parametrize(
    "labels",
    [
        [[1, 1], [-1, -1]],
        [[1, -1], [1, -1]],
        [[1, 0], [0, -1]],
    ],
    ids=["adjacent-horizontal-arcs", "adjacent-vertical-arcs", "free-corners"],
)
def test_two_by_two_overlap_handles_the_single_bounded_dual_face(labels):
    labels = np.array(labels, dtype=np.int8)
    differences = np.array([[0.1, 0.8], [0.3, 0.6]], dtype=np.float32)
    mask = _native.graphcut_mask_dijkstra(
        np.zeros_like(differences), differences, labels
    )
    np.testing.assert_array_equal(mask[labels == 1], 1)
    np.testing.assert_array_equal(mask[labels == -1], 0)
    assert cut_energy(differences, mask) == pytest.approx(
        exhaustive_cut(differences, labels), abs=1e-8
    )


@pytest.mark.parametrize("top_is_cheaper", [True, False])
def test_parallel_corner_edges_recover_the_cheaper_primal_edge(top_is_cheaper):
    differences = np.array(
        [[0.0, 0.1, 0.2], [0.0, 0.0, 0.9], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    if not top_is_cheaper:
        differences[0, 1], differences[1, 2] = 0.9, 0.1
    labels = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype=np.int8)
    mask = _native.graphcut_mask_dijkstra(
        np.zeros_like(differences), differences, labels
    )
    # Both corner edges connect the same two dual nodes, but assigning the
    # free corner requires knowing exactly which primal edge was crossed.
    assert mask[0, 2] == (0 if top_is_cheaper else 1)
    np.testing.assert_array_equal(mask[labels == 1], 1)
    np.testing.assert_array_equal(mask[labels == -1], 0)
    assert cut_energy(differences, mask) == pytest.approx(
        exhaustive_cut(differences, labels), abs=1e-8
    )
