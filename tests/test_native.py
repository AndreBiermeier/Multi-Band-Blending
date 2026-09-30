"""Native graph-cut buffer safety, terminal constraints and output ownership."""

import numpy as np
import pytest

from multiband_blending import _native


def test_disconnected_terminal_arcs_are_rejected():
    image = np.zeros((3, 3), dtype=np.float32)
    labels = np.array([[1, 0, -1], [0, 0, 0], [1, 0, -1]], np.int8)
    with pytest.raises(ValueError):
        _native.graphcut_mask_dijkstra(image, image, labels)


@pytest.mark.parametrize("argument", [0, 1, 2])
def test_native_rejects_incorrect_dtype(argument):
    arrays = [
        np.zeros((3, 3), np.float32),
        np.zeros((3, 3), np.float32),
        np.array([[1, 0, -1]] * 3, np.int8),
    ]
    arrays[argument] = arrays[argument].astype(np.float64)
    with pytest.raises(TypeError):
        _native.graphcut_mask_dijkstra(*arrays)


@pytest.mark.parametrize("argument", [0, 1, 2])
def test_native_rejects_noncontiguous_arrays(argument):
    arrays = [
        np.zeros((3, 3), np.float32),
        np.zeros((3, 3), np.float32),
        np.array([[1, 0, -1]] * 3, np.int8),
    ]
    arrays[argument] = arrays[argument][:, ::-1]
    with pytest.raises(ValueError):
        _native.graphcut_mask_dijkstra(*arrays)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_native_rejects_invalid_patch_values(value):
    image_a = np.zeros((3, 3), np.float32)
    image_b = image_a.copy()
    image_b[1, 1] = value
    labels = np.array([[1, 0, -1]] * 3, np.int8)
    with pytest.raises(ValueError):
        _native.graphcut_mask_dijkstra(image_a, image_b, labels)


@pytest.mark.parametrize(
    "labels",
    [
        np.zeros((3, 3), np.int8),
        np.array([[1, 2, -1]] * 3, np.int8),
        np.array([[1, 0, -1], [1, 1, -1], [1, 0, -1]], np.int8),
    ],
)
def test_native_rejects_invalid_terminal_labels(labels):
    image = np.zeros((3, 3), np.float32)
    with pytest.raises(ValueError):
        _native.graphcut_mask_dijkstra(image, image, labels)


def test_native_result_owns_its_memory():
    image = np.zeros((3, 3), np.float32)
    labels = np.array([[1, 0, -1]] * 3, np.int8)
    original_mask = _native.graphcut_mask_dijkstra(image, image, labels)
    expected = original_mask.copy()
    another_mask = _native.graphcut_mask_dijkstra(image, image, labels)
    another_mask[:] = 0.5
    np.testing.assert_array_equal(original_mask, expected)
    assert original_mask.flags.owndata
