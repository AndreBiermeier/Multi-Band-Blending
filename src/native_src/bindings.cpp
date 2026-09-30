// Expose the native graph-cut solver through validated NumPy array bindings.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "multiband_blending/graphcut/dijkstra_mask.hpp"
#include "multiband_blending/graphcut/problem_validation.hpp"

namespace py = pybind11;
namespace graphcut = multiband_blending::graphcut;

namespace {

/// Check that a NumPy array has the required element type and contiguous layout.
template <typename Value>
void validate_array(const py::array& array, const char* name, const char* dtype) {
    if (!array.dtype().is(py::dtype::of<Value>())) {
        throw py::type_error(std::string{name} + " must have dtype " + dtype);
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string{name} + " must be C-contiguous");
    }
}

/// Check array shapes and borrow their data without copying the pixels.
graphcut::ProblemView problem_view(const py::array& patch_a, const py::array& patch_b,
                                  const py::array& labels) {
    validate_array<float>(patch_a, "patch_a", "float32");
    validate_array<float>(patch_b, "patch_b", "float32");
    validate_array<std::int8_t>(labels, "terminal_labels", "int8");
    if (patch_a.ndim() != 2 && patch_a.ndim() != 3) {
        throw py::value_error("patches must be 2D grayscale or 3D channel-last arrays");
    }
    if (patch_a.ndim() != patch_b.ndim()) {
        throw py::value_error("overlap image shapes must match");
    }
    for (py::ssize_t axis = 0; axis < patch_a.ndim(); ++axis) {
        if (patch_a.shape(axis) != patch_b.shape(axis)) {
            throw py::value_error("overlap image shapes must match");
        }
    }
    if (labels.ndim() != 2 || labels.shape(0) != patch_a.shape(0) ||
        labels.shape(1) != patch_a.shape(1)) {
        throw py::value_error("terminal_labels shape must match overlap height and width");
    }
    return {
        std::span{static_cast<const float*>(patch_a.data()),
                  static_cast<std::size_t>(patch_a.size())},
        std::span{static_cast<const float*>(patch_b.data()),
                  static_cast<std::size_t>(patch_b.size())},
        std::span{static_cast<const std::int8_t*>(labels.data()),
                  static_cast<std::size_t>(labels.size())},
        static_cast<std::size_t>(patch_a.shape(0)),
        static_cast<std::size_t>(patch_a.shape(1)),
        patch_a.ndim() == 3 ? static_cast<std::size_t>(patch_a.shape(2)) : 1,
    };
}

/// Validate the inputs, run the C++ solver, and return an owning NumPy mask.
py::array_t<float> graphcut_mask(const py::array& patch_a, const py::array& patch_b,
                               const py::array& labels) {
    const graphcut::ProblemView problem = problem_view(patch_a, patch_b, labels);
    graphcut::validate_problem_values(problem);
    graphcut::BinaryMask mask;
    {
        py::gil_scoped_release release;
        mask = graphcut::graphcut_mask_dijkstra(problem);
    }
    py::array_t<float> result({static_cast<py::ssize_t>(problem.height),
                              static_cast<py::ssize_t>(problem.width)});
    std::transform(mask.begin(), mask.end(), result.mutable_data(),
                   [](std::uint8_t value) { return static_cast<float>(value); });
    return result;
}

} // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Dijkstra graph-cut implementation.";
    module.def("graphcut_mask_dijkstra", &graphcut_mask,
               py::arg("patch_a").noconvert(), py::arg("patch_b").noconvert(),
               py::arg("terminal_labels").noconvert(),
               R"doc(Return a binary overlap mask using the exact Dijkstra solver.

Args:
    patch_a (np.ndarray): Contiguous float32 image A patch in [0, 1], with shape
        (H, W) or (H, W, C).
    patch_b (np.ndarray): Contiguous float32 image B patch with the same shape.
    terminal_labels (np.ndarray): Contiguous int8 array with shape (H, W).
        Values are 1 for image A, -1 for image B, and 0 for free pixels.

Returns:
    np.ndarray: An owning float32 mask with shape (H, W). One selects image A
        and zero selects image B.
)doc");
}
