// Check native buffer sizes, normalized image values and terminal constraints.
#include "multiband_blending/graphcut/problem_validation.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace multiband_blending::graphcut {
namespace {

/// Reject nonfinite colors and values outside the normalized range [0, 1].
void validate_patch_values(const std::span<const float> patch, const char* name) {
    for (const float value : patch) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string{name} + " values must be finite");
        }
        if (value < 0.0F || value > 1.0F) {
            throw std::invalid_argument(std::string{name} + " must contain values in [0, 1]");
        }
    }
}

/// Multiply dimensions, rejecting a result that would overflow the size type.
std::size_t checked_product(const std::size_t first, const std::size_t second,
                            const char* description) {
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::invalid_argument(std::string{"graph cut problem "} + description +
                                    " is too large");
    }
    return first * second;
}

} // namespace

/// Check buffer sizes, color values and boundary labels before building the graph.
void validate_problem_values(const ProblemView& problem) {
    if (problem.height < 2 || problem.width < 2) {
        throw std::invalid_argument(
            "graph cut problems require overlap height and width at least 2");
    }
    if (problem.channels == 0) {
        throw std::invalid_argument("overlap patches must contain at least one channel");
    }

    const std::size_t pixel_count =
        checked_product(problem.height, problem.width, "pixel count");
    const std::size_t patch_value_count =
        checked_product(pixel_count, problem.channels, "patch storage");
    if (problem.patch_a.size() != patch_value_count ||
        problem.patch_b.size() != patch_value_count) {
        throw std::invalid_argument("overlap patch storage does not match its dimensions");
    }
    if (problem.terminal_labels.size() != pixel_count) {
        throw std::invalid_argument("terminal label storage does not match the overlap dimensions");
    }

    validate_patch_values(problem.patch_a, "patch_a");
    validate_patch_values(problem.patch_b, "patch_b");

    bool has_image_a = false;
    bool has_image_b = false;
    for (std::size_t row = 0; row < problem.height; ++row) {
        for (std::size_t column = 0; column < problem.width; ++column) {
            const std::int8_t label = problem.terminal_labels[row * problem.width + column];
            if (label != kImageBLabel && label != kFreeLabel && label != kImageALabel) {
                throw std::invalid_argument("terminal_labels may contain only -1, 0, and 1");
            }

            const bool is_boundary =
                row == 0 || row + 1 == problem.height || column == 0 || column + 1 == problem.width;
            if (!is_boundary && label != kFreeLabel) {
                throw std::invalid_argument(
                    "terminal labels may constrain only overlap-boundary pixels");
            }

            has_image_a = has_image_a || label == kImageALabel;
            has_image_b = has_image_b || label == kImageBLabel;
        }
    }

    if (!has_image_a || !has_image_b) {
        throw std::invalid_argument("terminal_labels must contain both image A and image B labels");
    }
}

} // namespace multiband_blending::graphcut
