// Define problem view.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace multiband_blending::graphcut {

inline constexpr std::int8_t kImageBLabel = -1;
inline constexpr std::int8_t kFreeLabel = 0;
inline constexpr std::int8_t kImageALabel = 1;

/// Non-owning view.
struct ProblemView {
    std::span<const float> patch_a;
    std::span<const float> patch_b;
    std::span<const std::int8_t> terminal_labels;
    std::size_t height;
    std::size_t width;
    std::size_t channels;
};

void validate_problem_values(const ProblemView& problem);

} // namespace multiband_blending::graphcut
