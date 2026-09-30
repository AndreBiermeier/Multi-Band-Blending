// Declare the native graph-cut pipeline from overlap patches to a binary mask.
#pragma once

#include "multiband_blending/graphcut/cut_mask.hpp"
#include "multiband_blending/graphcut/problem_validation.hpp"

namespace multiband_blending::graphcut {

/// Build a binary graph-cut mask using the split dual and Dijkstra's algorithm.
BinaryMask graphcut_mask_dijkstra(const ProblemView& problem);

} // namespace multiband_blending::graphcut
