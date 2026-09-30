// Declare conversion from separating primal edges to a binary image-selection mask.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "multiband_blending/graphcut/primal_grid.hpp"

namespace multiband_blending::graphcut {

using BinaryMask = std::vector<std::uint8_t>;

BinaryMask build_mask_from_cut(const PrimalGridGraph& primal,
                                             std::span<const PrimalEdge> cut_edges);

} // namespace multiband_blending::graphcut
