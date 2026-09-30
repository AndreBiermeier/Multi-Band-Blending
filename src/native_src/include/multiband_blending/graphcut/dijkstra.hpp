// Declare shortest-path search that retains the crossed primal edges.
#pragma once

#include <vector>

#include "multiband_blending/graphcut/dual_grid.hpp"

namespace multiband_blending::graphcut {
    
std::vector<PrimalEdge> dijkstra_shortest_path(
    const DualGridGraph& graph, DualNodeId source, DualNodeId target);

} // namespace multiband_blending::graphcut
