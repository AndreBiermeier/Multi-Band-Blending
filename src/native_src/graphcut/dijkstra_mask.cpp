// Run the primal-grid, split-dual, shortest-path and mask-recovery stages.

#include "multiband_blending/graphcut/dijkstra_mask.hpp"

#include "multiband_blending/graphcut/dijkstra.hpp"
#include "multiband_blending/graphcut/dual_grid.hpp"
#include "multiband_blending/graphcut/primal_grid.hpp"

namespace multiband_blending::graphcut {

/// Build the graphs, find the minimum-cost seam and construct the image A mask.
BinaryMask graphcut_mask_dijkstra(const ProblemView& problem) {
    const PrimalGridGraph primal = build_kwatra_primal_grid(problem);
    const DualGridGraph dual = build_dual_grid(primal);
    const std::vector<PrimalEdge> cut_edges =
        dijkstra_shortest_path(dual, dual.first_exterior_node(), dual.second_exterior_node());
    return build_mask_from_cut(primal, cut_edges);
}

} // namespace multiband_blending::graphcut
