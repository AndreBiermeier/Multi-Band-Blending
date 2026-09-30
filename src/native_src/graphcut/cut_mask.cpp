// Recover the primal cut represented by Hassin's split-dual path (1981).
// Flood from image A terminals while blocking every crossed primal edge.
#include "multiband_blending/graphcut/cut_mask.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace multiband_blending::graphcut {
namespace {

struct EdgeLocation {
    bool horizontal;
    std::size_t index;
};

/// Locate an adjacent pixel pair in the horizontal or vertical edge array.
EdgeLocation edge_location(const PrimalGridGraph& primal, const PrimalNodeId first,
                           const PrimalNodeId second) {
    const GridCoordinate first_coordinate = primal.coordinate(std::min(first, second));
    const GridCoordinate second_coordinate = primal.coordinate(std::max(first, second));

    if (first_coordinate.row == second_coordinate.row &&
        first_coordinate.column + 1 == second_coordinate.column) {
        return {true, first_coordinate.row * (primal.width() - 1) + first_coordinate.column};
    }

    if (first_coordinate.column == second_coordinate.column &&
        first_coordinate.row + 1 == second_coordinate.row) {
        return {false, first_coordinate.row * primal.width() + first_coordinate.column};
    }

    throw std::invalid_argument("cut edge endpoints must be distinct adjacent primal nodes");
}

class BlockedEdgeGrid {
  public:
    /// Create an edge grid with no blocked connections.
    explicit BlockedEdgeGrid(const PrimalGridGraph& primal)
        : primal_{primal}, horizontal_(primal.height() * (primal.width() - 1)),
          vertical_((primal.height() - 1) * primal.width()) {}

    /// Mark one cut edge as impassable during the flood fill.
    void block(const PrimalEdge edge) {
        const EdgeLocation location = edge_location(primal_, edge.first, edge.second);
        std::uint8_t& blocked = location.horizontal ? horizontal_[location.index]
                                                    : vertical_[location.index];
        blocked = 1;
    }

    /// Check whether the connection between two adjacent pixels is blocked.
    bool contains(const PrimalNodeId first, const PrimalNodeId second) const {
        const EdgeLocation location = edge_location(primal_, first, second);
        return location.horizontal ? horizontal_[location.index] != 0
                                   : vertical_[location.index] != 0;
    }

  private:
    const PrimalGridGraph& primal_;
    std::vector<std::uint8_t> horizontal_;
    std::vector<std::uint8_t> vertical_;
};

/// Mark pixels reachable from image A terminals without crossing a cut edge.
BinaryMask flood_image_a_region(const PrimalGridGraph& primal,
                                  const BlockedEdgeGrid& blocked_edges) {
    BinaryMask mask(primal.node_count(), 0);
    std::vector<PrimalNodeId> pending;
    pending.reserve(primal.node_count());

    for (PrimalNodeId node = 0; node < primal.node_count(); ++node) {
        if (primal.terminal_label(node) == kImageALabel) {
            mask[node] = 1;
            pending.push_back(node);
        }
    }

    while (!pending.empty()) {
        const PrimalNodeId current = pending.back();
        pending.pop_back();
        for (const PrimalNeighbor& neighbor : primal.neighbors(current)) {
            if (mask[neighbor.node] != 0 || blocked_edges.contains(current, neighbor.node)) {
                continue;
            }
            mask[neighbor.node] = 1;
            pending.push_back(neighbor.node);
        }
    }
    return mask;
}

/// Check that image B terminals remain separate and every cut edge divides the mask.
void validate_mask_boundary(const PrimalGridGraph& primal,
                            const std::span<const PrimalEdge> cut_edges,
                            const BinaryMask& mask) {
    for (PrimalNodeId node = 0; node < primal.node_count(); ++node) {
        if (primal.terminal_label(node) == kImageBLabel && mask[node] != 0) {
            throw std::invalid_argument(
                "cut does not separate Image A terminals from Image B terminals");
        }
    }

    for (const PrimalEdge edge : cut_edges) {
        if (mask[edge.first] == mask[edge.second]) {
            throw std::invalid_argument("cut contains an edge that does not separate mask regions");
        }
    }
}

} // namespace

/// Block the seam edges, flood the image A region and validate the resulting mask.
BinaryMask build_mask_from_cut(const PrimalGridGraph& primal,
                               const std::span<const PrimalEdge> cut_edges) {
    BlockedEdgeGrid blocked_edges{primal};
    for (const PrimalEdge edge : cut_edges) {
        blocked_edges.block(edge);
    }

    BinaryMask mask = flood_image_a_region(primal, blocked_edges);
    validate_mask_boundary(primal, cut_edges, mask);
    return mask;
}

} // namespace multiband_blending::graphcut
