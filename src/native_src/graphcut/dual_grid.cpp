// Build and traverse the split dual graph used to find a minimum-cost seam.
//
// Used:
// Hassin, "Maximum Flow in (s,t) Planar Networks", Information Processing
// Letters 13(3), 1981.
#include "multiband_blending/graphcut/dual_grid.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace multiband_blending::graphcut {
namespace {

constexpr std::int8_t kContractedBoundaryEdge = -1;

struct BoundaryData {
    std::vector<std::int8_t> top_gaps;
    std::vector<std::int8_t> right_gaps;
    std::vector<std::int8_t> bottom_gaps;
    std::vector<std::int8_t> left_gaps;
    std::array<std::vector<DualNeighbor>, 2> exterior_neighbors;
};

/// List boundary pixels clockwise from the top-left corner (no duplicates).
std::vector<GridCoordinate> perimeter_coordinates(const std::size_t height,
                                                  const std::size_t width) {
    std::vector<GridCoordinate> perimeter;
    perimeter.reserve(2 * height + 2 * width - 4);

    for (std::size_t column = 0; column < width; ++column) {
        perimeter.push_back({0, column});
    }
    for (std::size_t row = 1; row < height; ++row) {
        perimeter.push_back({row, width - 1});
    }
    for (std::size_t column = width - 1; column-- > 0;) {
        perimeter.push_back({height - 1, column});
    }
    for (std::size_t row = height - 1; row-- > 1;) {
        perimeter.push_back({row, 0});
    }
    return perimeter;
}

/// Check that the target label occupies one uninterrupted arc of the cyclic boundary.
bool is_single_circular_run(const std::vector<std::int8_t>& labels,
                            const std::int8_t target) {
    std::size_t run_starts = 0;
    for (std::size_t index = 0; index < labels.size(); ++index) {
        const std::size_t previous = (index + labels.size() - 1) % labels.size();
        if (labels[index] == target && labels[previous] != target) {
            ++run_starts;
        }
    }
    return run_starts == 1;
}

/// Convert two pixel coordinates to an edge with node identifiers in ascending order.
PrimalEdge canonical_primal_edge(const PrimalGridGraph& primal, const GridCoordinate first,
                                 const GridCoordinate second) {
    const PrimalNodeId first_node = primal.node_at(first.row, first.column);
    const PrimalNodeId second_node = primal.node_at(second.row, second.column);
    return {std::min(first_node, second_node), std::max(first_node, second_node)};
}

/// Find the interior grid face touching a boundary edge.
DualNodeId adjacent_bounded_face(const PrimalGridGraph& primal, const GridCoordinate first,
                                 const GridCoordinate second) {
    const std::size_t face_width = primal.width() - 1;
    if (first.row == second.row) {
        const std::size_t column = std::min(first.column, second.column);
        const std::size_t row = first.row == 0 ? 0 : primal.height() - 2;
        return row * face_width + column;
    }

    const std::size_t row = std::min(first.row, second.row);
    const std::size_t column = first.column == 0 ? 0 : primal.width() - 2;
    return row * face_width + column;
}

/// Assign a boundary edge to one exterior node and record its interior-face connection.
void assign_boundary_gap(const PrimalGridGraph& primal, BoundaryData& data,
                         const GridCoordinate first, const GridCoordinate second,
                         const std::int8_t gap) {
    if (first.row == second.row) {
        const std::size_t column = std::min(first.column, second.column);
        if (first.row == 0) {
            data.top_gaps[column] = gap;
        } else {
            data.bottom_gaps[column] = gap;
        }
    } else {
        const std::size_t row = std::min(first.row, second.row);
        if (first.column == 0) {
            data.left_gaps[row] = gap;
        } else {
            data.right_gaps[row] = gap;
        }
    }

    const PrimalEdge primal_edge = canonical_primal_edge(primal, first, second);
    data.exterior_neighbors[static_cast<std::size_t>(gap)].push_back(
        {adjacent_bounded_face(primal, first, second),
         primal.edge_cost(primal_edge.first, primal_edge.second), primal_edge});
}

/// Validate terminal arcs and map the two boundary routes between them to exterior nodes.
BoundaryData build_boundary_data(const PrimalGridGraph& primal) {
    const std::vector<GridCoordinate> perimeter =
        perimeter_coordinates(primal.height(), primal.width());
    std::vector<std::int8_t> labels;
    labels.reserve(perimeter.size());
    for (const GridCoordinate coordinate : perimeter) {
        labels.push_back(primal.terminal_label(primal.node_at(coordinate.row, coordinate.column)));
    }

    if (!is_single_circular_run(labels, kImageALabel) ||
        !is_single_circular_run(labels, kImageBLabel)) {
        throw std::invalid_argument(
            "dual graph requires each terminal label to form one connected boundary arc");
    }

    BoundaryData data{
        std::vector<std::int8_t>(primal.width() - 1, kContractedBoundaryEdge),
        std::vector<std::int8_t>(primal.height() - 1, kContractedBoundaryEdge),
        std::vector<std::int8_t>(primal.width() - 1, kContractedBoundaryEdge),
        std::vector<std::int8_t>(primal.height() - 1, kContractedBoundaryEdge),
        {},
    };

    const std::size_t perimeter_size = perimeter.size();
    const auto next = [perimeter_size](const std::size_t index) {
        return (index + 1) % perimeter_size;
    };

    // Go to beginning of A terminals.
    std::size_t current = 0;
    while (!(labels[current] == kImageALabel &&
             labels[(current + perimeter_size - 1) % perimeter_size] != kImageALabel)) {
        ++current;
    }
    // Go to end of A terminals and keep them contracted.
    while (labels[next(current)] == kImageALabel) {
        current = next(current);
    }
    // Go to beginning of B terminals and assign the walked gap.
    while (labels[current] != kImageBLabel) {
        const std::size_t following = next(current);
        assign_boundary_gap(primal, data, perimeter[current], perimeter[following], 0);
        current = following;
    }
    // Go to end of B terminals and keep them contracted.
    while (labels[next(current)] == kImageBLabel) {
        current = next(current);
    }
    // Go to beginning of A terminals and assign the walked gap.
    while (labels[current] != kImageALabel) {
        const std::size_t following = next(current);
        assign_boundary_gap(primal, data, perimeter[current], perimeter[following], 1);
        current = following;
    }

    return data;
}

} // namespace

/// Return the start of the neighbor range.
DualNeighborList::const_iterator DualNeighborList::begin() const noexcept {
    return external_neighbors_ == nullptr ? inline_neighbors_.data()
                                          : external_neighbors_->data();
}

/// Return the position after the last populated neighbor.
DualNeighborList::const_iterator DualNeighborList::end() const noexcept {
    return size_ == 0 ? begin() : begin() + size_;
}

/// Add one of an interior face's 4 neighbors.
void DualNeighborList::append(const DualNeighbor neighbor) noexcept {
    inline_neighbors_[size_] = neighbor;
    ++size_;
}

/// Borrow an exterior node's precomputed neighbors instead of copying them.
void DualNeighborList::use_external(const std::vector<DualNeighbor>& neighbors) noexcept {
    external_neighbors_ = &neighbors;
    size_ = neighbors.size();
}

/// Borrow the primal graph and take ownership of the split-boundary connections.
DualGridGraph::DualGridGraph(const PrimalGridGraph& primal, std::vector<std::int8_t> top_gaps,
                             std::vector<std::int8_t> right_gaps,
                             std::vector<std::int8_t> bottom_gaps,
                             std::vector<std::int8_t> left_gaps,
                             std::array<std::vector<DualNeighbor>, 2> exterior_neighbors)
    : primal_{&primal}, top_gaps_{std::move(top_gaps)}, right_gaps_{std::move(right_gaps)},
      bottom_gaps_{std::move(bottom_gaps)}, left_gaps_{std::move(left_gaps)},
      exterior_neighbors_{std::move(exterior_neighbors)} {}

/// Return the number of interior face rows between pixel rows.
std::size_t DualGridGraph::face_height() const noexcept {
    return primal_->height() - 1;
}

/// Return the number of interior face columns between pixel columns.
std::size_t DualGridGraph::face_width() const noexcept {
    return primal_->width() - 1;
}

/// Count the interior faces, excluding the two exterior nodes.
std::size_t DualGridGraph::bounded_face_count() const noexcept {
    return face_height() * face_width();
}

/// Count all interior faces and both exterior nodes.
std::size_t DualGridGraph::node_count() const noexcept {
    return bounded_face_count() + 2;
}

/// Convert an interior face's row and column to its row-major node identifier.
DualNodeId DualGridGraph::face_at(const std::size_t row, const std::size_t column) const {
    return row * face_width() + column;
}

/// Convert an interior face identifier back to its row and column.
GridCoordinate DualGridGraph::face_coordinate(const DualNodeId node) const {
    return {node / face_width(), node % face_width()};
}

/// Return the exterior node on the clockwise boundary route from image A to image B.
DualNodeId DualGridGraph::first_exterior_node() const noexcept {
    return bounded_face_count();
}

/// Return the exterior node on the other boundary route between the terminal arcs.
DualNodeId DualGridGraph::second_exterior_node() const noexcept {
    return bounded_face_count() + 1;
}

/// Collect adjacent faces and the primal edges crossed to reach them.
DualNeighborList DualGridGraph::neighbors(const DualNodeId node) const {
    DualNeighborList result;

    // Exterior node.
    if (node >= bounded_face_count()) {
        result.use_external(exterior_neighbors_[node - bounded_face_count()]);
        return result;
    }

    const auto [row, column] = face_coordinate(node);
    const auto append_crossing = [this, node, &result](const PrimalNodeId first,
                                                       const PrimalNodeId second) {
        const std::optional<DualEdge> edge = edge_crossing(first, second);
        if (!edge.has_value()) {
            return;
        }
        const DualNodeId neighbor = edge->first == node ? edge->second : edge->first;
        result.append({neighbor, edge->cost, edge->primal_edge});
    };

    append_crossing(primal_->node_at(row, column), primal_->node_at(row, column + 1));
    append_crossing(primal_->node_at(row, column + 1),
                    primal_->node_at(row + 1, column + 1));
    append_crossing(primal_->node_at(row + 1, column),
                    primal_->node_at(row + 1, column + 1));
    append_crossing(primal_->node_at(row, column), primal_->node_at(row + 1, column));
    return result;
}

/// Find the dual edge crossing a primal edge, omitting contracted terminal-boundary edges.
std::optional<DualEdge> DualGridGraph::edge_crossing(const PrimalNodeId first,
                                                     const PrimalNodeId second) const {
    const GridCoordinate first_coordinate = primal_->coordinate(first);
    const GridCoordinate second_coordinate = primal_->coordinate(second);
    const PrimalEdge primal_edge{std::min(first, second), std::max(first, second)};
    const double cost = primal_->edge_cost(primal_edge.first, primal_edge.second);

    DualNodeId first_face = 0;
    DualNodeId second_face = 0;
    if (first_coordinate.row == second_coordinate.row) {
        const std::size_t row = first_coordinate.row;
        const std::size_t column = std::min(first_coordinate.column, second_coordinate.column);
        if (row == 0) {
            if (top_gaps_[column] == kContractedBoundaryEdge) {
                return std::nullopt;
            }
            first_face = exterior_node(top_gaps_[column]);
        } else {
            first_face = face_at(row - 1, column);
        }

        if (row + 1 == primal_->height()) {
            if (bottom_gaps_[column] == kContractedBoundaryEdge) {
                return std::nullopt;
            }
            second_face = exterior_node(bottom_gaps_[column]);
        } else {
            second_face = face_at(row, column);
        }
    } else {
        const std::size_t row = std::min(first_coordinate.row, second_coordinate.row);
        const std::size_t column = first_coordinate.column;
        if (column == 0) {
            if (left_gaps_[row] == kContractedBoundaryEdge) {
                return std::nullopt;
            }
            first_face = exterior_node(left_gaps_[row]);
        } else {
            first_face = face_at(row, column - 1);
        }

        if (column + 1 == primal_->width()) {
            if (right_gaps_[row] == kContractedBoundaryEdge) {
                return std::nullopt;
            }
            second_face = exterior_node(right_gaps_[row]);
        } else {
            second_face = face_at(row, column);
        }
    }
    return DualEdge{first_face, second_face, cost, primal_edge};
}

/// Convert a boundary-route index to its exterior node identifier.
DualNodeId DualGridGraph::exterior_node(const std::int8_t gap) const noexcept {
    return bounded_face_count() + static_cast<std::size_t>(gap);
}

/// Build a dual graph that borrows the primal grid and splits its exterior face in two.
DualGridGraph build_dual_grid(const PrimalGridGraph& primal) {
    BoundaryData boundary = build_boundary_data(primal);
    return {primal,
            std::move(boundary.top_gaps),
            std::move(boundary.right_gaps),
            std::move(boundary.bottom_gaps),
            std::move(boundary.left_gaps),
            std::move(boundary.exterior_neighbors)};
}

} // namespace multiband_blending::graphcut
