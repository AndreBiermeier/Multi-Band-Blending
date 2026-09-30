// Build and traverse the pixel grid for the graph-cut energy.
//
// Uses:
// Kwatra et al., "Graphcut Textures: Image and Video Synthesis Using Graph
// Cuts", SIGGRAPH 2003, Equation (1): M(s,t) = ||A(s)-B(s)|| + ||A(t)-B(t)||.
#include "multiband_blending/graphcut/primal_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace multiband_blending::graphcut {
namespace {

/// Measure the Euclidean color difference between image A and image B at one pixel.
double pixel_difference(const ProblemView& problem, const std::size_t pixel) {
    const std::size_t first_channel = pixel * problem.channels;
    double squared_difference = 0.0;
    for (std::size_t channel = 0; channel < problem.channels; ++channel) {
        const std::size_t index = first_channel + channel;
        const double difference =
            static_cast<double>(problem.patch_a[index]) - problem.patch_b[index];
        squared_difference += difference * difference;
    }
    return std::sqrt(squared_difference);
}

} // namespace

/// Return the start of the stored neighbor range.
PrimalNeighborList::const_iterator PrimalNeighborList::begin() const noexcept {
    return neighbors_.begin();
}

/// Return the position after the last populated neighbor.
PrimalNeighborList::const_iterator PrimalNeighborList::end() const noexcept {
    return neighbors_.begin() + static_cast<std::ptrdiff_t>(size_);
}

/// Add one of the pixel's four neighbors and its edge cost.
void PrimalNeighborList::append(const PrimalNodeId node, const double cost) noexcept {
    neighbors_[size_] = {node, cost};
    ++size_;
}

/// Take ownership of the grid's edge costs and terminal labels.
PrimalGridGraph::PrimalGridGraph(const std::size_t height, const std::size_t width,
                                 std::vector<double> horizontal_costs,
                                 std::vector<double> vertical_costs,
                                 std::vector<std::int8_t> terminal_labels)
    : height_{height}, width_{width}, horizontal_costs_{std::move(horizontal_costs)},
      vertical_costs_{std::move(vertical_costs)}, terminal_labels_{std::move(terminal_labels)} {}

/// Return the number of pixel rows.
std::size_t PrimalGridGraph::height() const noexcept {
    return height_;
}

/// Return the number of pixel columns.
std::size_t PrimalGridGraph::width() const noexcept {
    return width_;
}

/// Return the total number of pixel nodes.
std::size_t PrimalGridGraph::node_count() const noexcept {
    return height_ * width_;
}

/// Convert a pixel's row and column to its row-major node identifier.
PrimalNodeId PrimalGridGraph::node_at(const std::size_t row, const std::size_t column) const {
    return row * width_ + column;
}

/// Convert a pixel node identifier back to its row and column.
GridCoordinate PrimalGridGraph::coordinate(const PrimalNodeId node) const {
    return {node / width_, node % width_};
}

/// Return the pixel's constraint: image A (1), image B (-1) or free (0).
std::int8_t PrimalGridGraph::terminal_label(const PrimalNodeId node) const {
    return terminal_labels_[node];
}

/// Collect the existing upper, right, lower, and left pixel neighbors with their costs.
PrimalNeighborList PrimalGridGraph::neighbors(const PrimalNodeId node) const {
    const auto [row, column] = coordinate(node);
    PrimalNeighborList result;
    if (row > 0) {
        result.append(node - width_, vertical_cost(row - 1, column));
    }
    if (column + 1 < width_) {
        result.append(node + 1, horizontal_cost(row, column));
    }
    if (row + 1 < height_) {
        result.append(node + width_, vertical_cost(row, column));
    }
    if (column > 0) {
        result.append(node - 1, horizontal_cost(row, column - 1));
    }
    return result;
}

/// Look up an edge cost, rejecting endpoints that are not distinct adjacent pixels.
double PrimalGridGraph::edge_cost(const PrimalNodeId first, const PrimalNodeId second) const {
    const GridCoordinate first_coordinate = coordinate(std::min(first, second));
    const GridCoordinate second_coordinate = coordinate(std::max(first, second));

    if (first_coordinate.row == second_coordinate.row &&
        first_coordinate.column + 1 == second_coordinate.column) {
        return horizontal_cost(first_coordinate.row, first_coordinate.column);
    }
    if (first_coordinate.column == second_coordinate.column &&
        first_coordinate.row + 1 == second_coordinate.row) {
        return vertical_cost(first_coordinate.row, first_coordinate.column);
    }
    throw std::invalid_argument("primal edge endpoints must be distinct adjacent nodes");
}

/// Return the cost of the edge from a pixel to its right neighbor.
double PrimalGridGraph::horizontal_cost(const std::size_t row,
                                        const std::size_t left_column) const noexcept {
    return horizontal_costs_[row * (width_ - 1) + left_column];
}

/// Return the cost of the edge from a pixel to its lower neighbor.
double PrimalGridGraph::vertical_cost(const std::size_t top_row,
                                      const std::size_t column) const noexcept {
    return vertical_costs_[top_row * width_ + column];
}

/// Weight each pixel connection by the sum of its endpoints' image A/B differences.
PrimalGridGraph build_kwatra_primal_grid(const ProblemView& problem) {
    const std::size_t pixel_count = problem.height * problem.width;
    std::vector<double> pixel_differences(pixel_count);
    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        pixel_differences[pixel] = pixel_difference(problem, pixel);
    }

    std::vector<double> horizontal_costs(problem.height * (problem.width - 1));
    for (std::size_t row = 0; row < problem.height; ++row) {
        for (std::size_t column = 0; column + 1 < problem.width; ++column) {
            const std::size_t pixel = row * problem.width + column;
            horizontal_costs[row * (problem.width - 1) + column] =
                pixel_differences[pixel] + pixel_differences[pixel + 1];
        }
    }

    std::vector<double> vertical_costs((problem.height - 1) * problem.width);
    for (std::size_t row = 0; row + 1 < problem.height; ++row) {
        for (std::size_t column = 0; column < problem.width; ++column) {
            const std::size_t pixel = row * problem.width + column;
            vertical_costs[row * problem.width + column] =
                pixel_differences[pixel] + pixel_differences[pixel + problem.width];
        }
    }

    std::vector<std::int8_t> terminal_labels(problem.terminal_labels.begin(),
                                            problem.terminal_labels.end());
    return {
        problem.height,
        problem.width,
        std::move(horizontal_costs),
        std::move(vertical_costs),
        std::move(terminal_labels),
    };
}

} // namespace multiband_blending::graphcut
