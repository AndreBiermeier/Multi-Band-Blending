// Define the weighted pixel grid and its implicit neighbor interface.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "multiband_blending/graphcut/problem_validation.hpp"

namespace multiband_blending::graphcut {

using PrimalNodeId = std::size_t;

/// Ordered endpoints of one undirected primal edge.
struct PrimalEdge {
    PrimalNodeId first;
    PrimalNodeId second;
};

/// Row and column of a pixel node in the overlap grid.
struct GridCoordinate {
    std::size_t row;
    std::size_t column;
};

/// One adjacent pixel node and the cost of the connecting undirected edge.
struct PrimalNeighbor {
    PrimalNodeId node;
    double cost;
};

/// Fixed-capacity neighbor range for a four-connected pixel node.
class PrimalNeighborList {
  public:
    using const_iterator = std::array<PrimalNeighbor, 4>::const_iterator;

    const_iterator begin() const noexcept;

    const_iterator end() const noexcept;

  private:
    friend class PrimalGridGraph;

    void append(PrimalNodeId node, double cost) noexcept;

    std::array<PrimalNeighbor, 4> neighbors_{};
    std::size_t size_ = 0;
};

/// Four-connected primal graph.
class PrimalGridGraph {
  public:
    std::size_t height() const noexcept;

    std::size_t width() const noexcept;

    std::size_t node_count() const noexcept;

    PrimalNodeId node_at(std::size_t row, std::size_t column) const;

    GridCoordinate coordinate(PrimalNodeId node) const;

    std::int8_t terminal_label(PrimalNodeId node) const;

    PrimalNeighborList neighbors(PrimalNodeId node) const;

    double edge_cost(PrimalNodeId first, PrimalNodeId second) const;

  private:
    friend PrimalGridGraph build_kwatra_primal_grid(const ProblemView& problem);

    PrimalGridGraph(std::size_t height, std::size_t width, std::vector<double> horizontal_costs,
                    std::vector<double> vertical_costs, std::vector<std::int8_t> terminal_labels);

    double horizontal_cost(std::size_t row, std::size_t left_column) const noexcept;

    double vertical_cost(std::size_t top_row, std::size_t column) const noexcept;

    std::size_t height_;
    std::size_t width_;
    std::vector<double> horizontal_costs_;
    std::vector<double> vertical_costs_;
    std::vector<std::int8_t> terminal_labels_;
};
    
/// Implements Equation (1) from Kwatra et al., "Graphcut Textures: Image and
/// Video Synthesis Using Graph Cuts," SIGGRAPH 2003:
/// M(s,t) = ||A(s)-B(s)|| + ||A(t)-B(t)||.
PrimalGridGraph build_kwatra_primal_grid(const ProblemView& problem);

} // namespace multiband_blending::graphcut
