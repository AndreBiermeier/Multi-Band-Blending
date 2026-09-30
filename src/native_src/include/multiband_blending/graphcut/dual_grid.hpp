// Define the split dual graph and its implicit neighbor interface.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "multiband_blending/graphcut/primal_grid.hpp"

namespace multiband_blending::graphcut {

using DualNodeId = std::size_t;

/// One dual neighbor plus the crossed primal edge needed to recover a cut.
struct DualNeighbor {
    DualNodeId node;
    double cost;
    PrimalEdge primal_edge;
};

/// One weighted dual edge and the primal edge it crosses.
struct DualEdge {
    DualNodeId first;
    DualNodeId second;
    double cost;
    PrimalEdge primal_edge;
};

/// Neighbors for bounded faces and the two exterior nodes.
class DualNeighborList {
  public:
    using const_iterator = const DualNeighbor*;

    const_iterator begin() const noexcept;

    const_iterator end() const noexcept;

  private:
    friend class DualGridGraph;

    void append(DualNeighbor neighbor) noexcept;

    void use_external(const std::vector<DualNeighbor>& neighbors) noexcept;

    std::array<DualNeighbor, 4> inline_neighbors_{};
    const std::vector<DualNeighbor>* external_neighbors_ = nullptr;
    std::size_t size_ = 0;
};

/// Non-owning implicit split dual of a terminal-contracted primal grid.
class DualGridGraph {
  public:
    std::size_t face_height() const noexcept;

    std::size_t face_width() const noexcept;

    std::size_t bounded_face_count() const noexcept;
    
    std::size_t node_count() const noexcept;

    DualNodeId face_at(std::size_t row, std::size_t column) const;

    GridCoordinate face_coordinate(DualNodeId node) const;

    DualNodeId first_exterior_node() const noexcept;

    DualNodeId second_exterior_node() const noexcept;

    DualNeighborList neighbors(DualNodeId node) const;

    std::optional<DualEdge> edge_crossing(PrimalNodeId first, PrimalNodeId second) const;

  private:
    friend DualGridGraph build_dual_grid(const PrimalGridGraph& primal);

    DualGridGraph(const PrimalGridGraph& primal, std::vector<std::int8_t> top_gaps,
                  std::vector<std::int8_t> right_gaps,
                  std::vector<std::int8_t> bottom_gaps,
                  std::vector<std::int8_t> left_gaps,
                  std::array<std::vector<DualNeighbor>, 2> exterior_neighbors);

    DualNodeId exterior_node(std::int8_t gap) const noexcept;

    const PrimalGridGraph* primal_;
    std::vector<std::int8_t> top_gaps_;
    std::vector<std::int8_t> right_gaps_;
    std::vector<std::int8_t> bottom_gaps_;
    std::vector<std::int8_t> left_gaps_;
    std::array<std::vector<DualNeighbor>, 2> exterior_neighbors_;
};

/// This applies the terminal contraction and outer-face split described by
/// Hassin, "Maximum Flow in (s, t) Planar Networks," Information Processing
/// Letters 13(3), 1981.
DualGridGraph build_dual_grid(const PrimalGridGraph& primal);

/// Prevent construction of a non-owning dual view.
DualGridGraph build_dual_grid(PrimalGridGraph&& primal) = delete;

} // namespace multiband_blending::graphcut
