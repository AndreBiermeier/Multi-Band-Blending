// Classical Dijkstra shortest-path search with nonnegative edge costs.
// The binary heap permits duplicate entries; stale distances are discarded.
// Node-id tie breaking and strict relaxation preserve deterministic seams.
#include "multiband_blending/graphcut/dijkstra.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <vector>

namespace multiband_blending::graphcut {
namespace {

struct QueueEntry {
    double distance;
    DualNodeId node;
};

struct GreaterQueueEntry {
    /// Give smaller distances priority, breaking ties by the smaller node identifier.
    bool operator()(const QueueEntry& first, const QueueEntry& second) const noexcept {
        if (first.distance != second.distance) {
            return first.distance > second.distance;
        }
        return first.node > second.node;
    }
};

struct Predecessor {
    DualNodeId node;
    PrimalEdge primal_edge;
};

/// Follow predecessors backward and return crossed primal edges from source to target.
std::vector<PrimalEdge> reconstruct_cut_edges(
    const DualNodeId source, const DualNodeId target,
    const std::vector<std::optional<Predecessor>>& predecessors) {
    std::vector<PrimalEdge> crossed_primal_edges;

    DualNodeId current = target;
    while (current != source) {
        const Predecessor& predecessor = predecessors[current].value();
        crossed_primal_edges.push_back(predecessor.primal_edge);
        current = predecessor.node;
    }

    std::reverse(crossed_primal_edges.begin(), crossed_primal_edges.end());
    return crossed_primal_edges;
}

} // namespace

/// Find a minimum-cost dual path and return the primal edges it crosses.
std::vector<PrimalEdge> dijkstra_shortest_path(const DualGridGraph& graph,
                                               const DualNodeId source,
                                               const DualNodeId target) {
    if (source == target) {
        return {};
    }

    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> distances(graph.node_count(), infinity);
    std::vector<std::optional<Predecessor>> predecessors(graph.node_count());
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, GreaterQueueEntry> queue;

    distances[source] = 0.0;
    queue.push({0.0, source});

    while (!queue.empty()) {
        const QueueEntry current = queue.top();
        queue.pop();

        if (current.distance > distances[current.node]) {
            continue;
        }
        if (current.node == target) {
            return reconstruct_cut_edges(source, target, predecessors);
        }

        for (const DualNeighbor& neighbor : graph.neighbors(current.node)) {
            const double candidate = current.distance + neighbor.cost;
            if (candidate >= distances[neighbor.node]) {
                continue;
            }

            distances[neighbor.node] = candidate;
            predecessors[neighbor.node] = {current.node,
                                           neighbor.primal_edge};
            queue.push({candidate, neighbor.node});
        }
    }

    throw std::logic_error("dual target is unreachable from the source");
}

} // namespace multiband_blending::graphcut
