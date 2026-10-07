"""Physical-radius vessel graphs; widest path is a documented max-flow surrogate."""

import heapq
from itertools import pairwise, product

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def vessel_graph(mask, spacing):
    mask = np.asarray(mask, dtype=bool)
    graph = nx.Graph()
    if not mask.any():
        return graph
    radius = ndi.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    skeleton = skeletonize(np.pad(mask, 1))[1:-1, 1:-1, 1:-1]
    nodes = {tuple(int(x) for x in p) for p in np.argwhere(skeleton)}
    for p in sorted(nodes):
        graph.add_node(p, radius_mm=float(radius[p]))
    offsets = [p for p in product((-1, 0, 1), repeat=3) if any(p)]
    for p in sorted(nodes):
        for delta in offsets:
            q = tuple(a + b for a, b in zip(p, delta))
            if q in nodes and p < q:
                graph.add_edge(
                    p,
                    q,
                    capacity=(radius[p] + radius[q]) / 2,
                    length_mm=float(np.linalg.norm(np.asarray(delta) * spacing)),
                )
    return graph


def widest_path(graph, source, target):
    """Maximum bottleneck-capacity path, not an assertion of the paper's exact solver."""
    if source not in graph or target not in graph:
        return []
    capacity = {source: float("inf")}
    previous = {}
    heap = [(-float("inf"), source)]
    while heap:
        negative, u = heapq.heappop(heap)
        if -negative < capacity.get(u, -1):
            continue
        if u == target:
            path = [u]
            while u != source:
                u = previous[u]
                path.append(u)
            return list(reversed(path))
        for v, edge in graph[u].items():
            score = min(capacity[u], edge["capacity"])
            if score > capacity.get(v, -1):
                capacity[v], previous[v] = score, u
                heapq.heappush(heap, (-score, v))
    return []


def graph_summary(mask, spacing):
    graph = vessel_graph(mask, spacing)
    if not graph:
        return {
            "nodes": 0,
            "components": None,
            "radius_max_mm": None,
            "path_length_mm": None,
            "radius_min_over_max": None,
        }
    components = list(nx.connected_components(graph))
    main = graph.subgraph(max(components, key=len))
    low, high = min(main, key=lambda p: p[2]), max(main, key=lambda p: p[2])
    path = widest_path(main, low, high)
    radii = [main.nodes[p]["radius_mm"] for p in path]
    return {
        "nodes": len(graph),
        "components": len(components),
        "radius_max_mm": max(d["radius_mm"] for _, d in graph.nodes(data=True)),
        "path_length_mm": sum(main[a][b]["length_mm"] for a, b in pairwise(path)),
        "radius_min_over_max": min(radii) / max(radii) if radii else None,
    }
