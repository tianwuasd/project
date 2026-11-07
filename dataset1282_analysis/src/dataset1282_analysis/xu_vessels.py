"""Orientation-independent vessel-path refinement on a local millimetre grid.

Anonymous cache geometry cannot identify the patient's superior/left direction.
These endpoints are topological landmarks only. Never invent anatomical sides.
"""

from itertools import product
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components, dijkstra, maximum_flow
from skimage.morphology import skeletonize


def _main(labels, cls):
    parts, count = ndimage.label(labels == cls)
    if not count:
        return np.zeros(labels.shape, bool), 0
    counts = np.bincount(parts.ravel())
    counts[0] = 0
    return parts == counts.argmax(), count


def _radius_graph(mask, spacing, anchor):
    """Use maximal-inscribed-radius edge capacities; return paths carrying max flow."""
    centre = np.argwhere(skeletonize(np.ascontiguousarray(mask), method="lee"))
    n = len(centre)
    if n < 2 or n > 80000:
        return None, dict(status="unusable_skeleton", nodes=n)
    # Padding supplies exterior zeros even when a whole bounding-box face contains vessel.
    radius = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[
        tuple((centre + 1).T)
    ]
    lookup = np.full(mask.shape, -1, dtype=np.int32)
    lookup[tuple(centre.T)] = np.arange(n)
    rows, cols, capacities, lengths = [], [], [], []
    for offset in product((-1, 0, 1), repeat=3):
        if offset <= (0, 0, 0):
            continue
        neighbors = centre + offset
        inside = np.all((neighbors >= 0) & (neighbors < mask.shape), axis=1)
        left = np.flatnonzero(inside)
        right = lookup[tuple(neighbors[inside].T)]
        valid = right >= 0
        left, right = left[valid], right[valid]
        if left.size:
            rows.extend((left, right))
            cols.extend((right, left))
            cap = np.maximum(1, np.rint((radius[left] + radius[right]) * 500)).astype(
                np.int32
            )
            capacities.extend((cap, cap))
            length = float(np.linalg.norm(np.asarray(offset) * spacing))
            lengths.extend((np.full(left.size, length), np.full(left.size, length)))
    if not rows:
        return None, dict(status="isolated_skeleton", nodes=n)
    idx = (np.concatenate(rows), np.concatenate(cols))
    capacity = sparse.csr_matrix((np.concatenate(capacities), idx), shape=(n, n))
    edges = sparse.csr_matrix((np.concatenate(lengths), idx), shape=(n, n))
    ends = np.flatnonzero(np.diff(edges.indptr) == 1)
    if (
        len(ends) < 2
        or len(ends) > 128
        or connected_components(edges, directed=False)[0] != 1
    ):
        return None, dict(status="ambiguous_skeleton", nodes=n, endpoints=len(ends))
    within = anchor[tuple(centre[ends].T)]
    if not within.any():
        return None, dict(status="no_anchor_endpoint", nodes=n)
    # Reference endpoint in the predicted anchor, selected by the longest geodesic from
    # a deterministic seed. It is not interpreted as the valve or either anatomical side.
    first_dist = dijkstra(edges, directed=False, indices=int(ends[within][0]))
    root = int(ends[within][np.argmax(first_dist[ends[within]])])
    distances = dijkstra(edges, directed=False, indices=root)
    selected = sorted(
        (int(t) for t in ends if t != root), key=lambda t: (-distances[t], t)
    )[:2]
    active = np.zeros(n, bool)
    routes = []
    for terminal in selected:
        flow = maximum_flow(capacity, root, terminal)
        coords = flow.flow.tocoo()
        positive = coords.data > 0
        active[coords.row[positive]] = True
        active[coords.col[positive]] = True
        routes.append(
            dict(
                end_local=centre[terminal].tolist(),
                capacity_mm=float(flow.flow_value) / 1000,
                geodesic_mm=float(distances[terminal]),
            )
        )
    on_route = np.zeros(mask.shape, bool)
    on_route[tuple(centre[active].T)] = True
    return on_route, dict(
        status="flow_computed",
        nodes=n,
        endpoints=len(ends),
        root_local=centre[root].tolist(),
        median_radius_mm=float(np.median(radius)),
        paths=routes,
    )


def refine_vessels(labels, spacing, anatomy):
    """Transfer small disconnected cross-label fragments only with unambiguous graph support."""
    labels = np.asarray(labels)
    spacing = np.asarray(spacing, dtype=float)
    if (
        labels.ndim != 3
        or not labels.size
        or labels.dtype.kind not in "iu"
        or labels.min() < 0
        or labels.max() > 10
    ):
        raise ValueError("Expected 3D ten-structure labels")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Expected physical grid spacing in output axes")
    ao, pa = anatomy["AO"], anatomy["PA"]
    if ao == pa or not {ao, pa} <= set(range(1, 11)):
        raise ValueError("Expected distinct AO and PA identifiers")
    result = labels.copy()
    info = dict(
        mode="radius_maxflow_connectivity",
        landmark_orientation="unavailable_local_array_only",
        reassigned_voxels=0,
        vessels={},
    )
    ao_main, ao_parts = _main(labels, ao)
    pa_main, pa_parts = _main(labels, pa)
    if not ao_parts or not pa_parts or max(ao_parts, pa_parts) > 5000:
        info["status"] = "abstained_missing_or_fragmented_anchors"
        return result, info
    united, _ = ndimage.label((labels == ao) | (labels == pa))
    # Each main is connected by six-neighborhood and therefore lies in one union component.
    a = int(united[tuple(np.unravel_index(ao_main.argmax(), ao_main.shape))])
    p = int(united[tuple(np.unravel_index(pa_main.argmax(), pa_main.shape))])
    if a == p:
        info["status"] = "abstained_anchors_connected"
        return result, info
    bounds = ndimage.find_objects(united)
    for cls, other, cc, kind in ((ao, pa, a, "AO"), (pa, ao, p, "PA")):
        cut = bounds[cc - 1]
        mask = united[cut] == cc
        region = labels[cut]
        primary = (ao_main if cls == ao else pa_main)[cut]
        path, detail = _radius_graph(mask, spacing, primary)
        info["vessels"][kind] = detail
        detail["local_origin"] = [s.start for s in cut]
        if path is None:
            continue
        pieces, count = ndimage.label((region == other) & mask)
        size = int(primary.sum())
        reassigned = 0
        for i, slices in enumerate(ndimage.find_objects(pieces), 1):
            if slices is None:
                continue
            # Inspect only the fragment and a one-voxel context, never the full volume per island.
            region_slice = tuple(
                slice(max(0, s.start - 1), min(n, s.stop + 1))
                for s, n in zip(slices, mask.shape)
            )
            fragment = pieces[region_slice] == i
            nvox = int(fragment.sum())
            touches = ndimage.binary_dilation(fragment) & primary[region_slice]
            if (
                nvox <= 0.05 * size
                and np.count_nonzero(touches) >= 2
                and np.any(fragment & path[region_slice])
            ):
                result[cut][region_slice][fragment] = cls
                reassigned += nvox
        detail.update(candidate_islands=count, reassigned_voxels=reassigned)
        info["reassigned_voxels"] += reassigned
    info["status"] = "applied" if info["reassigned_voxels"] else "preserved"
    return result, info
