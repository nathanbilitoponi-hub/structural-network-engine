import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


import numpy as np
import pandas as pd


def load_point_cloud(file_path):

    df = pd.read_csv(file_path)

    # tieni solo colonne numeriche
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 2 numeric columns")

    points = numeric_df.values

    return points


def anisotropy_scores(points, k):
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k + 1, len(points)))

    scores = np.zeros(len(points), dtype=float)

    for j in range(len(points)):
        nb = idx[j, 1:]
        if len(nb) < 3:
            continue

        Q = points[nb] - points[j]
        C = np.cov(Q.T)

        w = np.linalg.eigvalsh(C)
        w = np.sort(np.maximum(w, 1e-12))[::-1]

        scores[j] = (w[0] - w[1]) / w[0]

    return scores


def multiscale_scores(points):
    ks = [8, 12, 16]
    all_scores = [anisotropy_scores(points, k) for k in ks]
    return np.mean(np.vstack(all_scores), axis=0)


def estimate_local_scale(points, k=8):
    tree = cKDTree(points)
    d, _ = tree.query(points, k=min(k + 1, len(points)))
    if d.shape[1] < 2:
        return 1.0
    return float(np.median(d[:, 1]))


def build_initial_core_graph(core_points, core_scores):
    n = len(core_points)
    if n < 2:
        return [], np.zeros(n, dtype=int), n

    tree = cKDTree(core_points)
    d, idx = tree.query(core_points, k=min(12, n))

    nn_med = np.median(d[:, 1]) if d.shape[1] > 1 else 1.0
    edge_cut = 1.9 * nn_med

    edges = []
    degree = np.zeros(n, dtype=int)

    rows, cols, vals = [], [], []

    for a in range(n):
        for bpos in range(1, idx.shape[1]):
            b = int(idx[a, bpos])
            dist = float(d[a, bpos])

            if a < b and dist <= edge_cut:
                conf = float(
                    core_scores[a] *
                    core_scores[b] *
                    np.exp(-dist / max(edge_cut, 1e-12))
                )

                if conf < 0.12:
                    continue

                edges.append((a, b, dist, conf))
                degree[a] += 1
                degree[b] += 1

                rows += [a, b]
                cols += [b, a]
                vals += [1.0, 1.0]

    if len(rows) == 0:
        return edges, degree, n

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    n_components, _ = connected_components(A, directed=False)

    return edges, degree, int(n_components)


def reconnect_components(core_points, core_scores, edges):
    n = len(core_points)
    if n == 0:
        return edges, 0, 0

    rows, cols, vals = [], [], []
    for a, b, _, _ in edges:
        rows += [a, b]
        cols += [b, a]
        vals += [1.0, 1.0]

    if len(rows) == 0:
        labels = np.arange(n)
        n_components = n
    else:
        A = csr_matrix((vals, (rows, cols)), shape=(n, n))
        n_components, labels = connected_components(A, directed=False)

    if n_components <= 1:
        degree = np.zeros(n, dtype=int)
        for a, b, _, _ in edges:
            degree[a] += 1
            degree[b] += 1
        return edges, 0, 1

    scale = estimate_local_scale(core_points, k=8)
    reconnect_edges = []

    while True:
        rows, cols, vals = [], [], []
        all_edges = edges + reconnect_edges

        for a, b, _, _ in all_edges:
            rows += [a, b]
            cols += [b, a]
            vals += [1.0, 1.0]

        if len(rows) == 0:
            labels = np.arange(n)
            n_components = n
        else:
            A = csr_matrix((vals, (rows, cols)), shape=(n, n))
            n_components, labels = connected_components(A, directed=False)

        if n_components <= 1:
            break

        best = None
        comps = np.unique(labels)

        for i in range(len(comps)):
            ci = comps[i]
            pts_i = np.where(labels == ci)[0]

            for j in range(i + 1, len(comps)):
                cj = comps[j]
                pts_j = np.where(labels == cj)[0]

                Pi = core_points[pts_i]
                Pj = core_points[pts_j]

                tree_j = cKDTree(Pj)
                d, local_idx = tree_j.query(Pi, k=1)

                pos = int(np.argmin(d))
                dist = float(d[pos])

                if dist > 2.8 * scale:
                    continue

                a = int(pts_i[pos])
                b = int(pts_j[int(local_idx[pos])])

                conf = float(
                    0.5 * (core_scores[a] + core_scores[b]) * np.exp(-dist / max(2.8 * scale, 1e-12))
                )

                if best is None or dist < best[2]:
                    best = (a, b, dist, conf)

        if best is None:
            break

        reconnect_edges.append(best)

    final_edges = edges + reconnect_edges

    rows, cols, vals = [], [], []
    degree = np.zeros(n, dtype=int)

    for a, b, _, _ in final_edges:
        degree[a] += 1
        degree[b] += 1
        rows += [a, b]
        cols += [b, a]
        vals += [1.0, 1.0]

    if len(rows) == 0:
        final_components = n
    else:
        A = csr_matrix((vals, (rows, cols)), shape=(n, n))
        final_components, _ = connected_components(A, directed=False)

    return final_edges, len(reconnect_edges), int(final_components)


def build_sparse_adjacency(n, edges, weighted=False):
    rows, cols, vals = [], [], []
    for a, b, dist, conf in edges:
        w = float(dist) if weighted else 1.0
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    if len(rows) == 0:
        return csr_matrix((n, n))
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def extract_topology(core_points, degree, edges):
    n = len(core_points)
    if n == 0:
        return [], [], 0, 0

    topo_mask = (degree == 1) | (degree >= 3)
    topo_idx = np.where(topo_mask)[0]

    if len(topo_idx) == 0:
        topo_idx = np.array([int(np.argmax(degree))]) if n > 0 else np.array([], dtype=int)

    topo_set = set(int(x) for x in topo_idx.tolist())

    adj = [[] for _ in range(n)]
    for a, b, dist, conf in edges:
        adj[a].append((b, dist, conf))
        adj[b].append((a, dist, conf))

    topo_nodes = []
    topo_id_map = {}

    for tpos, old_idx in enumerate(topo_idx):
        topo_id_map[int(old_idx)] = int(tpos)
        topo_nodes.append({
            "id": int(tpos),
            "x": float(core_points[old_idx, 0]),
            "y": float(core_points[old_idx, 1]),
            "z": float(core_points[old_idx, 2]),
            "original_id": int(old_idx),
            "kind": "junction" if degree[old_idx] >= 3 else "terminal"
        })

    topo_edges = []
    used_pairs = set()

    for src in topo_idx:
        for nb, dist0, conf0 in adj[int(src)]:
            prev = int(src)
            cur = int(nb)
            path_dist = float(dist0)
            path_conf = [float(conf0)]

            while cur not in topo_set:
                nxts = [x for x in adj[cur] if x[0] != prev]
                if len(nxts) == 0:
                    break
                nxt, d2, c2 = nxts[0]
                prev, cur = cur, int(nxt)
                path_dist += float(d2)
                path_conf.append(float(c2))

            if cur == src or cur not in topo_set:
                continue

            a = topo_id_map[int(src)]
            b = topo_id_map[int(cur)]
            key = (min(a, b), max(a, b))
            if key in used_pairs:
                continue
            used_pairs.add(key)

            topo_edges.append({
                "source": int(a),
                "target": int(b),
                "distance": float(path_dist),
                "confidence": float(np.mean(path_conf)) if path_conf else 0.0
            })

    return topo_nodes, topo_edges, int(len(topo_nodes)), int(len(topo_edges))


def extract_trunk(core_points, edges):
    n = len(core_points)
    if n == 0 or len(edges) == 0:
        return [], 0.0, 0.0

    A = build_sparse_adjacency(n, edges, weighted=True)
    n_components, labels = connected_components(A, directed=False)

    largest_comp = np.argmax(np.bincount(labels))
    keep = np.where(labels == largest_comp)[0]

    if len(keep) < 2:
        return [], 0.0, 0.0

    remap = {old: new for new, old in enumerate(keep.tolist())}
    rev = {new: old for old, new in remap.items()}

    rows, cols, vals = [], [], []
    for a, b, dist, _ in edges:
        if a in remap and b in remap:
            aa = remap[a]
            bb = remap[b]
            rows += [aa, bb]
            cols += [bb, aa]
            vals += [float(dist), float(dist)]

    A2 = csr_matrix((vals, (rows, cols)), shape=(len(keep), len(keep)))
    D, pred = shortest_path(A2, directed=False, unweighted=False, return_predecessors=True)

    finite = np.isfinite(D)
    np.fill_diagonal(finite, False)

    if not np.any(finite):
        return [], 0.0, 0.0

    s, t = np.unravel_index(np.argmax(np.where(finite, D, -1)), D.shape)

    path = [t]
    cur = t
    while cur != s and cur != -9999:
        cur = pred[s, cur]
        if cur == -9999:
            break
        path.append(cur)
    path = path[::-1]

    trunk_nodes = [int(rev[p]) for p in path]

    if len(trunk_nodes) < 2:
        return trunk_nodes, 0.0, 0.0

    trunk_points = core_points[trunk_nodes]
    seg = np.linalg.norm(np.diff(trunk_points, axis=0), axis=1)
    trunk_length = float(np.sum(seg))
    endpoint_dist = float(np.linalg.norm(trunk_points[-1] - trunk_points[0]))
    trunk_straightness = float(endpoint_dist / trunk_length) if trunk_length > 0 else 0.0

    return trunk_nodes, trunk_length, trunk_straightness


def run_algorithm(csv_path):
    points, n_columns = load_points(csv_path)

    scores = multiscale_scores(points)

    thr = float(np.quantile(scores, 0.82))
    core_mask = scores >= thr

    core_points = points[core_mask]
    core_scores = scores[core_mask]

    edges0, degree0, components_before = build_initial_core_graph(core_points, core_scores)
    edges, reconnect_edges, components_after = reconnect_components(core_points, core_scores, edges0)

    degree = np.zeros(len(core_points), dtype=int)
    edge_conf = []
    for a, b, _, conf in edges:
        degree[a] += 1
        degree[b] += 1
        edge_conf.append(conf)

    topo_nodes, topo_edges, topo_n, topo_e = extract_topology(core_points, degree, edges)
    trunk_nodes, trunk_length, trunk_straightness = extract_trunk(core_points, edges)

    nodes = []
    trunk_set = set(trunk_nodes)

    for i in range(len(core_points)):
        nodes.append({
            "id": int(i),
            "x": float(core_points[i, 0]),
            "y": float(core_points[i, 1]),
            "z": float(core_points[i, 2]),
            "backbone": bool(degree[i] >= 2),
            "degree": int(degree[i]),
            "score": float(core_scores[i]),
            "is_trunk": bool(i in trunk_set),
        })

    links = []
    for a, b, dist, conf in edges:
        links.append({
            "source": int(a),
            "target": int(b),
            "distance": float(dist),
            "confidence": float(conf),
        })

    mean_degree = float(np.mean(degree)) if len(degree) > 0 else 0.0
    max_degree = int(np.max(degree)) if len(degree) > 0 else 0
    mean_edge_conf = float(np.mean(edge_conf)) if len(edge_conf) > 0 else 0.0
    base_scale = estimate_local_scale(points, k=8)

    metrics = {
        "file": os.path.basename(csv_path),
        "rows": int(len(points)),
        "columns": int(n_columns),
        "valid_points": int(len(points)),
        "network_sample": int(len(core_points)),
        "edges": int(len(edges)),
        "mean_degree": round(mean_degree, 4),
        "max_degree": int(max_degree),
        "components": int(components_after),
        "components_before_reconnect": int(components_before),
        "components_after_reconnect": int(components_after),
        "backbone_nodes": int(np.sum(degree >= 2)),
        "topological_nodes": int(topo_n),
        "topological_edges": int(topo_e),
        "trunk_nodes": int(len(trunk_nodes)),
        "trunk_length": round(float(trunk_length), 6),
        "trunk_straightness": round(float(trunk_straightness), 6),
        "reconnect_edges": int(reconnect_edges),
        "mean_edge_confidence": round(mean_edge_conf, 6),
        "base_scale": round(float(base_scale), 6),
    }

    return {
        "result": "Structural network extracted",
        "metrics": metrics,
        "nodes": nodes,
        "links": links,
        "topology_nodes": topo_nodes,
        "topology_edges": topo_edges,
        "trunk_node_ids": [int(x) for x in trunk_nodes],
    }
