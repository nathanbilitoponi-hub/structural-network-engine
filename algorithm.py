import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def load_points(csv_path):
    df = pd.read_csv(csv_path)

    n_input_columns = int(df.shape[1])
    n_input_rows = int(df.shape[0])

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 2 numeric columns")

    if numeric_df.shape[1] == 2:
        numeric_df["z"] = 0.0

    points = numeric_df.iloc[:, :3].to_numpy(dtype=float)

    mask = np.isfinite(points).all(axis=1)
    points = points[mask]

    if len(points) < 10:
        raise ValueError("Not enough valid numeric points")

    return points, n_input_rows, n_input_columns


def estimate_local_scale(points, k=8):
    tree = cKDTree(points)
    d, _ = tree.query(points, k=min(k + 1, len(points)))
    if d.shape[1] < 2:
        return 1.0
    return float(np.median(d[:, 1]))


def anisotropy_scores(points, k=10):
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k + 1, len(points)))

    scores = np.zeros(len(points), dtype=float)

    for i in range(len(points)):
        nb = idx[i, 1:]
        if len(nb) < 3:
            continue

        Q = points[nb] - points[i]
        C = np.cov(Q.T)

        w = np.linalg.eigvalsh(C)
        w = np.sort(np.maximum(w, 1e-12))[::-1]

        if w[0] > 0:
            scores[i] = (w[0] - w[1]) / w[0]

    return scores


def select_core(points, scores):
    n = len(points)

    if n >= 100000:
        q = 0.92
    elif n >= 50000:
        q = 0.90
    elif n >= 10000:
        q = 0.88
    else:
        q = 0.85

    thr = float(np.quantile(scores, q))
    mask = scores >= thr

    core_points = points[mask]
    core_scores = scores[mask]

    return core_points, core_scores, thr


def build_graph(core_points, core_scores):
    n = len(core_points)

    if n < 2:
        return [], np.zeros(n, dtype=int), n, 0.0

    tree = cKDTree(core_points)
    d, idx = tree.query(core_points, k=min(10, n))

    if d.shape[1] < 2:
        return [], np.zeros(n, dtype=int), n, 1.0

    base_scale = float(np.median(d[:, 1]))
    edge_cut = 1.9 * base_scale

    edges = []
    degree = np.zeros(n, dtype=int)

    rows, cols, vals = [], [], []

    for a in range(n):
        for bpos in range(1, idx.shape[1]):
            b = int(idx[a, bpos])
            dist = float(d[a, bpos])

            if a < b and dist <= edge_cut:
                conf = float(
                    core_scores[a]
                    * core_scores[b]
                    * np.exp(-dist / max(edge_cut, 1e-12))
                )

                if conf < 0.10:
                    continue

                edges.append((a, b, dist, conf))
                degree[a] += 1
                degree[b] += 1

                rows += [a, b]
                cols += [b, a]
                vals += [1.0, 1.0]

    if len(rows) == 0:
        return edges, degree, n, base_scale

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    n_components, _ = connected_components(A, directed=False)

    return edges, degree, int(n_components), base_scale


def reconnect_components_fast(core_points, core_scores, edges, base_scale):
    n = len(core_points)

    if n < 2:
        return edges, 0, n

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
        return edges, 0, 1

    component_ids = np.unique(labels)
    component_points = []
    component_centers = []

    for cid in component_ids:
        pts = np.where(labels == cid)[0]
        component_points.append(pts)
        component_centers.append(np.mean(core_points[pts], axis=0))

    component_centers = np.asarray(component_centers, dtype=float)

    if len(component_centers) < 2:
        return edges, 0, int(n_components)

    center_tree = cKDTree(component_centers)
    _, nn_idx = center_tree.query(component_centers, k=min(3, len(component_centers)))

    reconnect_edges = []
    used_pairs = set()
    max_bridge = 3.0 * max(base_scale, 1e-12)

    for i in range(len(component_ids)):
        neighbors = nn_idx[i]
        neighbors = np.atleast_1d(neighbors)

        for jpos in range(1, len(neighbors)):
            j = int(neighbors[jpos])
            if i == j:
                continue

            key = (min(i, j), max(i, j))
            if key in used_pairs:
                continue
            used_pairs.add(key)

            pts_i = component_points[i]
            pts_j = component_points[j]

            Pi = core_points[pts_i]
            Pj = core_points[pts_j]

            tree_j = cKDTree(Pj)
            dmin, local_idx = tree_j.query(Pi, k=1)

            pos = int(np.argmin(dmin))
            dist = float(dmin[pos])

            if dist > max_bridge:
                continue

            a = int(pts_i[pos])
            b = int(pts_j[int(local_idx[pos])])

            conf = float(
                0.5 * (core_scores[a] + core_scores[b]) * np.exp(-dist / max(max_bridge, 1e-12))
            )

            reconnect_edges.append((a, b, dist, conf))

    all_edges = edges + reconnect_edges

    rows, cols, vals = [], [], []
    for a, b, _, _ in all_edges:
        rows += [a, b]
        cols += [b, a]
        vals += [1.0, 1.0]

    if len(rows) == 0:
        final_components = n
    else:
        A = csr_matrix((vals, (rows, cols)), shape=(n, n))
        final_components, _ = connected_components(A, directed=False)

    return all_edges, len(reconnect_edges), int(final_components)


def extract_trunk_fast(core_points, edges):
    n = len(core_points)

    if n < 2 or len(edges) == 0:
        return [], 0.0, 0.0

    degree = np.zeros(n, dtype=int)
    for a, b, _, _ in edges:
        degree[a] += 1
        degree[b] += 1

    leaves = np.where(degree == 1)[0]

    if len(leaves) < 2:
        trunk_nodes = np.argsort(-degree)[: min(2, n)].tolist()
        if len(trunk_nodes) < 2:
            return trunk_nodes, 0.0, 0.0
    else:
        leaf_points = core_points[leaves]
        best_i = 0
        best_j = 1
        best_d = -1.0

        step = max(1, len(leaves) // 120)

        sampled = np.arange(0, len(leaves), step)
        sample_pts = leaf_points[sampled]

        for i in range(len(sample_pts)):
            diff = sample_pts[i + 1 :] - sample_pts[i]
            if len(diff) == 0:
                continue
            dsq = np.sum(diff * diff, axis=1)
            jrel = int(np.argmax(dsq))
            val = float(dsq[jrel])
            if val > best_d:
                best_d = val
                best_i = int(sampled[i])
                best_j = int(sampled[i + 1 + jrel])

        trunk_nodes = [int(leaves[best_i]), int(leaves[best_j])]

    P = core_points[trunk_nodes]
    trunk_length = float(np.linalg.norm(P[1] - P[0]))
    trunk_straightness = 1.0 if trunk_length > 0 else 0.0

    return trunk_nodes, trunk_length, trunk_straightness


def simplify_for_visualization(core_points, degree, target_nodes=1200):
    n = len(core_points)

    if n <= target_nodes:
        keep_idx = np.arange(n)
        remap = {int(i): int(i) for i in keep_idx}
        return keep_idx, remap

    priority = degree.astype(float)
    priority += 0.001 * np.arange(n)

    keep_idx = np.argsort(-priority)[:target_nodes]
    keep_idx = np.sort(keep_idx)

    remap = {int(old): int(new) for new, old in enumerate(keep_idx)}

    return keep_idx, remap


def run_algorithm(csv_path):
    points, input_rows, input_columns = load_points(csv_path)

    scores = anisotropy_scores(points, k=10)
    core_points, core_scores, _ = select_core(points, scores)

    edges0, degree0, components_before, base_scale = build_graph(core_points, core_scores)
    edges, reconnect_edges, components_after = reconnect_components_fast(
        core_points, core_scores, edges0, base_scale
    )

    degree = np.zeros(len(core_points), dtype=int)
    confs = []
    for a, b, _, conf in edges:
        degree[a] += 1
        degree[b] += 1
        confs.append(float(conf))

    trunk_nodes, trunk_length, trunk_straightness = extract_trunk_fast(core_points, edges)
    trunk_set = set(trunk_nodes)

    keep_idx, remap = simplify_for_visualization(core_points, degree, target_nodes=1200)
    keep_set = set(int(x) for x in keep_idx.tolist())

    vis_nodes = []
    for old_idx in keep_idx:
        vis_nodes.append({
            "id": int(remap[int(old_idx)]),
            "x": float(core_points[old_idx, 0]),
            "y": float(core_points[old_idx, 1]),
            "z": float(core_points[old_idx, 2]),
            "degree": int(degree[old_idx]),
            "score": float(core_scores[old_idx]),
            "backbone": bool(degree[old_idx] >= 2),
            "is_trunk": bool(int(old_idx) in trunk_set),
        })

    vis_links = []
    for a, b, dist, conf in edges:
        if a in keep_set and b in keep_set:
            vis_links.append({
                "source": int(remap[int(a)]),
                "target": int(remap[int(b)]),
                "distance": float(dist),
                "confidence": float(conf),
            })

    mean_degree = float(np.mean(degree)) if len(degree) > 0 else 0.0
    max_degree = int(np.max(degree)) if len(degree) > 0 else 0
    mean_edge_conf = float(np.mean(confs)) if len(confs) > 0 else 0.0
    backbone_nodes = int(np.sum(degree >= 2))

    metrics = {
        "file": os.path.basename(csv_path),
        "rows": int(input_rows),
        "columns": int(input_columns),
        "valid_points": int(len(points)),
        "network_sample": int(len(core_points)),
        "edges": int(len(edges)),
        "mean_degree": round(mean_degree, 3),
        "max_degree": int(max_degree),
        "components": int(components_after),
        "components_before_reconnect": int(components_before),
        "components_after_reconnect": int(components_after),
        "backbone_nodes": int(backbone_nodes),
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
        "nodes": vis_nodes,
        "links": vis_links,
        "trunk_node_ids": [int(x) for x in trunk_nodes if int(x) in keep_set],
    }
