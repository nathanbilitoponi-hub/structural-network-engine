import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def load_points(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError("CSV must contain at least 3 numeric columns")

    points = data[:, :3].astype(float)
    return points, data.shape[1]


def compute_anisotropy_scores(points, k=10):
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


def build_core_graph(core_points):
    if len(core_points) < 2:
        return [], np.zeros(len(core_points), dtype=int), 0

    tree = cKDTree(core_points)
    d, idx = tree.query(core_points, k=min(8, len(core_points)))

    if d.shape[1] < 2:
        return [], np.zeros(len(core_points), dtype=int), len(core_points)

    nn_med = float(np.median(d[:, 1]))
    edge_cut = 1.6 * nn_med

    edges = []
    degree = np.zeros(len(core_points), dtype=int)

    rows, cols, vals = [], [], []

    for a in range(len(core_points)):
        for bpos in range(1, idx.shape[1]):
            b = int(idx[a, bpos])
            dist = float(d[a, bpos])

            if a < b and dist <= edge_cut:
                edges.append((a, b, dist))
                degree[a] += 1
                degree[b] += 1

                rows += [a, b]
                cols += [b, a]
                vals += [1, 1]

    if len(core_points) == 0:
        n_components = 0
    elif len(rows) == 0:
        n_components = len(core_points)
    else:
        A = csr_matrix((vals, (rows, cols)), shape=(len(core_points), len(core_points)))
        n_components, _ = connected_components(A, directed=False)

    return edges, degree, int(n_components)


def run_algorithm(csv_path):
    points, n_columns = load_points(csv_path)

    scores = compute_anisotropy_scores(points, k=10)
    thr = float(np.quantile(scores, 0.85))
    core_mask = scores >= thr
    core_points = points[core_mask]

    edges, degree, n_components = build_core_graph(core_points)

    nodes = []
    for i in range(len(core_points)):
        nodes.append({
            "id": int(i),
            "x": float(core_points[i, 0]),
            "y": float(core_points[i, 1]),
            "z": float(core_points[i, 2]),
            "backbone": True,
            "degree": int(degree[i]),
        })

    links = []
    for a, b, dist in edges:
        links.append({
            "source": int(a),
            "target": int(b),
            "distance": float(dist),
        })

    mean_degree = float(np.mean(degree)) if len(degree) > 0 else 0.0
    max_degree = int(np.max(degree)) if len(degree) > 0 else 0

    metrics = {
        "file": os.path.basename(csv_path),
        "rows": int(len(points)),
        "columns": int(n_columns),
        "valid_points": int(len(points)),
        "network_sample": int(len(core_points)),
        "edges": int(len(edges)),
        "mean_degree": round(mean_degree, 4),
        "max_degree": int(max_degree),
        "components": int(n_components),
        "backbone_nodes": int(len(core_points)),
    }

    return {
        "result": "Structural network extracted",
        "metrics": metrics,
        "nodes": nodes,
        "links": links,
    }
