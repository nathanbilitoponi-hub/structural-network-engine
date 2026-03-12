import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# -----------------------------
# LOAD DATA
# -----------------------------
def load_points(csv_path):

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError("CSV must contain at least 3 columns")

    points = data[:, :3].astype(float)

    return points, data.shape[1]


# -----------------------------
# MULTISCALE ANISOTROPY
# -----------------------------
def anisotropy_scores(points, k):

    tree = cKDTree(points)
    d, idx = tree.query(points, k=min(k + 1, len(points)))

    scores = np.zeros(len(points))

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

    all_scores = []

    for k in ks:
        s = anisotropy_scores(points, k)
        all_scores.append(s)

    all_scores = np.array(all_scores)

    return np.mean(all_scores, axis=0)


# -----------------------------
# BUILD GRAPH
# -----------------------------
def build_graph(core_points, core_scores):

    tree = cKDTree(core_points)

    d, idx = tree.query(core_points, k=min(10, len(core_points)))

    nn_med = np.median(d[:, 1])

    edge_cut = 1.8 * nn_med

    edges = []

    degree = np.zeros(len(core_points))

    rows = []
    cols = []
    vals = []

    for a in range(len(core_points)):

        for bpos in range(1, idx.shape[1]):

            b = int(idx[a, bpos])
            dist = float(d[a, bpos])

            if a < b and dist <= edge_cut:

                conf = (
                    core_scores[a] *
                    core_scores[b] *
                    np.exp(-dist / edge_cut)
                )

                if conf < 0.15:
                    continue

                edges.append((a, b, dist, conf))

                degree[a] += 1
                degree[b] += 1

                rows += [a, b]
                cols += [b, a]
                vals += [1, 1]

    if len(rows) == 0:

        n_components = len(core_points)

    else:

        A = csr_matrix((vals, (rows, cols)), shape=(len(core_points), len(core_points)))

        n_components, labels = connected_components(A, directed=False)

    return edges, degree, int(n_components)


# -----------------------------
# MAIN ALGORITHM
# -----------------------------
def run_algorithm(csv_path):

    points, n_columns = load_points(csv_path)

    scores = multiscale_scores(points)

    thr = np.quantile(scores, 0.82)

    core_mask = scores >= thr

    core_points = points[core_mask]

    core_scores = scores[core_mask]

    edges, degree, n_components = build_graph(core_points, core_scores)

    # -----------------------------
    # NODES
    # -----------------------------
    nodes = []

    for i in range(len(core_points)):

        nodes.append({
            "id": int(i),
            "x": float(core_points[i, 0]),
            "y": float(core_points[i, 1]),
            "z": float(core_points[i, 2]),
            "backbone": bool(degree[i] >= 3),
            "degree": int(degree[i]),
            "score": float(core_scores[i])
        })

    # -----------------------------
    # LINKS
    # -----------------------------
    links = []

    for a, b, dist, conf in edges:

        links.append({
            "source": int(a),
            "target": int(b),
            "distance": float(dist),
            "confidence": float(conf)
        })

    mean_degree = float(np.mean(degree)) if len(degree) > 0 else 0
    max_degree = int(np.max(degree)) if len(degree) > 0 else 0

    metrics = {

        "file": os.path.basename(csv_path),

        "rows": int(len(points)),

        "columns": int(n_columns),

        "valid_points": int(len(points)),

        "network_sample": int(len(core_points)),

        "edges": int(len(edges)),

        "mean_degree": round(mean_degree, 3),

        "max_degree": int(max_degree),

        "components": int(n_components),

        "backbone_nodes": int(np.sum(degree >= 3)),
    }

    return {

        "result": "Structural network extracted",

        "metrics": metrics,

        "nodes": nodes,

        "links": links,
    }
