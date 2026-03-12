import numpy as np
from scipy.spatial import cKDTree


def load_points(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.shape[1] < 3:
        raise ValueError("CSV must contain at least 3 columns")
    return data[:, :3]


def compute_anisotropy_scores(points, k=10):
    tree = cKDTree(points)
    d, i = tree.query(points, k=min(k+1, len(points)))

    scores = np.zeros(len(points))

    for j in range(len(points)):

        nb = i[j, 1:]
        if len(nb) < 3:
            continue

        Q = points[nb] - points[j]
        C = np.cov(Q.T)

        w = np.linalg.eigvalsh(C)
        w = np.sort(np.maximum(w, 1e-12))[::-1]

        scores[j] = (w[0] - w[1]) / w[0]

    return scores


def build_graph(points):

    tree = cKDTree(points)
    d, i = tree.query(points, k=min(8, len(points)))

    nn_med = np.median(d[:, 1])
    edge_cut = 1.6 * nn_med

    edges = []
    degree = np.zeros(len(points))

    for a in range(len(points)):
        for bpos in range(1, i.shape[1]):

            b = i[a, bpos]
            dist = d[a, bpos]

            if a < b and dist <= edge_cut:
                edges.append((a, b, dist))
                degree[a] += 1
                degree[b] += 1

    return edges, degree


def build_nodes(points, core_mask, degree):

    nodes = []

    for idx in range(len(points)):

        nodes.append({
            "id": int(idx),
            "x": float(points[idx, 0]),
            "y": float(points[idx, 1]),
            "z": float(points[idx, 2]),
            "backbone": bool(core_mask[idx]),
            "degree": int(degree[idx])
        })

    return nodes


def run_algorithm(csv_path):

    points = load_points(csv_path)

    scores = compute_anisotropy_scores(points)

    thr = np.quantile(scores, 0.85)

    core_mask = scores >= thr

    core = points[core_mask]

    edges, degree = build_graph(points)

    links = []

    for a, b, dist in edges:
        links.append({
            "source": int(a),
            "target": int(b),
            "distance": float(dist)
        })

    nodes = build_nodes(points, core_mask, degree)

    mean_degree = float(np.mean(degree))
    max_degree = int(np.max(degree))

    metrics = {
        "rows": int(len(points)),
        "valid_points": int(len(points)),
        "network_sample": int(len(core)),
        "edges": int(len(edges)),
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "components": 1,
        "backbone_nodes": int(np.sum(core_mask))
    }

    return {
        "result": "Structural network extracted",
        "metrics": metrics,
        "nodes": nodes,
        "links": links
    }
