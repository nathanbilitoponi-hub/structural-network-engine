import os
import csv
import math


def euclidean_distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


def connected_components(adjacency):
    visited = set()
    components = 0

    for node in adjacency:
        if node in visited:
            continue

        components += 1
        stack = [node]

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return components


def run_algorithm(file_path):
    print("Running Structural Network Engine on:", file_path)

    filename = os.path.basename(file_path)
    extension = os.path.splitext(filename)[1].lower()
    file_size = os.path.getsize(file_path)

    if extension != ".csv":
        return {
            "result": "Unsupported file type for structural analysis.",
            "metrics": {
                "file": filename,
                "type": extension if extension else "unknown",
                "size_bytes": file_size
            },
            "nodes": [],
            "links": []
        }

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = list(csv.reader(f))

    row_count = len(reader)

    if row_count == 0:
        return {
            "result": "Empty CSV file.",
            "metrics": {
                "file": filename,
                "rows": 0,
                "columns": 0,
                "valid_points": 0,
                "size_bytes": file_size
            },
            "nodes": [],
            "links": []
        }

    col_count = len(reader[0]) if len(reader) > 0 else 0

    points = []

    for row in reader:
        if len(row) < 3:
            continue

        try:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            points.append((x, y, z))
        except Exception:
            continue

    valid_points = len(points)

    if valid_points < 2:
        return {
            "result": "CSV loaded but not enough valid 3D points.",
            "metrics": {
                "file": filename,
                "rows": row_count,
                "columns": col_count,
                "valid_points": valid_points,
                "size_bytes": file_size
            },
            "nodes": [],
            "links": []
        }

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_x = sum(xs) / valid_points
    center_y = sum(ys) / valid_points
    center_z = sum(zs) / valid_points

    distances_from_center = []
    for x, y, z in points:
        d = math.sqrt(
            (x - center_x) ** 2 +
            (y - center_y) ** 2 +
            (z - center_z) ** 2
        )
        distances_from_center.append(d)

    mean_center_distance = sum(distances_from_center) / valid_points

    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z

    volume = dx * dy * dz if dx > 0 and dy > 0 and dz > 0 else 0.0
    density = valid_points / volume if volume > 0 else 0.0

    max_points_for_network = 220

    if valid_points > max_points_for_network:
        step = max(1, valid_points // max_points_for_network)
        sampled_points = points[::step][:max_points_for_network]
    else:
        sampled_points = points

    n = len(sampled_points)

    nearest_neighbor_distances = []

    for i in range(n):
        p_i = sampled_points[i]
        best_d = None

        for j in range(n):
            if i == j:
                continue

            p_j = sampled_points[j]
            d = euclidean_distance(p_i, p_j)

            if best_d is None or d < best_d:
                best_d = d

        if best_d is not None:
            nearest_neighbor_distances.append(best_d)

    if len(nearest_neighbor_distances) == 0:
        return {
            "result": "Analysis failed: nearest-neighbor distances could not be computed.",
            "metrics": {
                "file": filename,
                "size_bytes": file_size
            },
            "nodes": [],
            "links": []
        }

    mean_nn_distance = sum(nearest_neighbor_distances) / len(nearest_neighbor_distances)
    threshold = 2.4 * mean_nn_distance

    adjacency = {i: [] for i in range(n)}
    edge_count = 0
    links = []

    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(sampled_points[i], sampled_points[j])

            if d <= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
                edge_count += 1
                links.append({
                    "source": i,
                    "target": j,
                    "distance": round(d, 6)
                })

    degrees = [len(adjacency[i]) for i in range(n)]
    mean_degree = sum(degrees) / n if n > 0 else 0.0
    max_degree = max(degrees) if len(degrees) > 0 else 0

    component_count = connected_components(adjacency)

    backbone_nodes = [i for i in range(n) if len(adjacency[i]) >= mean_degree]
    backbone_count = len(backbone_nodes)
    backbone_fraction = backbone_count / n if n > 0 else 0.0
    backbone_set = set(backbone_nodes)

    nodes = []
    for i in range(n):
        nodes.append({
            "id": i,
            "backbone": i in backbone_set,
            "degree": len(adjacency[i])
        })

    metrics = {
        "file": filename,
        "rows": row_count,
        "columns": col_count,
        "valid_points": valid_points,
        "network_sample": n,
        "size_bytes": file_size,
        "bbox_x_min": round(min_x, 3),
        "bbox_x_max": round(max_x, 3),
        "bbox_y_min": round(min_y, 3),
        "bbox_y_max": round(max_y, 3),
        "bbox_z_min": round(min_z, 3),
        "bbox_z_max": round(max_z, 3),
        "mean_center_distance": round(mean_center_distance, 3),
        "density": round(density, 8),
        "mean_nearest_neighbor": round(mean_nn_distance, 3),
        "threshold": round(threshold, 3),
        "edges": edge_count,
        "mean_degree": round(mean_degree, 3),
        "max_degree": max_degree,
        "components": component_count,
        "backbone_nodes": backbone_count,
        "backbone_fraction": round(backbone_fraction, 3)
    }

    return {
        "result": "Structural network extraction completed successfully.",
        "metrics": metrics,
        "nodes": nodes,
        "links": links
    }