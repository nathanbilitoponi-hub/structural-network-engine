import os
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

jobs_dir = "jobs"

jobs = [j for j in os.listdir(jobs_dir) if os.path.isdir(os.path.join(jobs_dir, j))]

print("Jobs trovati:")
for i, j in enumerate(jobs):
    print(i, j)

choice = int(input("Scegli job: "))
job = jobs[choice]

gal_path = os.path.join(jobs_dir, job, "sdss_galaxies.csv")

if not os.path.exists(gal_path):
    print(f"File non trovato: {gal_path}")
    raise SystemExit

gal = pd.read_csv(gal_path)

required = ["ra", "dec", "z"]
for col in required:
    if col not in gal.columns:
        print(f"Manca la colonna: {col}")
        raise SystemExit

# campionamento per rendere il test veloce
MAX_POINTS = 1200
if len(gal) > MAX_POINTS:
    gal = gal.sample(MAX_POINTS, random_state=42).reset_index(drop=True)
else:
    gal = gal.reset_index(drop=True)

# normalizzazione
ra_vals = gal["ra"].astype(float)
dec_vals = gal["dec"].astype(float)
z_vals = gal["z"].astype(float)

ra_n = (ra_vals - ra_vals.mean()) / (ra_vals.std() + 1e-9)
dec_n = (dec_vals - dec_vals.mean()) / (dec_vals.std() + 1e-9)
z_n = (z_vals - z_vals.mean()) / (z_vals.std() + 1e-9)

points3d = list(zip(ra_n.tolist(), dec_n.tolist(), z_n.tolist()))
points2d = list(zip(ra_vals.tolist(), dec_vals.tolist()))

K = 5
MAX_DIST_3D = 0.45

candidate_edges = []

# costruzione KNN 3D
for i, (x1, y1, z1) in enumerate(points3d):
    dists = []

    for j, (x2, y2, z2) in enumerate(points3d):
        if i == j:
            continue

        d = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        dists.append((d, j))

    dists.sort(key=lambda x: x[0])

    count = 0
    for d, j in dists:
        if d > MAX_DIST_3D:
            continue

        a, b = sorted((i, j))
        candidate_edges.append((d, a, b))
        count += 1

        if count >= K:
            break

# rimuove duplicati tenendo la distanza minima
edge_map = {}
for d, a, b in candidate_edges:
    key = (a, b)
    if key not in edge_map or d < edge_map[key]:
        edge_map[key] = d

unique_edges = [(d, a, b) for (a, b), d in edge_map.items()]
unique_edges.sort(key=lambda x: x[0])

# union-find per MST
parent = list(range(len(points3d)))
rank = [0] * len(points3d)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    rx = find(x)
    ry = find(y)

    if rx == ry:
        return False

    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1

    return True

mst_edges = []
for d, a, b in unique_edges:
    if union(a, b):
        mst_edges.append((d, a, b))

# pruning: togli edges troppo lunghi rispetto alla mediana
distances = [d for d, _, _ in mst_edges]
if len(distances) == 0:
    print("Nessun edge MST trovato.")
    raise SystemExit

distances_sorted = sorted(distances)
median_d = distances_sorted[len(distances_sorted) // 2]
MAX_MST_EDGE = median_d * 1.8

pruned_edges = [(d, a, b) for d, a, b in mst_edges if d <= MAX_MST_EDGE]

plt.figure(figsize=(12, 8))

# galassie
plt.scatter(ra_vals, dec_vals, s=5, color="lightsteelblue", alpha=0.45)

# mst pruned
for d, a, b in pruned_edges:
    x1, y1 = points2d[a]
    x2, y2 = points2d[b]
    plt.plot([x1, x2], [y1, y2], linewidth=0.9, alpha=0.9, color="red")

plt.title("3D MST Structural Skeleton projected on RA-DEC")
plt.xlabel("RA")
plt.ylabel("DEC")
plt.tight_layout()
plt.show()

print(f"Punti usati: {len(gal)}")
print(f"Candidate edges: {len(unique_edges)}")
print(f"MST edges: {len(mst_edges)}")
print(f"Pruned MST edges: {len(pruned_edges)}")
print(f"MST median distance: {median_d:.4f}")
print(f"Pruning threshold: {MAX_MST_EDGE:.4f}")
