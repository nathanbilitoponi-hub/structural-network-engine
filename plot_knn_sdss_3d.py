import os
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from collections import defaultdict

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

# campionamento per rendere il test veloce e leggibile
MAX_POINTS = 1000
if len(gal) > MAX_POINTS:
    gal = gal.sample(MAX_POINTS, random_state=42).reset_index(drop=True)
else:
    gal = gal.reset_index(drop=True)

# normalizzazione semplice per evitare che RA domini troppo su z
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

edges = set()

# KNN in 3D
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
        edges.add((a, b))
        count += 1

        if count >= K:
            break

# grado dei nodi
degree = defaultdict(int)
for a, b in edges:
    degree[a] += 1
    degree[b] += 1

# filtro di grado
filtered_edges = []
for a, b in edges:
    if degree[a] >= 2 and degree[b] >= 2:
        filtered_edges.append((a, b))

plt.figure(figsize=(12, 8))

# disegno in proiezione RA-DEC, ma edges costruiti in 3D
for a, b in filtered_edges:
    x1, y1 = points2d[a]
    x2, y2 = points2d[b]
    plt.plot([x1, x2], [y1, y2], linewidth=0.5, alpha=0.7, color="red")

plt.scatter(ra_vals, dec_vals, s=5, color="navy", alpha=0.55)

plt.title("3D KNN Structural Graph projected on RA-DEC")
plt.xlabel("RA")
plt.ylabel("DEC")
plt.tight_layout()
plt.show()

print(f"Punti usati: {len(gal)}")
print(f"Edges 3D KNN totali: {len(edges)}")
print(f"Edges filtrati: {len(filtered_edges)}")