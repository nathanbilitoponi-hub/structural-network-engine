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

if "ra" not in gal.columns or "dec" not in gal.columns:
    print("Il file deve contenere le colonne 'ra' e 'dec'")
    raise SystemExit

# campionamento per rendere il grafico veloce e leggibile
MAX_POINTS = 1200
if len(gal) > MAX_POINTS:
    gal = gal.sample(MAX_POINTS, random_state=42).reset_index(drop=True)
else:
    gal = gal.reset_index(drop=True)

K = 5
MAX_DIST = 6.0

points = list(zip(gal["ra"].tolist(), gal["dec"].tolist()))
edges = set()

# costruzione grafo KNN
for i, (x1, y1) in enumerate(points):
    dists = []

    for j, (x2, y2) in enumerate(points):
        if i == j:
            continue

        d = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        dists.append((d, j))

    dists.sort(key=lambda x: x[0])

    count = 0
    for d, j in dists:
        if d > MAX_DIST:
            continue

        a, b = sorted((i, j))
        edges.add((a, b))
        count += 1

        if count >= K:
            break

# calcolo grado nodi
degree = defaultdict(int)
for a, b in edges:
    degree[a] += 1
    degree[b] += 1

# filtro: teniamo solo edges i cui estremi hanno grado >= 2
filtered_edges = []
for a, b in edges:
    if degree[a] >= 2 and degree[b] >= 2:
        filtered_edges.append((a, b))

plt.figure(figsize=(12, 8))

# disegna edges filtrati
for a, b in filtered_edges:
    x1, y1 = points[a]
    x2, y2 = points[b]
    plt.plot([x1, x2], [y1, y2], linewidth=0.5, alpha=0.7, color="red")

# disegna galassie
plt.scatter(gal["ra"], gal["dec"], s=5, color="navy", alpha=0.6)

plt.title("Filtered Structural Graph")
plt.xlabel("RA")
plt.ylabel("DEC")
plt.tight_layout()
plt.show()

print(f"Punti usati: {len(gal)}")
print(f"Edges KNN totali: {len(edges)}")
print(f"Edges filtrati: {len(filtered_edges)}")