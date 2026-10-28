"""Core TSP: distance, construction heuristics, local search, Simulated Annealing, Tabu Search, and VNS."""

import time
import numpy as np


# ---------- Distance ----------

def dist_matrix(P, metric="Euclidean"):
    d = P[:, None, :] - P[None, :, :]
    if metric == "Euclidean":
        return np.sqrt((d ** 2).sum(2))
    if metric == "Manhattan":
        return np.abs(d).sum(2)
    raise ValueError(f"Distance metric tidak dikenal: {metric}")


def length(r, D):
    r = np.asarray(r, dtype=int)
    return float(D[r, np.roll(r, -1)].sum())


# ---------- Construction heuristics ----------

def nearest_neighbor(D, s):
    r = [s]
    left = set(range(len(D))) - {s}

    while left:
        c = min(left, key=lambda j: D[r[-1], j])
        r.append(c)
        left.remove(c)

    return r


def nearest_neighbor_variation(D, s):
    """Nearest Neighbor variation that extends either end of an open path.
    Start with node ``s``. At each step, connect the closest unvisited node
    to either the left or right endpoint of the current path. Ties are
    resolved deterministically by node index, then preferring the left end.
    The returned route contains each node once; ``length`` closes the tour.
    """
    D = np.asarray(D)
    n = len(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D harus berupa matriks jarak persegi.")
    if not 0 <= int(s) < n:
        raise ValueError(f"Starting node di luar rentang: {s}.")

    route = [int(s)]
    unvisited = set(range(n)) - {int(s)}

    while unvisited:
        candidates = []
        left_endpoint = route[0]
        right_endpoint = route[-1]
        for city in sorted(unvisited):
            candidates.append((float(D[city, left_endpoint]), city, 0))
            candidates.append((float(D[city, right_endpoint]), city, 1))

        _, city, endpoint = min(candidates)
        if endpoint == 0:
            route.insert(0, city)
        else:
            route.append(city)
        unvisited.remove(city)

    seed_position = route.index(int(s))
    return route[seed_position:] + route[:seed_position]

def insertion(D, s, kind, rng):
    """NI: nearest, CI: cheapest, FI: farthest, AI: arbitrary insertion."""
    left = set(range(len(D))) - {s}

    first = (
        int(rng.choice(sorted(left))) if kind == "AI"
        else max(left, key=lambda j: D[s, j]) if kind == "FI"
        else min(left, key=lambda j: D[s, j])
    )

    r = [s, first]
    left.remove(first)

    def cost(c, p):
        return (
            D[r[p], c]
            + D[c, r[(p + 1) % len(r)]]
            - D[r[p], r[(p + 1) % len(r)]]
        )

    while left:
        if kind == "CI":
            c, p = min(
                ((c, p) for c in left for p in range(len(r))),
                key=lambda x: cost(*x),
