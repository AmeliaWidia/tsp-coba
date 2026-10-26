
"""Core TSP: distance, construction heuristics, local search, SA, TS, ILS, VNS."""

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
        # Tuple order makes ties deterministic: distance, city index,
        # then endpoint (left=0 before right=1).
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

    # The result is a closed TSP tour, so cyclic rotation preserves its edges.
    # Rotate the representation back to the selected seed for consistent UI.
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
            )
        else:
            if kind == "NI":
                c = min(left, key=lambda j: min(D[j, k] for k in r))
            elif kind == "FI":
                c = max(left, key=lambda j: min(D[j, k] for k in r))
            else:
                c = int(rng.choice(sorted(left)))

            p = min(range(len(r)), key=lambda p: cost(c, p))

        r.insert(p + 1, c)
        left.remove(c)

    return r


# ---------- Neighborhood moves ----------

def apply(r, m):
    op, i, j, k = m

    if op == "Swap":
        r = r[:]
        r[i], r[j] = r[j], r[i]
        return r

    if op == "2-opt":
        return r[:i] + r[i:j + 1][::-1] + r[j + 1:]

    # 3-opt: segment exchange
    return r[:i] + r[j:k] + r[i:j] + r[k:]


def rand_move(n, op, rng):
    if op == "3-opt":
        i, j, k = sorted(rng.choice(n + 1, 3, replace=False))
        return op, int(i), int(j), int(k)

    i, j = sorted(rng.choice(n, 2, replace=False))
    return op, int(i), int(j), 0


def neighborhood(n, op, size, rng):
    if op != "3-opt" and n * (n - 1) // 2 <= size:
        return [
            (op, i, j, 0)
            for i in range(n)
            for j in range(i + 1, n)
        ]

    return [rand_move(n, op, rng) for _ in range(size)]


def move_key(r, m):
    op, i, j, k = m
    cities = [r[i], r[j]]
    if op == "3-opt":
        cities.append(r[k % len(r)])
    return op, tuple(sorted(cities))


# ---------- Local search ----------

def two_opt_ls(r, D, strategy="Best Improvement", max_pass=200):
    r = r[:]
    n = len(r)

    for _ in range(max_pass):
        best = (-1e-9, None)

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = r[i - 1], r[i]
                c, d = r[j], r[(j + 1) % n]

                delta = (
                    D[a, c] + D[b, d]
                    - D[a, b] - D[c, d]
                )

                if delta < best[0]:
                    best = (delta, (i, j))

                    if strategy == "First Improvement":
                        break

            if strategy == "First Improvement" and best[1]:
                break

        if best[1] is None:
            break

        i, j = best[1]
        r[i:j + 1] = r[i:j + 1][::-1]

    return r


# ---------- Simulated Annealing ----------

def simulated_annealing(
    r, D, T0=1000, alpha=0.9, Tmin=0.1, per_T=1, max_iter=5000, op="2-opt", seed=0,
):
    """Simulated Annealing: satu kandidat per iterasi, geometric cooling.

    per_T tetap diterima demi kompatibilitas pemanggil lama, tetapi sengaja
    diabaikan: tidak ada kontrol 'iterasi per level suhu'. Suhu dijadwalkan
    agar Tmin tercapai pada batas max_iter.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    current = list(r)
    current_dist = length(current, D)
    best, best_dist = current[:], current_dist
    moves = all_moves(len(r), op) if 'all_moves' in globals() else None
    if not moves:
        moves = [(op, i, j, 0) for i in range(len(r)) for j in range(i+1, len(r))]
    if not moves:
        raise ValueError("Tidak ada gerakan yang dapat dilakukan.")
    T0_actual = max(float(T0), float(Tmin) * 1.000001, 1e-12)
    Tmin_actual = max(float(Tmin), 1e-12)
    alpha_used = min(max(float(alpha), 1e-6), 0.999999)
    temperature = T0_actual
    hist, log = [best_dist], []
    accepted = accepted_worse = best_updates = evals = 0
    iteration = 0
    while iteration < int(max_iter) and temperature > Tmin_actual:
        iteration += 1
        move = rand_move(len(r), op, rng)
        candidate = apply(current, move)
        candidate_dist = length(candidate, D)
        evals += 1
        delta = candidate_dist - current_dist
        take = delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12))
        if take:
            accepted += 1
            if delta > 0:
                accepted_worse += 1
            current, current_dist = candidate, candidate_dist
        updated = current_dist < best_dist - 1e-12
        if updated:
            best, best_dist = current[:], current_dist
            best_updates += 1
        hist.append(best_dist)
        log.append({"Iteration": iteration, "Temperature": temperature, "Move": str(move[:3]),
                    "Delta": delta, "Accepted": bool(take), "Current_Distance": current_dist,
                    "Best_Distance": best_dist, "Best_Updated": updated,
                    "Worse_Move_Accepted": bool(take and delta > 0)})
        temperature *= alpha_used
    reason = "Suhu minimum tercapai" if temperature <= Tmin_actual else "Batas iterasi bersama tercapai"
    return {"route": best, "dist": best_dist, "hist": hist, "log": log,
            "iters": iteration, "iterations": iteration, "evaluations": evals,
            "accepted": accepted, "accepted_worse": accepted_worse, "best_updates": best_updates,
            "T0_actual": T0_actual, "Tmin_actual": Tmin_actual,
            "alpha_effective": alpha_used, "T_final": temperature,
            "time": time.perf_counter() - t0, "stop_reason": reason}


# ---------- Tabu Search ----------

def tabu_search(
    r, D, max_iter=500, tabu_size=10, nbr_size=50, op="2-opt", aspiration=True, seed=0, detail=False,
):
    """Tabu Search dengan waktu kedaluwarsa tabu dalam iterasi.

    Atribut gerakan yang baru saja dihapus/dilepas dari rute dilarang dibentuk
    kembali selama tenure. Kandidat tabu hanya boleh diterima oleh aspiration
    jika mengalahkan solusi terbaik global. Tidak ada fallback yang melanggar tabu.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    current = list(r)
    current_dist = length(current, D)
    best, best_dist = current[:], current_dist
    all_candidates = [(op, i, j, 0) for i in range(len(r)) for j in range(i+1, len(r))]
    if op == "3-opt":
        all_candidates = [rand_move(len(r), op, rng) for _ in range(max(100, int(nbr_size)*5))]
    if not all_candidates:
        raise ValueError("Tidak ada gerakan yang dapat dilakukan.")
    tenure = max(1, int(tabu_size))
    tabu_expiry = {}
    hist, log = [best_dist], []
    evals = best_updates = skipped = aspiration_used = 0
    for iteration in range(1, int(max_iter)+1):
        expired = [key for key, expiry in tabu_expiry.items() if expiry <= iteration]
        for key in expired:
            del tabu_expiry[key]
        count = min(max(1, int(nbr_size)), len(all_candidates))
        ids = rng.choice(len(all_candidates), size=count, replace=False)
        candidates = []
        for idx in ids:
            move = all_candidates[int(idx)]
            route = apply(current, move)
            dist = length(route, D)
            key = move_key(current, move)
            is_tabu = tabu_expiry.get(key, 0) > iteration
            allowed_asp = bool(is_tabu and aspiration and dist < best_dist - 1e-12)
            candidates.append((move, route, dist, key, is_tabu, allowed_asp))
            evals += 1
        valid = [x for x in candidates if not x[4] or x[5]]
        chosen = min(valid, key=lambda x: x[2]) if valid else None
        if chosen is None:
            skipped += 1
            hist.append(best_dist)
            log.append({"Iteration": iteration, "Selected_Move": "None (all candidates tabu)",
                        "Current_Distance": current_dist, "Best_Distance": best_dist,
                        "Candidates_Evaluated": count, "Tabu_Candidates": sum(x[4] for x in candidates),
                        "Aspiration_Used": False, "Expired": str(expired),
                        "Tabu_List": str({str(k): v for k,v in tabu_expiry.items()}),
                        "Left": str(expired), "Best_Updated": False, "Status": "Skipped; no admissible move"})
            continue
        move, route, dist, key, is_tabu, used_asp = chosen
        old_route, old_dist = current[:], current_dist
        current, current_dist = route, dist
        # expiry means first iteration where the attribute is no longer tabu
        tabu_expiry[key] = iteration + tenure + 1
        updated = current_dist < best_dist - 1e-12
        if updated:
            best, best_dist = current[:], current_dist
            best_updates += 1
        aspiration_used += int(used_asp)
        hist.append(best_dist)
        log.append({"Iteration": iteration, "Selected_Move": str(move[:3]),
                    "Current_Before": [x+1 for x in old_route], "Current_After": [x+1 for x in current],
                    "Current_Distance": current_dist, "Best_Distance": best_dist,
                    "Candidates_Evaluated": count, "Tabu_Candidates": sum(x[4] for x in candidates),
                    "Aspiration_Used": used_asp, "Tabu_List": str({str(k): v for k,v in tabu_expiry.items()}),
                    "Left": str(expired), "Best_Updated": updated,
                    "Candidates": [{"Move": str(x[0][:3]), "Distance": round(x[2],4),
                                    "Tabu": "Yes" if x[4] else "No", "Aspiration": x[5],
                                    "Selected": "✓" if x is chosen else ""} for x in candidates],
                    "Status": "Aspiration" if used_asp else "Admissible best"})
    return {"route": best, "dist": best_dist, "hist": hist, "log": log,
            "iters": int(max_iter), "iterations": int(max_iter), "evaluations": evals,
            "best_updates": best_updates, "skipped_iters": skipped, "aspiration_used": aspiration_used,
            "time": time.perf_counter() - t0, "stop_reason": "Batas iterasi bersama tercapai"}


# ---------- Variable Neighborhood Search ----------

def _vns_shake(route, operator, rng, moves=1):
    """Menghasilkan kandidat melalui shaking pada neighborhood tertentu."""
    candidate = route[:]
    n = len(candidate)

    for _ in range(max(1, int(moves))):
        i, j = map(int, rng.choice(n, 2, replace=False))

        if operator == "Swap":
            candidate[i], candidate[j] = candidate[j], candidate[i]

        elif operator == "Insertion":
            city = candidate.pop(i)
            candidate.insert(j, city)

        elif operator == "2-opt":
            i, j = sorted((i, j))
            candidate[i:j + 1] = candidate[i:j + 1][::-1]

        else:
            raise ValueError(
                f"Neighborhood VNS tidak dikenal: {operator}"
            )

    return candidate


def variable_neighborhood_search(
    r,
    D,
    max_iter=500,
    k_max=3,
    neighborhoods=None,
    shaking="Satu move acak",
    local_search=True,
    strategy="Best Improvement",
    seed=0,
):
    """
    Variable Neighborhood Search (VNS).

    Menggunakan shaking, perpindahan neighborhood, dan optional 2-opt.
    Seluruh jarak dihitung dengan matriks D yang diberikan.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    allowed = ["Swap", "Insertion", "2-opt"]
    neighborhoods = (
        allowed[:]
        if neighborhoods is None
        else [op for op in neighborhoods if op in allowed]
    )

    if not neighborhoods:
        raise ValueError("Pilih minimal satu neighborhood VNS.")

    k_max = max(1, min(int(k_max), len(neighborhoods)))
    neighborhoods = neighborhoods[:k_max]

    current = r[:]
    current_dist = length(current, D)
    best = current[:]
    best_dist = current_dist

    hist = [best_dist]
    log = []
    k = 0

    for iteration in range(1, max(0, int(max_iter)) + 1):
        operator = neighborhoods[k]
        neighborhood_index = k + 1

        moves = (
            1 if shaking == "Satu move acak"
            else neighborhood_index
        )

        candidate = _vns_shake(
            current, operator, rng, moves
        )

        if local_search:
            candidate = two_opt_ls(candidate, D, strategy)

        candidate_dist = length(candidate, D)
        accepted = candidate_dist < current_dist - 1e-12

        if accepted:
            current = candidate
            current_dist = candidate_dist

            if current_dist < best_dist - 1e-12:
                best = current[:]
                best_dist = current_dist

            # Perbaikan: kembali ke neighborhood pertama.
            k = 0
        else:
            # Tidak ada perbaikan: pindah ke neighborhood berikutnya.
            k = (k + 1) % len(neighborhoods)

        hist.append(best_dist)
        log.append({
            "Iteration": iteration,
            "Neighborhood": operator,
            "Neighborhood_Index": neighborhood_index,
            "Shaking_Moves": moves,
            "Candidate_Distance": round(candidate_dist, 4),
            "Current_Distance": round(current_dist, 4),
            "Best_Distance": round(best_dist, 4),
            "Accepted": bool(accepted),
        })

    return {
        "route": best,
        "dist": best_dist,
        "hist": hist,
        "log": log,
        "iters": len(log),
        "time": time.perf_counter() - t0,
    }
