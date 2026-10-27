
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

    raise ValueError(f"Move operator tidak dikenal: {op}")


def rand_move(n, op, rng):
    if op not in ("Swap", "2-opt"):
        raise ValueError(f"Move operator tidak dikenal: {op}")
    if n < 2:
        raise ValueError("Minimal dua kota diperlukan untuk membuat gerakan.")
    i, j = sorted(rng.choice(n, 2, replace=False))
    return op, int(i), int(j), 0


def neighborhood(n, op, size, rng):
    if op not in ("Swap", "2-opt"):
        raise ValueError(f"Move operator tidak dikenal: {op}")
    all_moves = [
        (op, i, j, 0)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    if not all_moves:
        return []
    count = min(max(1, int(size)), len(all_moves))
    if count == len(all_moves):
        return all_moves
    indices = rng.choice(len(all_moves), size=count, replace=False)
    return [all_moves[int(i)] for i in indices]


def move_key(r, m):
    op, i, j, k = m
    cities = [r[i], r[j]]
    return op, tuple(sorted(cities))


# ---------- Local search ----------

def _local_search_candidates(route, D, operator):
    """Yield (candidate_route, move_description, operator_name) for a neighborhood."""
    n = len(route)
    operators = ["2-opt", "3-opt"] if operator == "2-opt + 3-opt" else [operator]

    for op in operators:
        if op == "2-opt":
            # Keep position 0 fixed to avoid redundant cyclic rotations.
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    candidate = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    yield candidate, f"2-opt (posisi {i + 1}, {j + 1})", op
        elif op == "3-opt":
            # Cut three edges between consecutive segments A | B | C | D,
            # then enumerate the seven non-identity reconnections of B and C.
            # The closing edge D->A remains part of the tour; k may reach n-1.
            seen = set()
            for i in range(0, n - 3):
                A = route[:i + 1]
                for j in range(i + 1, n - 2):
                    B = route[i + 1:j + 1]
                    for k in range(j + 1, n):
                        Cseg = route[j + 1:k + 1]
                        Dseg = route[k + 1:]
                        variants = [
                            (B[::-1], Cseg),
                            (B, Cseg[::-1]),
                            (B[::-1], Cseg[::-1]),
                            (Cseg, B),
                            (Cseg[::-1], B),
                            (Cseg, B[::-1]),
                            (Cseg[::-1], B[::-1]),
                        ]
                        for left, right in variants:
                            candidate = A + left + right + Dseg
                            key = tuple(candidate)
                            if key == tuple(route) or key in seen:
                                continue
                            seen.add(key)
                            yield candidate, f"3-opt (cut {i + 1}, {j + 1}, {k + 1})", op
        else:
            raise ValueError(f"Operator Local Search tidak dikenal: {operator}")


def local_search(r, D, strategy="First Improvement", operator="2-opt", max_pass=200, return_log=False, max_candidates_per_pass=50000):
    """Local Search baseline with 2-opt, 3-opt, or combined neighborhoods.

    First Improvement accepts the first improving candidate encountered.
    Best Improvement scans the selected neighborhood(s) and accepts the best
    improving candidate. A final log row records the no-improvement stop.
    """
    route = list(r)
    history = [length(route, D)]
    log = []
    total_evaluations = 0
    stop_reason = "Batas iterasi maksimum tercapai"
    valid_operators = {"2-opt", "3-opt", "2-opt + 3-opt"}
    if operator not in valid_operators:
        raise ValueError(f"Operator Local Search tidak dikenal: {operator}")
    if strategy not in {"First Improvement", "Best Improvement"}:
        raise ValueError(f"Strategi Local Search tidak dikenal: {strategy}")

    for iteration in range(1, max(0, int(max_pass)) + 1):
        before = length(route, D)
        selected = None
        best_improvement = 1e-12
        evaluations = 0
        neighborhood_truncated = False

        for candidate, move_desc, op_name in _local_search_candidates(route, D, operator):
            if evaluations >= max(1, int(max_candidates_per_pass)):
                neighborhood_truncated = True
                break
            evaluations += 1
            candidate_dist = length(candidate, D)
            improvement = before - candidate_dist
            if improvement > 1e-12:
                if strategy == "First Improvement":
                    selected = (candidate, move_desc, op_name, candidate_dist, improvement)
                    break
                if improvement > best_improvement:
                    best_improvement = improvement
                    selected = (candidate, move_desc, op_name, candidate_dist, improvement)

        total_evaluations += evaluations
        if selected is None:
            no_move_status = (
                f"Tidak ada perbaikan dalam {evaluations} kandidat pertama; "
                "pencarian neighborhood dibatasi demi performa"
                if neighborhood_truncated else
                f"Tidak ada perbaikan; optimum lokal {operator}"
            )
            log.append({
                "Iteration": iteration,
                "Operator": operator,
                "Distance_Before": round(before, 6),
                "Distance_After": round(before, 6),
                "Improvement": 0.0,
                "Move": "—",
                "Candidates_Evaluated": evaluations,
                "Accepted": False,
                "Status": no_move_status,
            })
            history.append(before)
            stop_reason = (
                f"Batas kandidat per iterasi tercapai ({evaluations}); belum dapat memastikan optimum lokal"
                if neighborhood_truncated else
                f"Optimum lokal terhadap neighborhood {operator}"
            )
            break

        candidate, move_desc, op_name, after, improvement = selected
        route = candidate
        log.append({
            "Iteration": iteration,
            "Operator": op_name,
            "Distance_Before": round(before, 6),
            "Distance_After": round(after, 6),
            "Improvement": round(improvement, 6),
            "Move": move_desc,
            "Candidates_Evaluated": evaluations,
            "Accepted": True,
            "Status": "Perbaikan diterima",
        })
        history.append(after)

    result = {
        "route": route,
        "dist": length(route, D),
        "hist": history,
        "log": log,
        "iters": len(log),
        "iterations": len(log),
        "evaluations": total_evaluations,
        "stop_reason": stop_reason,
        "operator": operator,
        "strategy": strategy,
        "max_candidates_per_pass": int(max_candidates_per_pass),
    }
    return result if return_log else route


def two_opt_ls(r, D, strategy="First Improvement", max_pass=200, return_log=False):
    """Backward-compatible 2-opt-only wrapper around :func:`local_search`."""
    return local_search(
        r, D, strategy=strategy, operator="2-opt",
        max_pass=max_pass, return_log=return_log,
    )


# ---------- Simulated Annealing ----------

def simulated_annealing(
    r, D, T0=1000, alpha=0.9, Tmin=0.1, per_T=1, max_iter=5000, op="2-opt", seed=0,
):
    """Simulated Annealing with geometric cooling every ``per_T`` iterations.

    If Tmin is reached before max_iter, the search continues at Tmin until
    the shared/selected iteration budget is consumed.
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
    while iteration < int(max_iter):
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
        # Cool only after the configured number of iterations at this temperature.
        # Once Tmin is reached, continue evaluating candidates at Tmin until max_iter.
        if iteration % max(1, int(per_T)) == 0:
            temperature = max(Tmin_actual, temperature * alpha_used)
    reason = (
        "Batas iterasi tercapai; suhu minimum pernah tercapai"
        if temperature <= Tmin_actual
        else "Batas iterasi tercapai sebelum suhu minimum"
    )
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
    """Tabu Search dengan tenure berbasis iterasi dan log yang mudah diaudit.

    Tabu List tetap menyimpan atribut gerakan (operator + pasangan kota), bukan
    melarang kota dikunjungi. Log menyediakan tabel terstruktur agar UI tidak
    perlu menampilkan representasi dictionary Python.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    current = list(r)
    current_dist = length(current, D)
    best, best_dist = current[:], current_dist
    if op not in ("Swap", "2-opt"):
        raise ValueError(f"Operator Tabu Search tidak dikenal: {op}")
    all_candidates = [(op, i, j, 0) for i in range(len(r)) for j in range(i + 1, len(r))]
    if not all_candidates:
        raise ValueError("Tidak ada gerakan yang dapat dilakukan.")

    tenure = max(1, int(tabu_size))
    tabu_expiry = {}       # expiry = iterasi pertama saat atribut boleh dipakai lagi
    tabu_started = {}      # iterasi saat atribut terakhir kali dimasukkan/diperpanjang
    hist, log = [best_dist], []
    evals = best_updates = skipped = aspiration_used = 0

    def move_label(route, move):
        operator, i, j, _ = move
        a, b = int(route[i]) + 1, int(route[j]) + 1
        return f"{operator} (node {a}, node {b})"

    def tabu_rows(iteration_now):
        rows = []
        for key, expiry in sorted(tabu_expiry.items(), key=lambda item: (item[1], str(item[0]))):
            operator, cities = key
            nodes = [int(city) + 1 for city in cities]
            rows.append({
                "Atribut tabu": f"{operator} (node {nodes[0]}, node {nodes[1]})",
                "Node 1": nodes[0],
                "Node 2": nodes[1],
                "Mulai tabu": int(tabu_started.get(key, max(1, expiry - tenure - 1))),
                "Tabu sampai iterasi": int(expiry - 1),
                "Boleh digunakan mulai": int(expiry),
                "Sisa tenure": max(0, int(expiry) - int(iteration_now) - 1),
                "Status": "Masih tabu" if expiry > iteration_now else "Kedaluwarsa",
            })
        return rows

    def candidate_rows(candidates, chosen, current_before, iteration_now):
        rows = []
        for idx, item in enumerate(candidates, start=1):
            move, route, dist, key, is_tabu, allowed_asp = item
            rows.append({
                "No.": idx,
                "Gerakan": move_label(current_before, move),
                "Jarak kandidat": round(float(dist), 6),
                "Perubahan jarak": round(float(dist - length(current_before, D)), 6),
                "Tabu?": "Ya" if is_tabu else "Tidak",
                "Aspiration?": "Ya" if allowed_asp else "Tidak",
                "Dipilih?": "Ya" if item is chosen else "Tidak",
            })
        return rows

    for iteration in range(1, int(max_iter) + 1):
        expired_keys = [key for key, expiry in tabu_expiry.items() if expiry <= iteration]
        expired_rows = []
        for key in expired_keys:
            operator, cities = key
            nodes = [int(city) + 1 for city in cities]
            expired_rows.append({
                "Atribut tabu": f"{operator} (node {nodes[0]}, node {nodes[1]})",
                "Node 1": nodes[0], "Node 2": nodes[1],
                "Mulai tabu": int(tabu_started.get(key, max(1, tabu_expiry[key] - tenure - 1))),
                "Tabu sampai iterasi": int(tabu_expiry[key] - 1),
                "Boleh digunakan mulai": int(tabu_expiry[key]),
            })
            del tabu_expiry[key]
            tabu_started.pop(key, None)

        count = min(max(1, int(nbr_size)), len(all_candidates))
        ids = rng.choice(len(all_candidates), size=count, replace=False)
        candidates = []
        current_before = current[:]
        current_before_dist = current_dist
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
            log.append({
                "Iteration": iteration,
                "Selected_Move": "Tidak ada (semua kandidat tabu)",
                "Current_Before": [x + 1 for x in current_before],
                "Current_After": [x + 1 for x in current],
                "Current_Distance": current_dist,
                "Best_Distance": best_dist,
                "Candidates_Evaluated": count,
                "Tabu_Candidates": sum(x[4] for x in candidates),
                "Aspiration_Used": False,
                "Expired": expired_rows,
                "Tabu_List": str({str(k): v for k, v in tabu_expiry.items()}),
                "Tabu_List_Table": tabu_rows(iteration),
                "Left": expired_rows,
                "Candidates": candidate_rows(candidates, None, current_before, iteration),
                "Best_Updated": False,
                "Status": "Dilewati; tidak ada gerakan yang diizinkan",
            })
            continue

        move, route, dist, key, is_tabu, used_asp = chosen
        current, current_dist = route, dist
        # Atribut yang dipilih menjadi tabu selama `tenure` iterasi berikutnya.
        tabu_started[key] = iteration
        tabu_expiry[key] = iteration + tenure + 1
        updated = current_dist < best_dist - 1e-12
        if updated:
            best, best_dist = current[:], current_dist
            best_updates += 1
        aspiration_used += int(used_asp)
        hist.append(best_dist)
        log.append({
            "Iteration": iteration,
            "Selected_Move": move_label(current_before, move),
            "Current_Before": [x + 1 for x in current_before],
            "Current_After": [x + 1 for x in current],
            "Current_Distance": current_dist,
            "Best_Distance": best_dist,
            "Candidates_Evaluated": count,
            "Tabu_Candidates": sum(x[4] for x in candidates),
            "Aspiration_Used": bool(used_asp),
            "Expired": expired_rows,
            "Tabu_List": str({str(k): v for k, v in tabu_expiry.items()}),
            "Tabu_List_Table": tabu_rows(iteration),
            "Left": expired_rows,
            "Candidates": candidate_rows(candidates, chosen, current_before, iteration),
            "Best_Updated": bool(updated),
            "Status": "Aspiration" if used_asp else "Gerakan yang diizinkan",
        })

    return {
        "route": best, "dist": best_dist, "hist": hist, "log": log,
        "iters": int(max_iter), "iterations": int(max_iter), "evaluations": evals,
        "best_updates": best_updates, "skipped_iters": skipped,
        "aspiration_used": aspiration_used,
        "time": time.perf_counter() - t0,
        "stop_reason": "Batas iterasi bersama tercapai",
    }


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
    best_updates = 0

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
                best_updates += 1

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
        # One candidate-level evaluation per VNS iteration. Any evaluations
        # performed internally by the optional 2-opt local search are not included.
        "evaluations": len(log),
        "best_updates": best_updates,
        "stop_reason": "Batas iterasi bersama tercapai",
        "time": time.perf_counter() - t0,
    }
