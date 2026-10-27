import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsp_core import (
    dist_matrix, length, two_opt_ls,
    simulated_annealing, tabu_search, variable_neighborhood_search,
)


def sample():
    points = np.array([
        [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0],
        [1.0, 0.3], [1.7, 1.4], [0.2, 1.0], [2.5, 0.7],
    ])
    return points, dist_matrix(points, "Euclidean"), list(range(len(points)))


def assert_valid(route, n):
    assert len(route) == n
    assert sorted(route) == list(range(n))


def test_sa_uses_shared_iteration_budget_and_valid_route():
    _, D, route = sample()
    result = simulated_annealing(route, D, 100.0, 0.9, 0.1, 1, 40, "2-opt", 123)
    assert result["iters"] == 40
    assert len(result["hist"]) == 41
    assert result["evaluations"] == 40
    assert_valid(result["route"], len(route))


def test_tabu_search_uses_shared_iteration_budget_and_valid_route():
    _, D, route = sample()
    result = tabu_search(route, D, 40, 5, 20, "2-opt", True, 123)
    assert result["iters"] == 40
    assert len(result["hist"]) == 41
    assert result["evaluations"] >= 40
    assert_valid(result["route"], len(route))


def test_vns_uses_shared_iteration_budget_and_valid_route():
    _, D, route = sample()
    result = variable_neighborhood_search(
        route, D, max_iter=40, k_max=3,
        neighborhoods=["Swap", "Insertion", "2-opt"],
        shaking="Satu move acak", local_search=True,
        strategy="Best Improvement", seed=123,
    )
    assert result["iters"] == 40
    assert len(result["hist"]) == 41
    assert result["evaluations"] == 40
    assert_valid(result["route"], len(route))


def test_two_opt_baseline_never_worsens_symmetric_tsp():
    _, D, route = sample()
    before = length(route, D)
    improved = two_opt_ls(route, D, "Best Improvement")
    assert_valid(improved, len(route))
    assert length(improved, D) <= before + 1e-9


def test_sa_reproducible_with_same_seed():
    _, D, route = sample()
    a = simulated_annealing(route, D, 100.0, 0.9, 0.1, 1, 40, "Swap", 9)
    b = simulated_annealing(route, D, 100.0, 0.9, 0.1, 1, 40, "Swap", 9)
    assert a["route"] == b["route"]
    assert abs(a["dist"] - b["dist"]) < 1e-9
