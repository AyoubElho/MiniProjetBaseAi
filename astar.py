"""
astar.py — Implémentation de A*, UCS et Greedy sur une grille.
"""
import heapq
import time
import tracemalloc
from grid import get_neighbors, manhattan, zero_heuristic, reconstruct_path


# ──────────────────────────────────────────────
# Recherche générique (paramétrisée)
# ──────────────────────────────────────────────

def search(grid, start, goal, heuristic, use_g=True, use_h=True, name="A*"):
    """
    Recherche heuristique générique.

    Paramètres
    ----------
    grid      : np.ndarray  (0=libre, 1=obstacle)
    start     : tuple (r, c)
    goal      : tuple (r, c)
    heuristic : callable(state, goal) -> float
    use_g     : bool  — inclure g(n) dans f ?  (False → Greedy)
    use_h     : bool  — inclure h(n) dans f ?  (False → UCS)
    name      : str   — nom pour affichage

    Retourne
    --------
    dict avec path, cost, nodes_expanded, open_size_max, time_s, memory_kb
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    open_set = []          # (f, g, état)
    g_score = {start: 0}
    came_from = {}
    closed_set = set()
    nodes_expanded = 0
    open_size_max = 0

    h0 = heuristic(start, goal)
    f0 = (g_score[start] if use_g else 0) + (h0 if use_h else 0)
    heapq.heappush(open_set, (f0, 0, start))

    while open_set:
        open_size_max = max(open_size_max, len(open_set))
        f, g, current = heapq.heappop(open_set)

        if current in closed_set:
            continue
        closed_set.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            elapsed = time.perf_counter() - t0
            _, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return {
                "name": name,
                "path": path,
                "cost": g_score[goal],
                "nodes_expanded": nodes_expanded,
                "open_size_max": open_size_max,
                "time_s": elapsed,
                "memory_kb": mem_peak / 1024,
                "found": True,
            }

        for neighbor, step_cost in get_neighbors(current, grid):
            if neighbor in closed_set:
                continue
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = heuristic(neighbor, goal)
                f = (tentative_g if use_g else 0) + (h if use_h else 0)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    elapsed = time.perf_counter() - t0
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "name": name,
        "path": None,
        "cost": float("inf"),
        "nodes_expanded": nodes_expanded,
        "open_size_max": open_size_max,
        "time_s": elapsed,
        "memory_kb": mem_peak / 1024,
        "found": False,
    }


# ──────────────────────────────────────────────
# API simplifiée
# ──────────────────────────────────────────────

def astar(grid, start, goal, heuristic=manhattan):
    return search(grid, start, goal, heuristic, use_g=True, use_h=True, name="A*")

def ucs(grid, start, goal):
    return search(grid, start, goal, zero_heuristic, use_g=True, use_h=False, name="UCS")

def greedy(grid, start, goal, heuristic=manhattan):
    return search(grid, start, goal, heuristic, use_g=False, use_h=True, name="Greedy")

def astar_zero(grid, start, goal):
    """A* avec h=0 (équivaut UCS, utile pour comparaison heuristique)."""
    return search(grid, start, goal, zero_heuristic, use_g=True, use_h=True, name="A*(h=0)")


# ──────────────────────────────────────────────
# Vérification admissibilité et cohérence
# ──────────────────────────────────────────────

def check_admissibility(grid, goal, heuristic, name="h"):
    """
    Vérifie l'admissibilité sur toutes les cellules libres :
    h(n) <= h*(n) (coût réel optimal avec UCS).
    Retourne (is_admissible, violations)
    """
    import numpy as np
    # Calculer h* pour chaque cellule libre via UCS depuis chaque état
    rows, cols = grid.shape
    violations = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                continue
            state = (r, c)
            res = ucs(grid, state, goal)
            if res["found"]:
                h_star = res["cost"]
                h_val = heuristic(state, goal)
                if h_val > h_star + 1e-9:
                    violations.append((state, h_val, h_star))
    is_adm = len(violations) == 0
    return is_adm, violations


def check_consistency(grid, heuristic, goal):
    """
    Vérifie la cohérence (inégalité triangulaire) :
    h(n) <= c(n, n') + h(n') pour chaque arc.
    """
    rows, cols = grid.shape
    violations = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                continue
            state = (r, c)
            h_n = heuristic(state, goal)
            for neighbor, cost in get_neighbors(state, grid):
                h_np = heuristic(neighbor, goal)
                if h_n > cost + h_np + 1e-9:
                    violations.append((state, neighbor, h_n, cost, h_np))
    return len(violations) == 0, violations
