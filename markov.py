"""
markov.py — Construction et analyse de la chaîne de Markov induite.
"""
import numpy as np
from grid import get_neighbors, path_to_policy


# ──────────────────────────────────────────────
# Indexation des états
# ──────────────────────────────────────────────

GOAL_IDX = -2   # état absorbant GOAL
FAIL_IDX = -1   # état absorbant FAIL (collision)


def build_state_index(grid, path):
    """
    Construit un index état <-> entier pour tous les états du chemin
    + états adjacents atteignables + GOAL + FAIL.
    """
    rows, cols = grid.shape
    accessible = set()

    # Tous les états accessibles depuis le chemin
    for state in path:
        accessible.add(state)
        for nb, _ in get_neighbors(state, grid):
            accessible.add(nb)

    # Trier de manière déterministe
    sorted_states = sorted(accessible)
    idx = {s: i for i, s in enumerate(sorted_states)}
    # Ajouter GOAL et FAIL
    n = len(sorted_states)
    GOAL = n
    FAIL = n + 1
    return sorted_states, idx, GOAL, FAIL


def build_transition_matrix(grid, path, epsilon, goal_state):
    """
    Construit la matrice de transition stochastique P.

    Modèle d'incertitude avec paramètre ε :
      - action voulue : prob 1 - ε
      - déviation latérale gauche : prob ε/2
      - déviation latérale droite : prob ε/2
      - si déviation mène à un obstacle/hors grille → prob reportée sur place (FAIL si option)

    Paramètres
    ----------
    epsilon    : taux de déviation [0, 1)
    goal_state : tuple (r,c) — état but

    Retourne
    --------
    P          : np.ndarray (n+2) x (n+2)  — matrice stochastique
    states     : liste des états (sans GOAL/FAIL)
    idx        : dict état -> index
    GOAL, FAIL : int — indices dans P
    """
    policy = path_to_policy(path)

    # Index
    states, idx, GOAL, FAIL = build_state_index(grid, path)
    n_states = len(states)
    N = n_states + 2  # + GOAL + FAIL

    P = np.zeros((N, N))

    # Directions orthogonales pour calculer déviations latérales
    def laterals(dr, dc):
        """Retourne les deux directions perpendiculaires."""
        return [(-dc, dr), (dc, -dr)]

    rows, cols = grid.shape

    def try_move(r, c, dr, dc):
        """Tente de déplacer (r,c) dans (dr,dc). Retourne nouvel état ou None si bloqué."""
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            return (nr, nc)
        return None  # collision / hors grille

    for s in states:
        i = idx[s]
        r, c = s

        # État but → absorbant GOAL
        if s == goal_state:
            P[i, GOAL] = 1.0
            continue

        # Politique : action recommandée
        action = policy.get(s, None)

        if action is None:
            # État hors politique (pas sur le chemin) → rester ou vers goal si adjacent
            # On choisit l'action vers le goal si possible, sinon rester
            nb_states = get_neighbors(s, grid)
            if nb_states:
                # Action aléatoire uniforme parmi voisins
                prob = 1.0 / len(nb_states)
                for (nb, _) in nb_states:
                    if nb == goal_state:
                        P[i, GOAL] += prob
                    elif nb in idx:
                        P[i, idx[nb]] += prob
                    else:
                        P[i, FAIL] += prob
            else:
                P[i, i] = 1.0  # bloqué
            continue

        dr, dc = action

        # ── Action principale ──
        dest_main = try_move(r, c, dr, dc)
        prob_main = 1.0 - epsilon

        # ── Déviations latérales ──
        lat_dirs = laterals(dr, dc)
        prob_lat = epsilon / 2.0

        moves = [(dest_main, prob_main)] + [(try_move(r, c, ld[0], ld[1]), prob_lat) for ld in lat_dirs]

        for dest, prob in moves:
            if prob == 0:
                continue
            if dest is None:
                # Collision → FAIL
                P[i, FAIL] += prob
            elif dest == goal_state:
                P[i, GOAL] += prob
            elif dest in idx:
                P[i, idx[dest]] += prob
            else:
                # Destination accessible mais pas indexée (cas rare)
                P[i, FAIL] += prob

    # GOAL et FAIL sont absorbants
    P[GOAL, GOAL] = 1.0
    P[FAIL, FAIL] = 1.0

    # Vérification stochasticité
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), f"P n'est pas stochastique ! Sommes : {row_sums}"

    return P, states, idx, GOAL, FAIL


# ──────────────────────────────────────────────
# Évolution de la distribution
# ──────────────────────────────────────────────

def compute_pi_n(P, start_idx, n_steps):
    """
    Calcule π^(n) = π^(0) P^n.
    Retourne tableau (n_steps+1, N) — distributions à chaque étape.
    """
    N = P.shape[0]
    pi = np.zeros(N)
    pi[start_idx] = 1.0

    history = [pi.copy()]
    for _ in range(n_steps):
        pi = pi @ P
        history.append(pi.copy())
    return np.array(history)


def prob_reach_goal(pi_history, GOAL):
    """Probabilité d'être dans GOAL à chaque pas de temps."""
    return pi_history[:, GOAL]


# ──────────────────────────────────────────────
# Analyse d'absorption
# ──────────────────────────────────────────────

def absorption_analysis(P, states, idx, GOAL, FAIL):
    """
    Analyse d'absorption via la décomposition canonique.

    P = [ I  0 ]
        [ R  Q ]

    N = (I - Q)^-1  (matrice fondamentale)
    B = N R          (probabilités d'absorption)

    Retourne dict avec :
      - absorbing_probs : (nb_transients, 2)  — [P(GOAL), P(FAIL)]
      - mean_absorption_times : vecteur
      - fundamental_matrix : N
    """
    n_transients = len(states)  # indices 0..n_transients-1
    # Sous-matrice Q (transients -> transients)
    Q = P[:n_transients, :n_transients]
    # Sous-matrice R (transients -> absorbants)
    R = P[:n_transients, n_transients:]  # colonnes GOAL, FAIL

    I = np.eye(n_transients)
    try:
        N_fund = np.linalg.inv(I - Q)   # matrice fondamentale
    except np.linalg.LinAlgError:
        N_fund = None
        absorbing_probs = None
        mean_times = None
        return {
            "fundamental_matrix": None,
            "absorbing_probs": None,
            "mean_absorption_times": None,
        }

    B = N_fund @ R          # probabilités d'absorption
    t = N_fund @ np.ones(n_transients)  # temps moyen d'absorption

    return {
        "fundamental_matrix": N_fund,
        "absorbing_probs": B,          # col 0 = GOAL, col 1 = FAIL
        "mean_absorption_times": t,
        "Q": Q, "R": R,
    }


# ──────────────────────────────────────────────
# Identification des classes
# ──────────────────────────────────────────────

def communication_classes(P, states, GOAL, FAIL):
    """
    Identifie les classes de communication via DFS dans le graphe orienté.
    Retourne : classes (list of sets), état_absorbant ou transitoire
    """
    n = P.shape[0]
    all_labels = list(states) + ["GOAL", "FAIL"]

    # Construction du graphe de voisinage
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0:
                adj[i].append(j)

    # Kosaraju SCC
    visited = set()
    finish_order = []

    def dfs1(v):
        stack = [(v, iter(adj[v]))]
        visited.add(v)
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(adj[child])))
            except StopIteration:
                finish_order.append(node)
                stack.pop()

    for v in range(n):
        if v not in visited:
            dfs1(v)

    # Graphe transposé
    adj_t = {i: [] for i in range(n)}
    for i in range(n):
        for j in adj[i]:
            adj_t[j].append(i)

    visited2 = set()
    components = []

    def dfs2(v, comp):
        stack = [v]
        while stack:
            node = stack.pop()
            if node in visited2:
                continue
            visited2.add(node)
            comp.add(node)
            for nb in adj_t[node]:
                if nb not in visited2:
                    stack.append(nb)

    for v in reversed(finish_order):
        if v not in visited2:
            comp = set()
            dfs2(v, comp)
            components.append(comp)

    return components, all_labels


# ──────────────────────────────────────────────
# Simulation Monte-Carlo
# ──────────────────────────────────────────────

def simulate_trajectories(P, start_idx, GOAL, FAIL, n_sim=5000, max_steps=1000, rng=None):
    """
    Simule N trajectoires de la chaîne de Markov.

    Retourne
    --------
    dict avec :
      - prob_goal   : float  — proportion atteignant GOAL
      - prob_fail   : float  — proportion atteignant FAIL
      - mean_time_goal  : float  — temps moyen pour atteindre GOAL (si atteint)
      - std_time_goal   : float
      - times_goal  : list
      - times_fail  : list
      - trajectories_sample : 5 premières trajectoires (pour visualisation)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = P.shape[0]
    times_goal = []
    times_fail = []
    times_neither = []
    sample_trajectories = []

    for sim in range(n_sim):
        state = start_idx
        trajectory = [state]
        for step in range(max_steps):
            # Transition stochastique
            state = rng.choice(N, p=P[state])
            trajectory.append(state)
            if state == GOAL:
                times_goal.append(step + 1)
                break
            if state == FAIL:
                times_fail.append(step + 1)
                break
        else:
            times_neither.append(max_steps)

        if sim < 5:
            sample_trajectories.append(trajectory)

    total = n_sim
    prob_goal = len(times_goal) / total
    prob_fail = len(times_fail) / total
    prob_neither = len(times_neither) / total

    return {
        "prob_goal": prob_goal,
        "prob_fail": prob_fail,
        "prob_neither": prob_neither,
        "mean_time_goal": np.mean(times_goal) if times_goal else float("nan"),
        "std_time_goal": np.std(times_goal) if times_goal else float("nan"),
        "mean_time_fail": np.mean(times_fail) if times_fail else float("nan"),
        "times_goal": times_goal,
        "times_fail": times_fail,
        "sample_trajectories": sample_trajectories,
        "n_sim": n_sim,
    }
