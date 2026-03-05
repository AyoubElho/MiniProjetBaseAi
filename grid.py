"""
grid.py — Définition de la grille, obstacles, et utilitaires.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ──────────────────────────────────────────────
# Grilles prédéfinies  (0 = libre, 1 = obstacle)
# ──────────────────────────────────────────────

GRIDS = {
    "facile": {
        "grid": np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]),
        "start": (0, 0),
        "goal":  (4, 4),
    },
    "moyenne": {
        "grid": np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
        "start": (0, 0),
        "goal":  (6, 6),
    },
    "difficile": {
        "grid": np.array([
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        "start": (0, 0),
        "goal":  (9, 9),
    },
}


def get_neighbors(state, grid):
    """Retourne les voisins libres (4-connexité) avec le coût 1."""
    r, c = state
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # haut, bas, gauche, droite
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            neighbors.append(((nr, nc), 1))  # (état, coût)
    return neighbors


def manhattan(p, goal):
    """Heuristique de Manhattan."""
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])


def zero_heuristic(p, goal):
    """Heuristique nulle (équivalente à UCS)."""
    return 0


def reconstruct_path(came_from, start, goal):
    """Reconstitue le chemin depuis came_from."""
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path


def path_to_policy(path):
    """Convertit un chemin en politique : état -> direction."""
    policy = {}
    for i in range(len(path) - 1):
        s = path[i]
        s_next = path[i + 1]
        dr = s_next[0] - s[0]
        dc = s_next[1] - s[1]
        policy[s] = (dr, dc)
    # Le but est absorbant
    policy[path[-1]] = None
    return policy


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────

def visualize_grid(grid, start, goal, path=None, title="Grille", ax=None, pi_n=None):
    """Affiche la grille avec le chemin."""
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    rows, cols = grid.shape
    display = grid.copy().astype(float)

    # Chemin
    if path:
        for (r, c) in path:
            display[r, c] = 2
    # Start / Goal
    display[start[0], start[1]] = 3
    display[goal[0], goal[1]] = 4

    # Palette simple : noir / blanc / gris
    COLORS = {
        "libre":    "#FFFFFF",   # blanc
        "obstacle": "#1A1A1A",   # noir
        "chemin":   "#AAAAAA",   # gris moyen
        "depart":   "#555555",   # gris foncé
        "but":      "#333333",   # gris très foncé
    }
    cmap = ListedColormap([COLORS["libre"], COLORS["obstacle"],
                           COLORS["chemin"], COLORS["depart"], COLORS["but"]])
    ax.imshow(display, cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#CCCCCC", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, pad=8)

    # Annotations S / G
    ax.text(start[1], start[0], "S", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.text(goal[1],  goal[0],  "G", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    patches = [
        mpatches.Patch(facecolor=COLORS["libre"], edgecolor="gray", label="Libre"),
        mpatches.Patch(color=COLORS["obstacle"], label="Obstacle"),
        mpatches.Patch(color=COLORS["chemin"],   label="Chemin"),
        mpatches.Patch(color=COLORS["depart"],   label="Départ"),
        mpatches.Patch(color=COLORS["but"],      label="But"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              framealpha=0.9, edgecolor="#999999")

    if show:
        plt.tight_layout()
        plt.savefig(f"/mnt/user-data/outputs/{title.replace(' ', '_')}.png", dpi=150)
        plt.close()
