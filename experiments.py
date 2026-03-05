"""
experiments.py — Expériences E.1 à E.4 + analyse Markov.
Compatible Windows / Linux / macOS.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Assure que les autres modules du même dossier sont trouvables ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid   import GRIDS, manhattan, zero_heuristic, visualize_grid
from astar  import astar, ucs, greedy, astar_zero, check_admissibility, check_consistency
from markov import (build_transition_matrix, compute_pi_n, prob_reach_goal,
                    absorption_analysis, simulate_trajectories)

# Dossier de sortie des figures (même dossier que ce script)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
OUT = OUT + os.sep


# ══════════════════════════════════════════════════════════════════
# E.1 — Comparaison UCS / Greedy / A* sur 3 grilles
# ══════════════════════════════════════════════════════════════════

def experiment1():
    print("\n" + "="*60)
    print("E.1 — Comparaison UCS / Greedy / A* sur 3 grilles")
    print("="*60)

    records = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for row_i, gname in enumerate(["facile", "moyenne", "difficile"]):
        cfg  = GRIDS[gname]
        grid, start, goal = cfg["grid"], cfg["start"], cfg["goal"]

        algos = [
            ucs(grid, start, goal),
            greedy(grid, start, goal, manhattan),
            astar(grid, start, goal, manhattan),
        ]

        for col_i, res in enumerate(algos):
            ax    = axes[row_i, col_i]
            title = (f"{gname.capitalize()} — {res['name']}\n"
                     f"Coût={res['cost']}, Nœuds={res['nodes_expanded']}")
            visualize_grid(grid, start, goal, res["path"], title=title, ax=ax)

            records.append({
                "Grille":           gname,
                "Algorithme":       res["name"],
                "Coût chemin":      res["cost"],
                "Nœuds développés": res["nodes_expanded"],
                "OPEN max":         res["open_size_max"],
                "Temps (ms)":       round(res["time_s"] * 1000, 3),
                "Mémoire (KB)":     round(res["memory_kb"], 2),
                "Trouvé":           res["found"],
            })

    plt.suptitle("E.1 — Comparaison des algorithmes de recherche",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "E1_comparaison_algos.png"), dpi=150, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════
# E.2 — Impact de ε sur la robustesse du plan
# ══════════════════════════════════════════════════════════════════

def experiment2(grid_name="moyenne"):
    print("\n" + "="*60)
    print(f"E.2 — Impact de ε sur la robustesse du plan (grille {grid_name})")
    print("="*60)

    cfg  = GRIDS[grid_name]
    grid, start, goal = cfg["grid"], cfg["start"], cfg["goal"]

    res = astar(grid, start, goal, manhattan)
    if not res["found"]:
        print("Aucun chemin trouvé !")
        return None, None

    path = res["path"]
    print(f"Chemin A* : {path}")
    print(f"Coût prévu : {res['cost']}")

    epsilons = [0.0, 0.1, 0.2, 0.3]
    N_STEPS  = 60
    linestyles = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    records   = []

    for eps, ls in zip(epsilons, linestyles):
        P, states, idx, GOAL, FAIL = build_transition_matrix(grid, path, eps, goal)
        start_i = idx[start]

        pi_hist = compute_pi_n(P, start_i, N_STEPS)
        p_goal  = prob_reach_goal(pi_hist, GOAL)
        p_fail  = pi_hist[:, FAIL]

        sim = simulate_trajectories(P, start_i, GOAL, FAIL, n_sim=5000)

        axes[0].plot(p_goal, color="black", linestyle=ls, linewidth=1.8,
                     label=f"ε={eps} (théo)")
        axes[0].axhline(sim["prob_goal"], color="black", linestyle=ls,
                        alpha=0.35, linewidth=0.8)
        axes[1].plot(p_fail, color="black", linestyle=ls, linewidth=1.8,
                     label=f"ε={eps}")

        records.append({
            "ε":                  eps,
            "Coût prévu A*":      res["cost"],
            "P(GOAL) théo n=60":  round(float(p_goal[-1]), 4),
            "P(GOAL) MC":         round(sim["prob_goal"], 4),
            "P(FAIL) théo n=60":  round(float(p_fail[-1]), 4),
            "P(FAIL) MC":         round(sim["prob_fail"], 4),
            "Temps moyen GOAL":   round(sim["mean_time_goal"], 2)
                                  if not np.isnan(sim["mean_time_goal"]) else "N/A",
        })

    for ax, title in zip(axes,
            ["P(GOAL) — Théorique (trait) vs MC (pointillé)",
             "P(FAIL) selon ε"]):
        ax.set_xlabel("Pas de temps n")
        ax.set_ylabel("Probabilité")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f"E.2 — Impact de ε (grille {grid_name})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "E2_epsilon_impact.png"), dpi=150, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    return df, res


# ══════════════════════════════════════════════════════════════════
# E.3 — Heuristiques : h=0 vs Manhattan
# ══════════════════════════════════════════════════════════════════

def experiment3():
    print("\n" + "="*60)
    print("E.3 — Comparaison heuristiques : h=0 vs Manhattan")
    print("="*60)

    records = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    for row_i, gname in enumerate(["facile", "moyenne", "difficile"]):
        cfg  = GRIDS[gname]
        grid, start, goal = cfg["grid"], cfg["start"], cfg["goal"]

        res_h0 = astar_zero(grid, start, goal)
        res_mh = astar(grid, start, goal, manhattan)

        for col_i, res in enumerate([res_h0, res_mh]):
            ax    = axes[row_i, col_i]
            title = (f"{gname.capitalize()} — {res['name']}\n"
                     f"Nœuds={res['nodes_expanded']}, Coût={res['cost']}")
            visualize_grid(grid, start, goal, res["path"], title=title, ax=ax)

        records.append({
            "Grille":          gname,
            "Nœuds A*(h=0)":   res_h0["nodes_expanded"],
            "Nœuds Manhattan": res_mh["nodes_expanded"],
            "Réduction (%)":   round((1 - res_mh["nodes_expanded"] /
                                      res_h0["nodes_expanded"]) * 100, 1),
            "Coût h=0":        res_h0["cost"],
            "Coût Manhattan":  res_mh["cost"],
        })

    plt.suptitle("E.3 — Comparaison h=0 vs Manhattan",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "E3_heuristiques.png"), dpi=150, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    print("\n── Admissibilité et cohérence de Manhattan ──")
    for gname in ["facile", "moyenne", "difficile"]:
        cfg  = GRIDS[gname]
        grid, goal = cfg["grid"], cfg["goal"]
        adm, _ = check_admissibility(grid, goal, manhattan)
        coh, _ = check_consistency(grid, manhattan, goal)
        print(f"  {gname:10s} | Admissible: {adm} | Cohérente: {coh}")

    return df


# ══════════════════════════════════════════════════════════════════
# E.4 — Weighted A*
# ══════════════════════════════════════════════════════════════════

def experiment4_weighted_astar(grid_name="difficile"):
    print("\n" + "="*60)
    print(f"E.4 — Weighted A* (grille {grid_name})")
    print("="*60)

    from astar import search as asearch
    cfg  = GRIDS[grid_name]
    grid, start, goal = cfg["grid"], cfg["start"], cfg["goal"]

    weights = [1.0, 1.5, 2.0, 3.0, 5.0]
    records = []

    fig, axes = plt.subplots(1, len(weights), figsize=(4 * len(weights), 4))
    for i, w in enumerate(weights):
        def weighted_h(p, g, w=w):
            return w * manhattan(p, g)
        res = asearch(grid, start, goal, weighted_h,
                      use_g=True, use_h=True, name=f"WA*(w={w})")
        visualize_grid(grid, start, goal, res["path"],
                       title=f"w={w}\nCoût={res['cost']}\nNœuds={res['nodes_expanded']}",
                       ax=axes[i])
        records.append({
            "w":       w,
            "Coût":    res["cost"],
            "Nœuds":   res["nodes_expanded"],
            "Temps (ms)": round(res["time_s"] * 1000, 3),
        })

    opt = records[0]["Coût"]
    for r in records:
        r["Suboptimalité (%)"] = round((r["Coût"] - opt) / opt * 100, 2) if opt else 0

    plt.suptitle(f"E.4 — Weighted A* (grille {grid_name})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "E4_weighted_astar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════
# Analyse Markov complète
# ══════════════════════════════════════════════════════════════════

def markov_analysis(grid_name="moyenne", epsilon=0.15):
    print("\n" + "="*60)
    print(f"Analyse Markov — grille {grid_name}, ε={epsilon}")
    print("="*60)

    cfg  = GRIDS[grid_name]
    grid, start, goal = cfg["grid"], cfg["start"], cfg["goal"]

    res  = astar(grid, start, goal, manhattan)
    path = res["path"]

    P, states, idx, GOAL, FAIL = build_transition_matrix(grid, path, epsilon, goal)
    start_i = idx[start]

    print(f"États transients : {len(states)}   |   Taille P : {P.shape}")
    print(f"P stochastique   : {np.allclose(P.sum(axis=1), 1.0)}")

    ab = absorption_analysis(P, states, idx, GOAL, FAIL)
    if ab["absorbing_probs"] is not None:
        si = states.index(start)
        print(f"\nDepuis s0={start} :")
        print(f"  P(GOAL) = {ab['absorbing_probs'][si, 0]:.4f}")
        print(f"  P(FAIL) = {ab['absorbing_probs'][si, 1]:.4f}")
        print(f"  Temps moyen d'absorption = {ab['mean_absorption_times'][si]:.2f} étapes")

    sim = simulate_trajectories(P, start_i, GOAL, FAIL, n_sim=10000)
    print(f"\nMonte-Carlo (N=10000) :")
    print(f"  P(GOAL) = {sim['prob_goal']:.4f}")
    print(f"  P(FAIL) = {sim['prob_fail']:.4f}")
    print(f"  Temps moyen (GOAL) = {sim['mean_time_goal']:.2f} ± {sim['std_time_goal']:.2f}")

    # Figure
    N_STEPS = 80
    pi_hist = compute_pi_n(P, start_i, N_STEPS)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    visualize_grid(grid, start, goal, path,
                   title=f"Chemin A* — {grid_name}\nε={epsilon}", ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    n_show = min(15, P.shape[0])
    im = ax2.imshow(P[:n_show, :n_show], cmap="Greys", aspect="auto")
    ax2.set_title(f"Matrice P (15 premiers états)\nε={epsilon}")
    ax2.set_xlabel("j"); ax2.set_ylabel("i")
    plt.colorbar(im, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(pi_hist[:, GOAL],    color="black",  linewidth=2,   label="P(GOAL)")
    ax3.plot(pi_hist[:, FAIL],    color="gray",   linewidth=2,   label="P(FAIL)")
    ax3.plot(pi_hist[:, start_i], color="black",  linewidth=1.5,
             linestyle="--", label="P(s0)")
    ax3.set_xlabel("n"); ax3.set_ylabel("Probabilité")
    ax3.set_title(f"Évolution π^(n) — ε={epsilon}")
    ax3.legend(); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    if sim["times_goal"]:
        ax4.hist(sim["times_goal"], bins=30, color="white",
                 edgecolor="black", linewidth=0.8)
        ax4.axvline(sim["mean_time_goal"], color="black", linestyle="--",
                    label=f"μ={sim['mean_time_goal']:.1f}")
        ax4.set_title("Temps d'atteinte GOAL")
        ax4.set_xlabel("Étapes"); ax4.set_ylabel("Fréquence"); ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    if sim["times_fail"]:
        ax5.hist(sim["times_fail"], bins=20, color="white",
                 edgecolor="black", linewidth=0.8, hatch="///")
        ax5.axvline(sim["mean_time_fail"], color="black", linestyle="--",
                    label=f"μ={sim['mean_time_fail']:.1f}")
        ax5.set_title("Temps d'atteinte FAIL")
        ax5.set_xlabel("Étapes"); ax5.set_ylabel("Fréquence"); ax5.legend()

    ax6 = fig.add_subplot(gs[1, 2])
    p_goal_th = float(pi_hist[-1, GOAL])
    p_fail_th = float(pi_hist[-1, FAIL])
    x = np.arange(2); bw = 0.3
    ax6.bar(x - bw/2, [p_goal_th, p_fail_th], bw,
            label="Théorique P^n", color="white", edgecolor="black")
    ax6.bar(x + bw/2, [sim["prob_goal"], sim["prob_fail"]], bw,
            label="Monte-Carlo", color="gray", edgecolor="black", alpha=0.8)
    ax6.set_xticks(x); ax6.set_xticklabels(["P(GOAL)", "P(FAIL)"])
    ax6.set_title("Théorique vs Monte-Carlo")
    ax6.set_ylabel("Probabilité"); ax6.legend(); ax6.grid(alpha=0.3, axis="y")

    plt.suptitle(f"Analyse Markov — {grid_name}, ε={epsilon}",
                 fontsize=13, fontweight="bold")
    fname = os.path.join(OUT, f"Markov_{grid_name}_eps{int(epsilon*100)}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Figure sauvegardée : {fname}")

    return P, states, idx, GOAL, FAIL, sim, ab
