# Planification Robuste sur Grille : A\* + Chaînes de Markov

<div align="center">

| | |
|---|---|
| **Auteur** | Ayoub El Houani |
| **Filière** | SDIA |
| **Encadrant** | Prof. M. Mohamed Mestari |
| **Date** | Mars 2026 |

</div>

---

## Description

Mini-projet combinant la **recherche heuristique A\*** et les **Chaînes de Markov à temps discret** pour résoudre le problème de planification robuste d'un agent mobile sur une grille 2D soumise à une dynamique stochastique.

> **Question centrale :** Comment planifier un chemin peu coûteux (A\*) tout en tenant compte d'une dynamique stochastique (Markov) et en évaluant la robustesse du plan par simulation et mesures probabilistes ?

---

## Structure du projet

```
.
├── grid.py            # Grilles 2D, voisinage 4-connexe, heuristiques, visualisation
├── astar.py           # A*, UCS, Greedy, Weighted A*, vérif. admissibilité/cohérence
├── markov.py          # Construction de P, π^(n), absorption N=(I-Q)^-1, Monte-Carlo
├── experiments.py     # 4 expériences reproductibles + figures
├── main.py            # Script principal — lance toutes les expériences
```

---

## Algorithmes implémentés

### Recherche heuristique (`astar.py`)

| Algorithme | f(n) | Optimal | Description |
|---|---|---|---|
| **UCS** | g(n) | ✅ | Coût uniforme, exploration exhaustive |
| **Greedy** | h(n) | ❌ | Rapide mais non optimal |
| **A\*** | g(n) + h(n) | ✅ | Optimal si h admissible |
| **Weighted A\*** | g(n) + w·h(n) | ⚠️ w-suboptimal | Compromis vitesse/optimalité |

**Heuristique de Manhattan :**
```
h(x, y) = |x - x_goal| + |y - y_goal|
```
Vérifiée **admissible** et **cohérente** sur les 3 grilles.

### Chaîne de Markov (`markov.py`)

Modèle d'incertitude avec paramètre **ε** :
- Action voulue : probabilité `1 - ε`
- Déviation latérale gauche : probabilité `ε/2`
- Déviation latérale droite : probabilité `ε/2`

**Analyse d'absorption :**
```
N = (I - Q)^(-1)    # Matrice fondamentale
B = N · R           # Probabilités d'absorption (GOAL / FAIL)
t = N · 1           # Temps moyen d'absorption
```

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/AyoubElho/MiniProjetBaseAi
cd astar-markov-grid

# Installer les dépendances
pip install numpy matplotlib pandas
```

---

## Utilisation

### Lancer toutes les expériences

```bash
python main.py
```

### Utiliser les modules individuellement

```python
from grid import GRIDS, manhattan
from astar import astar, ucs, greedy
from markov import build_transition_matrix, simulate_trajectories

# Planification A*
cfg = GRIDS['moyenne']
result = astar(cfg['grid'], cfg['start'], cfg['goal'], manhattan)
print(f"Chemin : {result['path']}")
print(f"Coût   : {result['cost']}")
print(f"Nœuds  : {result['nodes_expanded']}")

# Chaîne de Markov (ε = 0.1)
P, states, idx, GOAL, FAIL = build_transition_matrix(
    cfg['grid'], result['path'], epsilon=0.1, goal_state=cfg['goal']
)

# Simulation Monte-Carlo
sim = simulate_trajectories(P, idx[cfg['start']], GOAL, FAIL, n_sim=5000)
print(f"P(GOAL) = {sim['prob_goal']:.4f}")
print(f"P(FAIL) = {sim['prob_fail']:.4f}")
```

---

## Expériences et résultats

### E.1 — Comparaison UCS / Greedy / A\*

| Grille | Algo | Coût | Nœuds | Temps (ms) |
|---|---|---|---|---|
| Facile | UCS | 8 | 21 | 0.20 |
| Facile | Greedy | 8 | 9 | 0.07 |
| Facile | A\* | 8 | 21 | 0.15 |
| Moyenne | UCS | 12 | 30 | 0.31 |
| Moyenne | Greedy | **16** ❌ | 17 | 0.14 |
| Moyenne | A\* | 12 | 22 | 0.24 |
| Difficile | UCS | 18 | 48 | 0.45 |
| Difficile | Greedy | **20** ❌ | 21 | 0.16 |
| Difficile | A\* | 18 | 41 | 0.39 |

### E.2 — Impact du niveau d'incertitude ε

| ε | Coût A\* | P(GOAL) théo. | P(GOAL) MC | P(FAIL) |
|---|---|---|---|---|
| 0.0 | 12 | 1.0000 | 1.0000 | 0.0000 |
| 0.1 | 12 | 0.3415 | 0.3414 | 0.6585 |
| 0.2 | 12 | 0.0982 | 0.0996 | 0.9018 |
| 0.3 | 12 | 0.0228 | 0.0274 | 0.9772 |

> ⚠️ Avec ε = 0.1, la probabilité d'atteindre le but chute de **100%** à **34%**

### E.3 — Heuristiques h=0 vs Manhattan

| Grille | Nœuds h=0 | Nœuds Manhattan | Réduction |
|---|---|---|---|
| Facile | 21 | 21 | 0.0% |
| Moyenne | 30 | 22 | **26.7%** |
| Difficile | 48 | 41 | **14.6%** |

### E.4 — Weighted A\* (grille difficile)

| w | Coût | Nœuds | Suboptimalité |
|---|---|---|---|
| 1.0 | 18 | 41 | 0.00% |
| 1.5 | 18 | 24 | 0.00% ✅ |
| 2.0 | 20 | 21 | 11.11% |
| 3.0 | 20 | 21 | 11.11% |

> w = 1.5 préserve l'optimalité avec **41% moins de nœuds**

---

## Analyse Markov — Résultats d'absorption

| Grille | ε | P(GOAL) théo. | P(GOAL) MC | μ\_abs (th.) | μ\_GOAL (MC) |
|---|---|---|---|---|---|
| Facile | 0.10 | 0.5234 | 0.5244 | 6.47 | 8.42 |
| Moyenne | 0.20 | 0.0982 | 0.1021 | 5.52 | 12.70 |
| Difficile | 0.15 | 0.1023 | 0.1065 | 7.58 | 19.27 |

Concordance théorie / Monte-Carlo : **< 0.5%** sur toutes les configurations.

---

## Grilles

```
Facile (5×5)        Moyenne (7×7)         Difficile (10×10)
S . . . .           S . . # . . .         S . . . # . . . . .
. # # . .           . # . # . # .         . # # . # . # # # .
. . . . .           . # . . . # .         . # . . # . . . # .
. # . # .           . # # # . # .         . # . # # # # . # .
. . . . G           . . . . . # .         . . . . . . # . . .
                    # # . # # # .         . # # # # . # # # .
                    . . . . . . G         . . . . # . . . # .
                                          # # . # # # # . # .
                                          . . . . . . # . . .
                                          . . . . . . . . . G
```

---

## Dépendances

```
numpy >= 1.21
matplotlib >= 3.4
pandas >= 1.3
python >= 3.8
```

---

## Rapport

Le rapport complet est disponible dans [`rapport_final.pdf`](rapport_final.pdf).

Il couvre :
- Modélisation mathématique (f=g+h, matrice P, N=(I-Q)⁻¹)
- Preuves d'admissibilité et cohérence de Manhattan
- Résultats des 4 expériences avec figures et tableaux
- Analyse des classes de communication et absorption
- Discussion, limites et perspectives

---

## Licence

Projet académique — ENSET Mohammedia, 2026.
