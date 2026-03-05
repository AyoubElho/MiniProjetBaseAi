"""
main.py — Lance toutes les expériences du mini-projet.
Exécuter : python main.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

# ── Backend matplotlib non-interactif (compatible serveur & local) ──
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  Mini-projet : A* + Chaînes de Markov sur grille 2D")
print("=" * 60)

from experiments import (
    experiment1, experiment2, experiment3,
    markov_analysis, experiment4_weighted_astar,
)

df1        = experiment1()
df2, _     = experiment2("moyenne")
df3        = experiment3()

print("\n── Analyse Markov grille facile ──")
markov_analysis("facile",    epsilon=0.10)
print("\n── Analyse Markov grille moyenne ──")
markov_analysis("moyenne",   epsilon=0.20)
print("\n── Analyse Markov grille difficile ──")
markov_analysis("difficile", epsilon=0.15)

df4 = experiment4_weighted_astar("difficile")

print("\n" + "=" * 60)
print("  Terminé ! Figures dans le dossier outputs/")
print("=" * 60)
