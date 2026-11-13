"""
Tests de fast_marching.py (sujet, partie 2 : Q2.1 et Q2.3).

Lancer depuis le dossier Projet :
    python code/tests/test_fast_marching.py
Si seul "[ok]" s'affiche, tous les tests sont passes.
"""

import sys
from pathlib import Path

# Permet d'importer les modules du dossier code/ (dossier parent de tests/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import numpy as np

import fast_marching as fm

# ----------------------------------------------------------------------------
# Q2.1 - mise_a_jour
# ----------------------------------------------------------------------------
assert math.isclose(fm.mise_a_jour(0, 0, 1), 1 / math.sqrt(2)), "(0, 0, 1) : attendu 0.707"
assert math.isclose(fm.mise_a_jour(0, 10, 1), 1), "(0, 10, 1) : attendu 1 (un seul voisin compte)"
assert math.isclose(fm.mise_a_jour(1, 1, 1), 1 + 1 / math.sqrt(2)), "(1, 1, 1) : attendu 1.707"
assert math.isclose(fm.mise_a_jour(3, math.inf, 1), 4), "(3, inf, 1) : attendu 4 (un seul voisin connu)"
assert math.isclose(fm.mise_a_jour(0.2, 0.7, 1), fm.mise_a_jour(0.7, 0.2, 1)), "doit etre symetrique"
u = fm.mise_a_jour(0.2, 0.7, 1)
assert u >= 0.7 and math.isclose((u - 0.2) ** 2 + (u - 0.7) ** 2, 1), "doit resoudre l'equation"

# ----------------------------------------------------------------------------
# Q2.2 - quelques proprietes de fast_marching
# ----------------------------------------------------------------------------
cout = np.random.default_rng(0).uniform(0.2, 1.1, (30, 40))
U = fm.fast_marching(cout, (10, 15))
assert U[10, 15] == 0, "U doit valoir 0 au depart"
assert np.all(np.isfinite(U)), "tous les pixels doivent etre atteints"
assert np.all(U >= 0), "U doit etre positif"

# ----------------------------------------------------------------------------
# Q2.3 - validation : cout constant = 1  ->  U = distance euclidienne au depart
# ----------------------------------------------------------------------------
U = fm.fast_marching(np.ones((101, 101)), (50, 50))
yy, xx = np.indices(U.shape)
distance = np.hypot(yy - 50, xx - 50)
erreur = U - distance
assert np.isclose(U[50, 100], 50), "sur les axes, U doit etre exact"
assert erreur.min() >= -1e-9, "le Fast Marching ne doit pas sous-estimer la distance"
assert erreur.max() < 1.5, f"erreur maximale trop grande : {erreur.max():.2f} pixel"
assert erreur[100, 100] / distance[100, 100] < 0.02, "erreur relative en diagonale > 2 %"

print("[ok] tests de fast_marching termines "
      f"(erreur max sur le cercle : {erreur.max():.2f} pixel, soit "
      f"{100 * erreur[100, 100] / distance[100, 100]:.1f} % dans le coin)")
