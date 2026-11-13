"""
Tests de carte_cout.py (sujet, question 1.2).

Lancer depuis le dossier Projet :
    python code/tests/test_carte_cout.py
Si seul "[ok]" s'affiche, tous les tests sont passes.
"""

import sys
from pathlib import Path

# Permet d'importer les modules du dossier code/ (dossier parent de tests/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import carte_cout as cc

# 1. Image constante : pas de bord, gradient nul partout
constante = np.full((10, 10), 0.5)
gx, gy = cc.gradient(constante)
assert np.allclose(gx, 0) and np.allclose(gy, 0), "image constante : le gradient doit etre nul"

# 2. Rampe horizontale I(y, x) = x : gx = 8 (poids 1+2+1, ecart de 2 colonnes), gy = 0
rampe_x = np.tile(np.arange(10.0), (10, 1))
gx, gy = cc.gradient(rampe_x)
assert np.isclose(gx[5, 5], 8) and np.isclose(gy[5, 5], 0), "rampe en x : attendu gx = 8, gy = 0"

# 3. Rampe verticale I(y, x) = y : gx = 0, gy = 8
gx, gy = cc.gradient(rampe_x.T)
assert np.isclose(gx[5, 5], 0) and np.isclose(gy[5, 5], 8), "rampe en y : attendu gx = 0, gy = 8"

# 4. Bord clair -> sombre (de gauche a droite) : gx negatif (on ne doit pas le couper)
marche = np.ones((10, 10))
marche[:, 5:] = 0.0
gx, _ = cc.gradient(marche)
assert gx[5, 5] < 0, "bord clair -> sombre : gx doit etre negatif"

# 5. Norme : sqrt(3^2 + 4^2) = 5
assert np.isclose(cc.norme_gradient(np.array(3.0), np.array(4.0)), 5.0), "norme fausse"

# 6. Potentiel : toujours > 0, entre w + 1/(1+alpha) et 1 + w
norme = np.random.default_rng(0).random((20, 20)) * 3
w, alpha = 0.1, 10.0
P = cc.potentiel(norme, w, alpha)
assert P.min() > 0, "potentiel : doit etre strictement positif"
assert np.isclose(P.min(), w + 1 / (1 + alpha)), "potentiel : minimum attendu w + 1/(1+alpha)"
assert P.max() <= 1 + w + 1e-12, "potentiel : maximum attendu <= 1 + w"

# 7. Potentiel decroissant : plus le gradient est fort, plus le cout est faible
n = np.array([0.0, 1.0, 2.0, 3.0])
assert np.all(np.diff(cc.potentiel(n, w, alpha)) < 0), "potentiel : doit decroitre avec le gradient"

print("[ok] tests de carte_cout termines")
