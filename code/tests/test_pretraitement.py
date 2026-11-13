"""
Tests de pretraitement.py (sujet, question 1.1).

Lancer depuis le dossier Projet :
    python code/tests/test_pretraitement.py
Si seul "[ok]" s'affiche, tous les tests sont passes.
"""

import sys
from pathlib import Path

# Permet d'importer les modules du dossier code/ (dossier parent de tests/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import pretraitement as pt

# 1. Une image constante reste constante (les deux filtres)
constante = np.full((20, 20), 0.5)
assert np.allclose(pt.lisser_moyenne(constante), 0.5), "moyenne : image constante modifiee"
assert np.allclose(pt.lisser_gauss(constante, 1.0), 0.5), "gauss : image constante modifiee"

# 2. La taille de l'image est conservee
image = np.random.default_rng(0).random((30, 40))
assert pt.lisser_moyenne(image).shape == (30, 40), "moyenne : taille modifiee"
assert pt.lisser_gauss(image, 1.0).shape == (30, 40), "gauss : taille modifiee"

# 3. Exemple calcule a la main (image 3x3 de 1 a 9)
petite = np.arange(1, 10, dtype=float).reshape(3, 3)
m = pt.lisser_moyenne(petite)
assert np.isclose(m[1, 1], 5.0), "moyenne : pixel central faux (attendu 45/9 = 5)"
assert np.isclose(m[0, 0], 21 / 9), "moyenne : coin faux (attendu 21/9 = 2.33)"

# 4. Le noyau gaussien : taille 7x7 pour sigma = 1, somme 1, maximum au centre
noyau = pt.noyau_gaussien(1.0)
assert noyau.shape == (7, 7), "noyau : taille attendue 7x7"
assert np.isclose(noyau.sum(), 1.0), "noyau : la somme doit valoir 1"
assert noyau[3, 3] == noyau.max(), "noyau : le maximum doit etre au centre"

# 5. Le lissage reduit les variations (bruit)
assert pt.lisser_gauss(image, 1.0).std() < image.std(), "gauss : le bruit n'a pas diminue"

print("[ok] tests de pretraitement termines")

# 6. filtrer avec un noyau "identite" (1 au centre) ne change pas l'image
identite = np.zeros((3, 3))
identite[1, 1] = 1.0
assert np.allclose(pt.filtrer(image, identite), image), "filtrer : le noyau identite doit rendre l'image"
print("[ok] test de filtrer termine")
