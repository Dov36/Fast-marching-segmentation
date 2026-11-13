"""
Tests de chemin.py (sujet, partie 3 : Q3.1 et Q3.3).

Lancer depuis le dossier Projet :
    python code/tests/test_chemin.py
Si seul "[ok]" s'affiche, tous les tests sont passes.
"""

import sys
from pathlib import Path

# Permet d'importer les modules du dossier code/ (dossier parent de tests/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import chemin as ch

# Q3.1 - cout constant : le chemin entre deux points d'une meme ligne est tout droit
chemin, U = ch.chemin_minimal(np.ones((40, 60)), (20, 5), (20, 50))
assert tuple(chemin[0]) == (20, 5) and tuple(chemin[-1]) == (20, 50), "le chemin doit aller de p0 a p1"
assert np.all(chemin[:, 0] == 20), "a cout constant, le chemin doit etre une ligne droite"
pas = np.abs(np.diff(chemin, axis=0)).max(axis=1)
assert np.all(pas == 1), "chaque pas doit aller sur un pixel voisin"

# Q3.3 - cercle : cout faible sur un cercle de rayon 30, contour ferme avec 3 points
yy, xx = np.indices((100, 100))
distance_au_cercle = np.abs(np.hypot(yy - 50, xx - 50) - 30)
cout = np.where(distance_au_cercle < 1, 0.1, 1.0)
points = [(50, 20), (20, 50), (50, 80)]
contour, cartes = ch.contour_ferme(cout, points)
erreur = np.abs(np.hypot(contour[:, 0] - 50, contour[:, 1] - 50) - 30)
assert len(cartes) == 3, "3 points -> 3 morceaux"
assert erreur.max() < 1.5, f"le contour doit suivre le cercle (ecart max {erreur.max():.2f} pixel)"
assert len(contour) > 2 * np.pi * 30 * 0.8, "le contour doit faire tout le tour du cercle"

masque = ch.image_contour(contour, cout.shape)
assert masque.sum() == len(np.unique(contour, axis=0)), "image_contour : un pixel blanc par point"

print(f"[ok] tests de chemin termines (cercle : {len(contour)} pixels, "
      f"ecart max au cercle {erreur.max():.2f} pixel)")
