"""
Tests de evaluation.py (sujet, partie 4 : Q4.2).

Lancer depuis le dossier Projet :
    python code/tests/test_evaluation.py
Si seul "[ok]" s'affiche, tous les tests sont passes.
"""

import sys
from pathlib import Path

# Permet d'importer les modules du dossier code/ (dossier parent de tests/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import evaluation as ev

# Dice : cas simples
a = np.zeros((10, 10), bool); a[:, :5] = True        # moitie gauche (50 pixels)
b = np.zeros((10, 10), bool); b[:, 5:] = True        # moitie droite (50 pixels)
assert ev.dice(a, a) == 1.0, "deux zones identiques -> Dice = 1"
assert ev.dice(a, b) == 0.0, "deux zones sans pixel commun -> Dice = 0"
c = np.zeros((10, 10), bool); c[:, :10] = True       # image entiere (100 pixels)
assert np.isclose(ev.dice(a, c), 2 * 50 / 150), "Dice(moitie, tout) = 2*50 / (50+100)"

# masque_interieur : un carre de cote 21 (contour) -> 21 x 21 = 441 pixels remplis
cote = [(10, x) for x in range(10, 31)] + [(y, 30) for y in range(11, 31)] \
     + [(30, x) for x in range(29, 9, -1)] + [(y, 10) for y in range(29, 10, -1)]
masque = ev.masque_interieur(np.array(cote), (50, 50))
assert masque.sum() == 441, f"carre 21 x 21 : attendu 441 pixels, obtenu {masque.sum()}"
assert masque[20, 20] and not masque[5, 5], "le centre est dedans, le coin de l'image dehors"

# masque_interieur : un disque de rayon 15 -> Dice tres proche de 1 avec le vrai disque
yy, xx = np.indices((60, 60))
angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
contour = np.unique(np.round(np.c_[30 + 15 * np.sin(angles), 30 + 15 * np.cos(angles)]).astype(int), axis=0)
contour = contour[np.argsort(np.arctan2(contour[:, 0] - 30, contour[:, 1] - 30))]   # dans l'ordre
disque = np.hypot(yy - 30, xx - 30) <= 15
score = ev.dice(ev.masque_interieur(contour, (60, 60)), disque)
assert score > 0.95, f"disque : Dice trop faible ({score:.3f})"

# f1_chemin : chemin identique -> F1 = 1 ; chemin qui ne fait que la moitie -> rappel ~ 0.5
ligne = np.array([(10, x) for x in range(10, 50)])
assert ev.f1_chemin(ligne, ligne)[2] == 1.0, "chemin identique a la reference -> F1 = 1"
precision, rappel, f1 = ev.f1_chemin(ligne[:20], ligne)
assert precision == 1.0 and 0.5 <= rappel <= 0.6, "moitie du chemin : precision 1, rappel ~ 0.5"

# chemin_reference : reste dans la zone de l'expert
zone = np.zeros((30, 60), bool); zone[5:8, 5:55] = True; zone[5:25, 50:55] = True
ref = ev.chemin_reference(zone, (6, 6), (24, 52))
assert zone[ref[:, 0], ref[:, 1]].all(), "le chemin de reference doit rester dans la zone"

print(f"[ok] tests de evaluation termines (disque : Dice = {score:.3f})")
