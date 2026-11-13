"""
Module parametres : tous les reglages du projet, en un seul endroit.

Ils sont utilises par main_paris.py, main_formes.py, main_irm.py et traitement.py.
Pour changer un reglage pour toutes les images, c'est ici.
"""

import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Dossiers
# ----------------------------------------------------------------------------
DOSSIER_PROJET = Path(__file__).resolve().parent.parent   # ce fichier est dans Projet/code/
DOSSIER_DONNEES = DOSSIER_PROJET / "donnees"              # patient001/, formes/, Paris/
DOSSIER_FIGURES = DOSSIER_PROJET / "figures"              # un sous-dossier par image

# ----------------------------------------------------------------------------
# Etape 2 : lissage et carte de cout
# ----------------------------------------------------------------------------
TAILLE_MOYENNE = 3      # taille du filtre moyenneur (3 -> fenetre 3x3)
SIGMA = 1.0             # ecart-type du filtre gaussien (en pixels) ; plus grand = lissage plus fort
LISSAGE = "aucun"       # filtre utilise pour la suite : "gauss", "moyenne" ou "aucun"
W = 0.001               # regularisation (cout minimal par pixel) ; petit = colle au bord, grand = lisse (article : w ~ 0.1)
ALPHA = 100.0           # contraste bord / reste : un pixel de bord coute ~(1 + ALPHA) fois moins

# ----------------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------------
# False : seules l'IMAGE et la CARTE DE COUT s'affichent a l'ecran.
#         Les autres figures (lissages, gradient, comparaisons...) sont quand meme
#         calculees et ENREGISTREES dans figures/<image>/, mais sans ouvrir de fenetre.
# True  : toutes les figures s'affichent.
# Depuis le terminal, on peut aussi ajouter --detail :  python code/main_piece.py --detail
FIGURES_DETAILLEES = False
if "--detail" in sys.argv:
    FIGURES_DETAILLEES = True

# True : quand on lance un main_xxx.py seul, une fenetre interactive s'ouvre a la fin :
#        on clique des points sur un bord et le contour est trace (module interactif.py).
#        (Desactive automatiquement quand on lance main.py, qui traite toutes les images.)
INTERACTIF = True
NB_POINTS = 3           # nombre de clics avant de lancer le calcul (2 = chemin ouvert, 3 ou + = contour ferme)

TOUT = (slice(None), slice(None))   # zoom = image entiere
