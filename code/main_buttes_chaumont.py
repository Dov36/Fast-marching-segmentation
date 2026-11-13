"""
Programme : vue aerienne du parc des Buttes-Chaumont (Paris 19e).

But : tracer le contour du parc (zone sombre de vegetation) avec un contour ferme.
L'image est texturee (immeubles, rues, arbres) : on la lisse un peu (gaussien sigma = 2)
pour que les petits details ne soient pas pris pour des bords.

Utilisation : clique 6 points repartis sur le bord du parc (dans l'ordre, en tournant).
Lancer depuis le dossier Projet avec :
    python code/main_buttes_chaumont.py
(ou depuis VS Code avec le bouton Run). Les figures vont dans figures/buttes_chaumont/.
"""

import matplotlib.pyplot as plt

import donnees_formes as df
import traitement as tr
from parametres import DOSSIER_DONNEES

# Image : chemin du fichier a partir du dossier donnees/
FICHIER = "Paris/ButtesChaumont.png"
REDUCTION = 2          # image 1337 x 767 reduite 2 fois -> 668 x 383 (calcul ~4 fois plus rapide)


def lancer(afficher=True):
    """Charge l'image et applique le traitement commun.
    afficher=False : les figures sont enregistrees puis fermees (pas de fenetres)."""
    image = df.reduire(df.charger_image(DOSSIER_DONNEES / FICHIER), REDUCTION)
    tr.traiter_image(image, "buttes_chaumont", titre="Buttes-Chaumont", interactif=afficher,
                     lissage="gauss", sigma=2.0,   # lissage propre a cette image (texture)
                     nb_points=6)                  # 6 clics autour du parc
    if not afficher:
        plt.close("all")


if __name__ == "__main__":
    lancer()
    plt.show()
