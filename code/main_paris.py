"""
Programme : vue satellite de Paris (suivi de rues, cf. section 7.1 de l'article).

Lancer depuis le dossier Projet avec :
    python code/main_paris.py
(ou depuis VS Code avec le bouton Run). Les figures vont dans figures/paris/.
"""

import matplotlib.pyplot as plt

import donnees_formes as df
import traitement as tr
from parametres import DOSSIER_DONNEES

# Image : chemin du fichier a partir du dossier donnees/
FICHIER = "Paris/f87fabc380ca314426bcdc6a9f98486b.jpg"


def lancer(afficher=True):
    """Charge l'image et applique le traitement commun.
    afficher=False : les figures sont enregistrees puis fermees (pas de fenetres)."""
    image = df.charger_image(DOSSIER_DONNEES / FICHIER)
    tr.traiter_image(image, "paris", interactif=afficher)
    if not afficher:
        plt.close("all")


if __name__ == "__main__":
    lancer()
    plt.show()
