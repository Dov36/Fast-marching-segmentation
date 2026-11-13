"""
Programme : courbe en S avec trois trous (dans l'esprit de la figure 5 de l'article).

Lancer depuis le dossier Projet avec :
    python code/main_dessin.py
(ou depuis VS Code avec le bouton Run). Les figures vont dans figures/dessin/.
"""

import matplotlib.pyplot as plt

import donnees_formes as df
import traitement as tr
from parametres import DOSSIER_DONNEES

# Image : chemin du fichier a partir du dossier donnees/
FICHIER = "formes/dessin_ligne.png"


def lancer(afficher=True):
    """Charge l'image et applique le traitement commun.
    afficher=False : les figures sont enregistrees puis fermees (pas de fenetres)."""
    image = df.charger_image(DOSSIER_DONNEES / FICHIER)
    tr.traiter_image(image, "dessin", interactif=afficher)
    if not afficher:
        plt.close("all")


if __name__ == "__main__":
    lancer()
    plt.show()
