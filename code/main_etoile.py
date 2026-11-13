"""
Programme : place de l'Etoile vue du ciel (avenues en etoile, cf. section 7.1 de l'article).

Lancer depuis le dossier Projet avec :
    python code/main_etoile.py
(ou depuis VS Code avec le bouton Run). Les figures vont dans figures/etoile/.
"""

import matplotlib.pyplot as plt

import donnees_formes as df
import traitement as tr
from parametres import DOSSIER_DONNEES

# Image : chemin du fichier a partir du dossier donnees/
FICHIER = "Paris/satellite-aerial-photography-daily-overview-benjamin-grant-49-5816f6bfbd137__880.jpg"
# Source : Daily Overview, Benjamin Grant (a citer dans le rapport)


def lancer(afficher=True):
    """Charge l'image et applique le traitement commun.
    afficher=False : les figures sont enregistrees puis fermees (pas de fenetres)."""
    image = df.charger_image(DOSSIER_DONNEES / FICHIER)
    tr.traiter_image(image, "etoile", interactif=afficher)
    if not afficher:
        plt.close("all")


if __name__ == "__main__":
    lancer()
    plt.show()
