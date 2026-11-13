"""
Programme : labyrinthe (trouver le chemin le plus court entre l'entree et la sortie).

Difference avec les autres images : on ne cherche pas un BORD, mais un chemin qui
reste DANS les couloirs blancs. On utilise donc un cout base sur l'intensite
(couloir blanc = pas cher, mur rouge = cher), cf. section 7 de l'article.

Utilisation : clique l'entree (en bas a gauche), puis la sortie (a droite).
Lancer depuis le dossier Projet avec :
    python code/main_labyrinthe.py
(ou depuis VS Code avec le bouton Run). Les figures vont dans figures/labyrinthe/.
"""

import matplotlib.pyplot as plt

import donnees_formes as df
import traitement as tr
from parametres import DOSSIER_DONNEES

# Image : chemin du fichier a partir du dossier donnees/
FICHIER = "formes/labyrinthe.png"
REDUCTION = 4          # image 1300 x 1300 reduite 4 fois -> 325 x 325 (calcul ~16 fois plus rapide)


def lancer(afficher=True):
    """Charge l'image et applique le traitement commun.
    afficher=False : les figures sont enregistrees puis fermees (pas de fenetres)."""
    image = df.reduire(df.charger_image(DOSSIER_DONNEES / FICHIER), REDUCTION)
    tr.traiter_image(image, "labyrinthe", interactif=afficher,
                     type_cout="clair",    # couloirs blancs = bon marche
                     nb_points=2)          # 2 clics : entree et sortie
    if not afficher:
        plt.close("all")


if __name__ == "__main__":
    lancer()
    plt.show()
