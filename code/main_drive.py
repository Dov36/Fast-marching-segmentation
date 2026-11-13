"""
Programme : base DRIVE (vaisseaux de la retine) - suivre un vaisseau entre deux points.

Utilisation : clique un point sur un vaisseau, puis un 2e point plus loin sur le MEME vaisseau.
Le chemin minimal suit le vaisseau ; on mesure ensuite la part du chemin qui reste dans les
vaisseaux traces par l'expert (verite terrain de DRIVE).

Lancer depuis le dossier Projet avec :
    python code/main_drive.py
(ou depuis VS Code avec le bouton Run). Les figures vont dans figures/drive/.
Pour evaluer automatiquement toute la base : python code/evaluer_drive.py
"""

import matplotlib.pyplot as plt

import donnees_drive as dd
import traitement as tr
from parametres import DOSSIER_DONNEES

NUMERO = 21            # image de DRIVE a utiliser (1 a 40)


def lancer(afficher=True):
    """Charge l'image et applique le traitement commun.
    afficher=False : les figures sont enregistrees puis fermees (pas de fenetres)."""
    image, vaisseaux, zone = dd.charger_drive(DOSSIER_DONNEES / "DRIVE", NUMERO)
    nom = f"drive_{dd.nom_image(NUMERO)}"
    tr.traiter_image(image, "drive", nom=nom, titre=f"DRIVE {dd.nom_image(NUMERO)}",
                     interactif=afficher,
                     type_cout="vaisseaux",    # vaisseau = plus sombre que ses voisins
                     w=0.01, alpha=100.0,      # reglages mesures avec evaluer_drive.py
                     verite=vaisseaux,         # vaisseaux de l'expert, pour l'evaluation
                     nb_points=2)              # 2 clics : debut et fin du vaisseau
    if not afficher:
        plt.close("all")


if __name__ == "__main__":
    lancer()
    plt.show()
