"""
Module donnees_drive : chargement des images de la base DRIVE (vaisseaux de la retine).

Dossier attendu : donnees/DRIVE/ avec images/, manual/ et mask/ (voir donnees/DRIVE/LISEZMOI.txt).

Ce module ne fait aucun affichage.
"""

import numpy as np
from PIL import Image


def nom_image(numero):
    """Nom de l'image : 01 a 20 -> base de test, 21 a 40 -> base d'entrainement."""
    return f"{numero:02d}_test" if numero <= 20 else f"{numero:02d}_training"


def charger_drive(dossier, numero):
    """Charge l'image numero 'numero' (1 a 40) de DRIVE.

    Renvoie (image, vaisseaux, zone) :
        image     : CANAL VERT de la photo, entre 0 et 1. C'est le canal ou les vaisseaux
                    sont le plus contrastes (le rouge est sature, le bleu est bruite).
                    Hors de la retine, on remplace le noir par la valeur moyenne, sinon
                    le bord du disque serait vu comme un enorme "vaisseau" sombre.
        vaisseaux : verite terrain (True = vaisseau trace par l'expert)
        zone      : True dans le disque de la retine
    """
    couleur = np.asarray(Image.open(dossier / "images" / f"{nom_image(numero)}.tif"), dtype=float)
    vert = couleur[..., 1] / 255.0
    vaisseaux = np.asarray(Image.open(dossier / "manual" / f"{numero:02d}_manual1.gif")) > 0
    zone = np.asarray(Image.open(dossier / "mask" / f"{numero:02d}_mask.gif")) > 0
    image = np.where(zone, vert, vert[zone].mean())
    return image, vaisseaux, zone
