"""
Module donnees_formes : chargement des images simples (PNG, JPG...), en couleur ou non.

Images utilisees (la liste complete est dans main.py, dictionnaire IMAGES) :
    formes/piece.png         piece rectangulaire avec deux trous ronds (TP2, piece.tif recadree)
    formes/engrenage.png     roue dentee seule (TP2, rondeldent.tif recadree)
    formes/dessin_ligne.png  courbe en S avec trois trous (esprit de la figure 5 de l'article)
    formes/labyrinthe.png    labyrinthe (chemin a trouver dans les couloirs blancs)
    Paris/...jpg             vues aeriennes de Paris (routes a suivre, cf. section 7.1 de l'article)

Ce module ne fait aucun affichage.
"""

import numpy as np
from PIL import Image


def charger_image(fichier):
    """Ouvre une image et la renvoie en niveaux de gris, valeurs entre 0 et 1 (float).
    Une image en couleur est convertie en niveaux de gris (meme formule ponderee que
    convert_gris du TP1 : 0.299 R + 0.587 G + 0.114 B)."""
    image = Image.open(fichier).convert("L")          # "L" = niveaux de gris, 0 a 255
    return np.asarray(image, dtype=float) / 255.0


def reduire(image, facteur):
    """Reduit l'image d'un facteur entier : chaque bloc facteur x facteur devient 1 pixel
    (moyenne du bloc). Sert a accelerer le Fast Marching sur les grandes images
    (2 fois moins de pixels en largeur et en hauteur = environ 4 fois plus rapide)."""
    h = image.shape[0] // facteur * facteur
    l = image.shape[1] // facteur * facteur
    blocs = image[:h, :l].reshape(h // facteur, facteur, l // facteur, facteur)
    return blocs.mean(axis=(1, 3))
