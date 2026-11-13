"""
Module pretraitement : lissage de l'image avant le calcul du gradient.

Sujet, question 1.1. Le bruit de l'IRM est granuleux (et non impulsionnel) :
on utilise un filtre lineaire (moyenneur ou gaussien), cf. TP2 exercice 3.

Fonctions :
    filtrer(image, noyau)           applique un noyau quelconque (outil general,
                                    aussi utilise par carte_cout.py pour Sobel)
    lisser_moyenne(image, taille)   filtre moyenneur taille x taille
    noyau_gaussien(sigma)           poids du filtre gaussien
    lisser_gauss(image, sigma)      filtre gaussien
    moyenne_locale(image, taille)   moyenne sur une grande fenetre (rapide, image integrale)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def filtrer(image, noyau):
    """Applique un noyau (3x3, 5x5...) a l'image : pour chaque pixel,
    somme des voisins multiplies par les poids du noyau. Bords par replication."""
    r = noyau.shape[0] // 2
    image_bordee = np.pad(image, r, mode="edge")
    fenetres = sliding_window_view(image_bordee, noyau.shape)
    return (fenetres * noyau).sum(axis=(2, 3))


def lisser_moyenne(image, taille=3):
    """Filtre moyenneur taille x taille : chaque pixel devient la moyenne
    de ses voisins, tous avec le meme poids. Bords par replication."""
    r = taille // 2
    image_bordee = np.pad(image, r, mode="edge")
    fenetres = sliding_window_view(image_bordee, (taille, taille))
    return fenetres.mean(axis=(2, 3))


def noyau_gaussien(sigma):
    """Noyau gaussien 2D de taille (6*sigma+1) x (6*sigma+1), normalise (somme = 1)."""
    r = int(np.ceil(3 * sigma))                   # au-dela de 3 sigma, les poids sont negligeables
    t = np.arange(-r, r + 1)
    yy, xx = np.meshgrid(t, t, indexing="ij")     # coordonnees relatives au centre
    noyau = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return noyau / noyau.sum()                    # somme = 1 : la luminosite moyenne est conservee


def lisser_gauss(image, sigma=1.0):
    """Filtre gaussien : moyenne PONDEREE des voisins, les plus proches
    ayant le plus de poids. Bords par replication."""
    return filtrer(image, noyau_gaussien(sigma))

def moyenne_locale(image, taille):
    """Moyenne de chaque pixel sur une fenetre taille x taille (taille impaire), meme grande.

    filtrer() devient trop lent / trop gourmand en memoire pour une fenetre 21 x 21 sur une
    grande image. On utilise l'astuce de l'IMAGE INTEGRALE : S[i, j] = somme de tous les
    pixels au-dessus et a gauche de (i, j). La somme d'une fenetre se calcule alors avec
    4 valeurs de S seulement, quelle que soit la taille de la fenetre.
    """
    r = taille // 2
    bordee = np.pad(image, r + 1, mode="reflect")
    S = bordee.cumsum(axis=0).cumsum(axis=1)
    h, l = image.shape
    somme = (S[taille:taille + h, taille:taille + l] - S[:h, taille:taille + l]
             - S[taille:taille + h, :l] + S[:h, :l])
    return somme / taille**2
