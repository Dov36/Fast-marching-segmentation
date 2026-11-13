"""
Module carte_cout : construction de la carte de cout (potentiel) a partir de l'image.

Etape 2 du projet - Sujet, partie 1.
Article : section 2 (eq. 4), section 3.1 (eq. 8-9), section 5.

Fonctions :
    gradient(image)            -> Q1.2 : composantes gx, gy (filtres de Sobel)
    norme_gradient(gx, gy)     -> Q1.2 : force du bord
    potentiel(norme, w, alpha) -> Q1.3 : carte de cout P~ = 1 / (1 + alpha * norme) + w
    potentiel_intensite(image, w, clair) -> cout base sur l'intensite (labyrinthe, routes)
    potentiel_vaisseaux(image, w, alpha) -> cout pour suivre un vaisseau sombre (retine, DRIVE)

Regle : ces fonctions CALCULENT et RENVOIENT des tableaux numpy.
Aucun affichage ici (les figures vont dans affichage.py).
"""

import numpy as np

from pretraitement import filtrer, moyenne_locale

# Noyaux de Sobel
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)   # variations horizontales (bords verticaux)
SOBEL_Y = SOBEL_X.T                             # variations verticales (bords horizontaux)


def gradient(image):
    """Gradient de l'image par les filtres de Sobel.
    Renvoie (gx, gy) : variations horizontale et verticale, avec leur signe
    (pas de clip : un gradient negatif est un vrai bord, de clair vers sombre)."""
    gx = filtrer(image, SOBEL_X)
    gy = filtrer(image, SOBEL_Y)
    return gx, gy


def norme_gradient(gx, gy):
    """Norme du gradient : force du bord, quelle que soit son orientation."""
    return np.sqrt(gx**2 + gy**2)


def potentiel(norme, w, alpha):
    """Carte de cout P~ (equations 4 et 9 de l'article).

    1. On normalise la norme du gradient entre 0 et 1 (division par son maximum),
       pour que alpha ait le meme sens quelle que soit l'image.
    2. P = 1 / (1 + alpha * n) : fonction DECROISSANTE du gradient,
       donc P est faible sur les bords (n proche de 1) et proche de 1 ailleurs (n proche de 0).
    3. P~ = P + w : w > 0 est le cout minimal de chaque pixel (regularisation,
       et indispensable car le Fast Marching avance a la vitesse 1 / P~).

    norme : norme du gradient (tableau 2D, valeurs >= 0)
    w     : constante de regularisation (> 0), de l'ordre de 0.1 dans l'article
    alpha : contraste bord / reste ; un pixel de bord coute environ (1 + alpha) fois moins cher
    Renvoie P~, tableau de meme taille, a valeurs entre w + 1/(1+alpha) et 1 + w.
    """
    n = norme / norme.max()
    P = 1.0 / (1.0 + alpha * n)
    return P + w


def potentiel_intensite(image, w, clair=True):
    """Carte de cout basee sur l'INTENSITE de l'image, et non sur le gradient (article, section 7.1).

    Utile quand on ne cherche pas un bord, mais un chemin A L'INTERIEUR d'une zone :
    les couloirs d'un labyrinthe, une route, une ligne dessinee...

    clair=True  : les zones CLAIRES sont bon marche   -> P = 1 - I  (couloirs blancs)
    clair=False : les zones SOMBRES sont bon marche   -> P = I      (route sombre, trait noir)
    L'image est d'abord normalisee entre 0 et 1. Renvoie P~ = P + w.
    """
    I = (image - image.min()) / (image.max() - image.min())
    P = 1.0 - I if clair else I
    return P + w


def potentiel_vaisseaux(image, w, alpha, taille_fond=21):
    """Carte de cout pour suivre un VAISSEAU sombre (fond d'oeil, base DRIVE).

    Probleme : l'eclairage n'est pas uniforme (centre clair, bords sombres), donc
    "sombre = pas cher" (potentiel_intensite) rend aussi les bords de l'image bon marche.
    Solution : on compare chaque pixel a la MOYENNE DE SES VOISINS (le fond local) :
        v = fond_local - I   (positif si le pixel est plus sombre que ce qui l'entoure)
    Un vaisseau est plus sombre que le fond autour de lui -> v grand -> cout faible.
    Puis meme forme que potentiel() : P~ = 1 / (1 + alpha * v) + w.

    taille_fond : taille de la fenetre du fond local (plus grande que l'epaisseur d'un vaisseau)
    """
    v = np.clip(moyenne_locale(image, taille_fond) - image, 0, None)
    v = v / v.max()
    return 1.0 / (1.0 + alpha * v) + w
