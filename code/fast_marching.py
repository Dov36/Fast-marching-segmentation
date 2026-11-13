"""
Module fast_marching : calcul de la carte d'action minimale U (equation eikonale).

Etape 3 du projet - Sujet, partie 2.
Article : section 3.3, section 4.4 (eq. 16, 18-19).

Idee : U(x) = cout du chemin le moins cher entre le point de depart et le pixel x.
On peut le voir comme l'heure d'arrivee d'un "feu" allume au depart, qui avance
vite la ou le cout est faible (les bords) et lentement ailleurs.
U verifie l'equation eikonale  ||grad U|| = P~  avec U(depart) = 0.

Fonctions :
    mise_a_jour(a, b, p)            -> Q2.1 : valeur d'un pixel a partir de ses voisins
    fast_marching(cout, depart)     -> Q2.2 : carte U complete

Regle : ces fonctions CALCULENT et RENVOIENT des tableaux numpy.
Aucun affichage ici (les figures vont dans affichage.py).
"""

import heapq
import math

import numpy as np

# Les trois etats possibles d'un pixel pendant la propagation
LOIN = 0    # pas encore atteint par le front (U = infini)
ESSAI = 1   # au bord du front : valeur provisoire, peut encore diminuer
FIGE = 2    # atteint par le front : valeur definitive

# Les 4 voisins d'un pixel : haut, bas, gauche, droite (dy, dx)
VOISINS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def mise_a_jour(a, b, p):
    """Valeur u d'un pixel, calculee a partir de ses voisins deja figes (Q2.1).

    a : plus petite valeur U parmi les voisins gauche et droite (infini si aucun n'est fige)
    b : plus petite valeur U parmi les voisins haut et bas    (infini si aucun n'est fige)
    p : cout P~ du pixel

    On discretise ||grad U|| = p :  (u - a)^2 + (u - b)^2 = p^2   (eq. 18-19 de l'article).

    - Si |a - b| >= p : un seul voisin compte vraiment (l'autre est trop loin ou pas encore
      atteint) -> le front arrive d'une seule direction : u = min(a, b) + p
      (c'est exactement la regle de Dijkstra).
    - Sinon : le front arrive des deux directions a la fois -> on resout l'equation du
      second degre et on garde la PLUS GRANDE racine (u doit etre >= a et >= b, car le
      front s'eloigne du depart) :
          u = (a + b + sqrt(2 p^2 - (a - b)^2)) / 2
    """
    if abs(a - b) >= p:          # vrai aussi si a ou b vaut l'infini
        return min(a, b) + p
    return (a + b + math.sqrt(2 * p * p - (a - b) ** 2)) / 2


def fast_marching(cout, depart):
    """Carte d'action minimale U depuis le pixel 'depart' (Q2.2, algorithme de Sethian).

    cout   : carte de cout P~ (tableau 2D, valeurs strictement positives)
    depart : pixel de depart (ligne, colonne)
    Renvoie U, tableau de meme taille : U[depart] = 0, et U augmente en s'eloignant.

    Principe : on traite les pixels du plus proche au plus lointain (par U croissant),
    comme le front d'un feu les atteint. Le pixel ESSAI de plus petit U ne peut plus
    diminuer : on le fige, puis on met a jour ses 4 voisins.
    Un tas (heapq) donne a chaque fois le pixel ESSAI de plus petit U.
    """
    hauteur, largeur = cout.shape
    U = np.full((hauteur, largeur), np.inf)
    etat = np.full((hauteur, largeur), LOIN, dtype=np.int8)

    def valeur_figee(y, x):
        """U du pixel (y, x) s'il est dans l'image et deja fige, infini sinon."""
        if 0 <= y < hauteur and 0 <= x < largeur and etat[y, x] == FIGE:
            return U[y, x]
        return math.inf

    # 1. Initialisation : le depart vaut 0 et entre dans le tas
    y0, x0 = depart
    U[y0, x0] = 0.0
    etat[y0, x0] = ESSAI
    tas = [(0.0, y0, x0)]

    # 2. Boucle : on fige le pixel le plus proche, puis on met a jour ses voisins
    while tas:
        u, y, x = heapq.heappop(tas)
        if etat[y, x] == FIGE:
            continue                  # entree perimee (le pixel a deja ete fige plus tot)
        etat[y, x] = FIGE

        for dy, dx in VOISINS:
            yv, xv = y + dy, x + dx
            if not (0 <= yv < hauteur and 0 <= xv < largeur) or etat[yv, xv] == FIGE:
                continue
            a = min(valeur_figee(yv, xv - 1), valeur_figee(yv, xv + 1))   # gauche / droite
            b = min(valeur_figee(yv - 1, xv), valeur_figee(yv + 1, xv))   # haut / bas
            u_nouveau = mise_a_jour(a, b, cout[yv, xv])
            if u_nouveau < U[yv, xv]:
                U[yv, xv] = u_nouveau
                etat[yv, xv] = ESSAI
                heapq.heappush(tas, (u_nouveau, yv, xv))   # l'ancienne entree devient perimee

    return U
