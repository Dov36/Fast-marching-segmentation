"""
Module chemin : extraction du chemin minimal par descente de U (backtracking).

Etape 4 du projet - Sujet, partie 3.
Article : section 4.5 (eq. 21).

Rappel : U[x] = cout du chemin le moins cher entre le depart p0 et le pixel x.
U vaut 0 en p0 et augmente quand on s'eloigne. Donc si on part de p1 et qu'on
descend toujours la pente de U, on finit forcement en p0, en suivant le chemin
le moins cher (comme une bille qui roule au fond de la vallee).

Fonctions :
    descente(U, arrivee)                 -> Q3.1 : chemin de 'arrivee' jusqu'au depart (U = 0)
    chemin_minimal(cout, p0, p1)         -> Q3.2 : Fast Marching depuis p0, puis descente depuis p1
    contour_ferme(cout, points)          -> Q3.3 : enchaine p0->p1, p1->p2, ..., dernier->p0
    cout_sans_retour(cout, ...)          -> interdit de repasser sur un morceau deja trace
    image_contour(contour, forme)        -> image noire avec seulement le contour en blanc

Regle : ces fonctions CALCULENT et RENVOIENT des tableaux.
Aucun affichage ici (les figures vont dans affichage.py / interactif.py).
"""

import numpy as np

import fast_marching as fm

# Les 8 voisins d'un pixel (dy, dx) : on autorise aussi les diagonales
VOISINS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def descente(U, arrivee, nb_max=None):
    """Chemin minimal de 'arrivee' jusqu'au depart du Fast Marching (Q3.1).

    A chaque pas, on va sur le voisin (8-connexite) qui a le plus petit U.
    On s'arrete quand plus aucun voisin n'est plus bas : on est au depart (U = 0).
    nb_max : garde-fou contre une boucle infinie (par defaut : nombre de pixels).

    Renvoie un tableau (n, 2) de points (ligne, colonne), de 'arrivee' vers le depart.
    """
    hauteur, largeur = U.shape
    if nb_max is None:
        nb_max = hauteur * largeur
    y, x = arrivee
    chemin = [(y, x)]
    for _ in range(nb_max):
        meilleur = (U[y, x], y, x)
        for dy, dx in VOISINS_8:
            yv, xv = y + dy, x + dx
            if 0 <= yv < hauteur and 0 <= xv < largeur and U[yv, xv] < meilleur[0]:
                meilleur = (U[yv, xv], yv, xv)
        if (meilleur[1], meilleur[2]) == (y, x):     # aucun voisin plus bas : fin
            break
        _, y, x = meilleur
        chemin.append((y, x))
    return np.array(chemin)


def chemin_minimal(cout, p0, p1):
    """Chemin le moins cher entre p0 et p1 (Q3.2).

    1. Fast Marching depuis p0 -> carte U
    2. Descente de U depuis p1  -> chemin de p1 vers p0, qu'on retourne (p0 -> p1)
    Renvoie (chemin, U) : chemin = tableau (n, 2) de (ligne, colonne).
    """
    U = fm.fast_marching(cout, p0)
    chemin = descente(U, p1)[::-1]
    return chemin, U


def cout_sans_retour(cout, deja_trace, depart, arrivee, marge=3):
    """Carte de cout ou les pixels deja traces (et leurs voisins) deviennent tres chers.

    Pourquoi : pour un contour ferme, le dernier morceau (p2 -> p0) pourrait revenir
    en arriere en repassant sur p2 -> p1 -> p0, deja traces, au lieu de faire le tour
    par l'autre cote. On "interdit" donc les pixels deja utilises.

    deja_trace : tableau (n, 2) des pixels deja dans le contour (ou None)
    marge      : on interdit aussi les pixels a moins de 'marge' pixels du trace
                 (sinon le chemin longe le trace juste a cote)
    Les zones autour de 'depart' et 'arrivee' restent autorisees (le morceau doit y passer).
    """
    if deja_trace is None or len(deja_trace) == 0:
        return cout
    hauteur, largeur = cout.shape
    interdit = np.zeros((hauteur, largeur), dtype=bool)
    for dy in range(-marge, marge + 1):          # on "epaissit" le trace de 'marge' pixels
        for dx in range(-marge, marge + 1):
            ys = np.clip(deja_trace[:, 0] + dy, 0, hauteur - 1)
            xs = np.clip(deja_trace[:, 1] + dx, 0, largeur - 1)
            interdit[ys, xs] = True
    yy, xx = np.indices((hauteur, largeur))
    for y, x in (depart, arrivee):                # on garde l'acces au depart et a l'arrivee
        interdit[np.hypot(yy - y, xx - x) <= 2 * marge] = False
    nouveau = cout.copy()
    nouveau[interdit] = 100 * cout.max()          # tres cher, mais pas infini
    return nouveau


def contour_ferme(cout, points):
    """Contour ferme passant par tous les points cliques (Q3.3).

    On enchaine les chemins minimaux p0->p1, p1->p2, ..., et le dernier -> p0.
    Chaque morceau a interdiction de repasser sur les morceaux precedents (cout_sans_retour).
    Avec 2 points seulement, on renvoie le chemin ouvert p0 -> p1.
    Renvoie (contour, cartes) : contour = tableau (n, 2) de (ligne, colonne),
    cartes = liste des cartes U (une par morceau).
    """
    if len(points) == 2:
        paires = [(points[0], points[1])]
    else:
        paires = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    morceaux, cartes = [], []
    for depart, arrivee in paires:
        deja_trace = np.concatenate(morceaux) if morceaux else None
        chemin, U = chemin_minimal(cout_sans_retour(cout, deja_trace, depart, arrivee),
                                   depart, arrivee)
        morceaux.append(chemin if not morceaux else chemin[1:])   # evite les doublons
        cartes.append(U)
    return np.concatenate(morceaux), cartes


def image_contour(contour, forme):
    """Image noire de taille 'forme' avec les pixels du contour a 1 (blanc)."""
    resultat = np.zeros(forme)
    resultat[contour[:, 0], contour[:, 1]] = 1.0
    return resultat
