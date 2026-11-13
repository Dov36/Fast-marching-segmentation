"""
Module evaluation : comparaison du contour obtenu avec la verite terrain (gt).

Etape 5 du projet - Sujet, partie 4.
Article : section 5, section 7.

Probleme : notre resultat est un CONTOUR (une ligne de pixels), alors que l'expert
donne une ZONE remplie (tous les pixels du ventricule gauche, label 3).
Pour comparer, on remplit d'abord l'interieur de notre contour, puis on compare
les deux zones avec le score de Dice.

Fonctions :
    masque_interieur(contour, forme)   -> Q4.2 : zone a l'interieur du contour (1 dedans, 0 dehors)
    dice(masque_a, masque_b)           -> Q4.2 : score de recouvrement entre 0 et 1
    part_dans_verite(chemin, verite)   -> pour un CHEMIN ouvert (vaisseau) : part du chemin
                                          qui reste dans la zone de l'expert (0 a 1)
    chemin_reference(verite, a, b)     -> chemin de reference de l'expert entre a et b
    f1_chemin(chemin, reference)       -> precision, rappel et F1 d'un chemin ouvert

Le Dice est exactement le F1-score de la segmentation :
    Dice = 2 VP / (2 VP + FP + FN) = 2 x precision x rappel / (precision + rappel)

Rappel : c'est le SEUL module qui a le droit d'utiliser gt.
"""

import numpy as np
from matplotlib.path import Path


def masque_interieur(contour, forme):
    """Remplit l'interieur d'un contour ferme (Q4.2).

    contour : tableau (n, 2) de points (ligne, colonne), par exemple renvoye par
              chemin.contour_ferme (le dernier point est relie au premier)
    forme   : taille de l'image (hauteur, largeur)
    Renvoie un tableau de booleens : True a l'interieur du contour (contour compris).

    Methode : on voit le contour comme un polygone, et on teste pour chaque pixel
    s'il est dedans (matplotlib.path.Path.contains_points).
    """
    hauteur, largeur = forme
    polygone = Path(contour[:, ::-1])                 # Path veut des points (x, y) = (colonne, ligne)
    yy, xx = np.indices((hauteur, largeur))
    centres = np.column_stack([xx.ravel(), yy.ravel()])
    masque = polygone.contains_points(centres).reshape(hauteur, largeur)
    masque[contour[:, 0], contour[:, 1]] = True       # les pixels du contour comptent aussi
    return masque


def dice(masque_a, masque_b):
    """Score de Dice entre deux zones (Q4.2) :

        Dice = 2 |A inter B| / (|A| + |B|)

    |A| = nombre de pixels de la zone A. Le Dice vaut 1 si les deux zones sont
    identiques, 0 si elles ne se touchent pas.
    """
    a = np.asarray(masque_a, dtype=bool)
    b = np.asarray(masque_b, dtype=bool)
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / total


def part_dans_verite(chemin, verite, tolerance=1):
    """Pour un chemin ouvert (par exemple le long d'un vaisseau) : proportion des pixels
    du chemin qui sont dans la zone de l'expert (entre 0 et 1).

    Le Dice n'a pas de sens ici (un chemin n'a pas d'interieur) : on verifie plutot que
    le chemin reste DANS le vaisseau. tolerance : on accepte un ecart de 'tolerance'
    pixel(s) autour de la zone de l'expert (le bord d'un vaisseau fin est imprecis).
    """
    zone = elargir(np.asarray(verite, dtype=bool), tolerance)
    return zone[chemin[:, 0], chemin[:, 1]].mean()


def elargir(zone, nb_pixels):
    """Elargit une zone (tableau de booleens) de 'nb_pixels' pixels (4 voisins a chaque fois)."""
    for _ in range(nb_pixels):
        elargie = zone.copy()
        elargie[1:, :] |= zone[:-1, :]
        elargie[:-1, :] |= zone[1:, :]
        elargie[:, 1:] |= zone[:, :-1]
        elargie[:, :-1] |= zone[:, 1:]
        zone = elargie
    return zone


def chemin_reference(verite, depart, arrivee):
    """Chemin de REFERENCE entre deux points, d'apres l'expert : le plus court chemin qui
    reste dans la zone de l'expert (ex. dans le vaisseau). On le calcule avec notre propre
    Fast Marching, sur une carte de cout 'expert' : 1 dans la zone, tres cher en dehors.
    Il sert de verite terrain pour un chemin ouvert (voir f1_chemin)."""
    import chemin as ch                        # import ici : chemin n'a pas besoin de gt
    cout_expert = np.where(np.asarray(verite, dtype=bool), 1.0, 1000.0)
    reference, _ = ch.chemin_minimal(cout_expert, depart, arrivee)
    return reference


def f1_chemin(chemin, reference, tolerance=2):
    """Precision, rappel et F1 d'un chemin ouvert, compare a un chemin de reference.

        precision = part de NOTRE chemin proche de la reference   (ce qu'on a trace est-il juste ?)
        rappel    = part de la REFERENCE proche de notre chemin   (a-t-on suivi tout le vaisseau ?)
        F1        = 2 x precision x rappel / (precision + rappel)  (meme formule que le Dice)

    'Proche' = a moins de 'tolerance' pixels (les deux chemins ne passent pas exactement
    par les memes pixels au milieu d'un vaisseau epais).
    """
    forme = (max(chemin[:, 0].max(), reference[:, 0].max()) + tolerance + 2,
             max(chemin[:, 1].max(), reference[:, 1].max()) + tolerance + 2)
    trace = np.zeros(forme, dtype=bool)
    trace[chemin[:, 0], chemin[:, 1]] = True
    trace_ref = np.zeros(forme, dtype=bool)
    trace_ref[reference[:, 0], reference[:, 1]] = True
    precision = elargir(trace_ref, tolerance)[chemin[:, 0], chemin[:, 1]].mean()
    rappel = elargir(trace, tolerance)[reference[:, 0], reference[:, 1]].mean()
    f1 = 0.0 if precision + rappel == 0 else 2 * precision * rappel / (precision + rappel)
    return precision, rappel, f1
