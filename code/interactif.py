"""
Module interactif : on clique des points sur le bord d'un objet, et le programme trace
le contour qui passe par ces points (chemin minimal, Fast Marching + backtracking).

Fenetre principale (deux panneaux) :
    - a gauche : l'image. On clique NB_POINTS points (parametres.py) : le calcul part tout seul.
      Pour chaque morceau p_i -> p_(i+1), on voit le front du Fast Marching se propager
      (zone rouge) jusqu'a atteindre le point suivant, puis le chemin trouve (ligne).
    - a droite : la carte U depuis p0 (cout du chemin le moins cher depuis p0),
      avec ses lignes de niveau et le chemin.

    2 points  -> chemin ouvert p0 -> p1
    3 points ou plus -> contour ferme p0 -> p1 -> ... -> p0

Fenetre resultat : l'image avec le contour, et le contour seul (fond noir).

Commandes pendant le choix des points :
    clic gauche : ajouter un point      clic droit : enlever le dernier point
    au NB_POINTS-ieme clic, le calcul demarre tout seul
    (Entree, la figure etant active : valider avant, avec au moins 2 points)
    fermer la fenetre : terminer
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import affichage as af
import chemin as ch
import evaluation as ev
import fast_marching as fm
from parametres import NB_POINTS

NB_IMAGES_ANIMATION = 25     # nombre d'etapes de l'animation (par morceau)
PAUSE_ANIMATION = 0.03       # duree d'une etape (secondes)
COULEUR_CONTOUR = "cyan"


def _fenetre_ouverte(fig):
    return plt.fignum_exists(fig.number)


def _choisir_points(fig, hauteur, largeur, nb_points):
    """Attend les clics de l'utilisateur. Renvoie une liste de (ligne, colonne), ou None si fin."""
    plt.figure(fig.number)
    try:
        clics = plt.ginput(n=nb_points, timeout=0)   # s'arrete tout seul apres nb_points clics
    except Exception:                           # fenetre fermee pendant l'attente
        return None
    if not clics or not _fenetre_ouverte(fig):
        return None
    # ginput renvoie (x, y) = (colonne, ligne) -> on convertit en (ligne, colonne) entiers
    return [(min(max(int(round(y)), 0), hauteur - 1), min(max(int(round(x)), 0), largeur - 1))
            for x, y in clics]


def contour_interactif(image, cout, titre="", dossier=None, nom="image", nb_points=None,
                       verite=None, zoom=None):
    """Fenetre interactive : clics -> Fast Marching -> backtracking -> contour.

    image   : image affichee (niveaux de gris, 0 a 1)
    cout    : carte de cout P~ utilisee par le Fast Marching
    dossier : si donne, les figures y sont enregistrees ({nom}_fast_marching.png,
              {nom}_contour.png et {nom}_contour_seul.png)
    nb_points : nombre de clics avant le calcul (par defaut NB_POINTS de parametres.py)
    verite    : zone de l'expert (tableau de booleens), pour l'IRM et DRIVE. Si elle est donnee :
                contour ferme -> score de Dice ; chemin ouvert -> part du chemin dans la zone
    zoom      : zone agrandie dans la figure d'evaluation (lignes, colonnes)
    Renvoie le dernier contour trace (tableau (n, 2) de (ligne, colonne)), ou None.
    """
    nb_points = nb_points or NB_POINTS
    hauteur, largeur = image.shape
    fig, (ax_img, ax_u) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(f"{titre} : contour par chemin minimal\n"
                 f"clic gauche = ajouter un point ({nb_points} points), clic droit = annuler, "
                 "fermer la fenetre = terminer", fontsize=11)

    # Panneau de gauche : image + calque rouge (zone atteinte par le front) + contour
    ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
    calque = np.zeros((hauteur, largeur, 4))
    calque[..., 0] = 1.0                                   # rouge
    zone = ax_img.imshow(calque)
    ligne_contour, = ax_img.plot([], [], "-", color=COULEUR_CONTOUR, linewidth=2)
    points_affiches, = ax_img.plot([], [], "o", color="yellow", markersize=7,
                                   markeredgecolor="black")
    ax_img.set_xlabel("colonne x (pixels)")
    ax_img.set_ylabel("ligne y (pixels)")
    ax_img.legend(handles=[
        Line2D([], [], marker="o", color="yellow", markeredgecolor="black", linestyle="",
               label="points cliques"),
        Patch(color="red", alpha=0.45, label="zone deja atteinte par le front"),
        Line2D([], [], color=COULEUR_CONTOUR, linewidth=2, label="chemin minimal (contour)"),
    ], loc="lower right", fontsize=8, framealpha=0.8)

    # Panneau de droite : d'abord la carte de cout, puis la carte U
    im_droite = ax_u.imshow(cout, cmap="gray")
    barre = fig.colorbar(im_droite, ax=ax_u, fraction=0.035)
    barre.set_label("cout P~ d'un pixel (faible = bord)")
    ax_u.set_title("Carte de cout P~")
    ax_u.set_xlabel("colonne x (pixels)")
    ax_u.set_ylabel("ligne y (pixels)")
    lignes_niveau = None
    chemin_droite, = ax_u.plot([], [], "-", color=COULEUR_CONTOUR, linewidth=1.5)
    depart_droite, = ax_u.plot([], [], "o", color="yellow", markersize=7, markeredgecolor="black")
    fig.tight_layout()

    contour = None
    print("\n[interactif] Clique des points sur le bord de l'objet (image de gauche), "
          f"Le calcul demarre au {nb_points}e clic.")
    while _fenetre_ouverte(fig):
        ax_img.set_title(f"Clique {nb_points} points sur un bord (le calcul part au {nb_points}e clic)")
        fig.canvas.draw_idle()
        points = _choisir_points(fig, hauteur, largeur, nb_points)
        if points is None:
            break
        if len(points) < 2:
            print("[interactif] Il faut au moins 2 points.")
            continue

        for texte in list(ax_img.texts):     # on enleve les etiquettes du contour precedent
            texte.remove()
        ferme = len(points) >= 3
        ys, xs = zip(*points)
        points_affiches.set_data(xs, ys)
        ligne_contour.set_data([], [])
        for k, (y, x) in enumerate(points):
            ax_img.annotate(f"p{k}", (x, y), xytext=(6, -6), textcoords="offset points",
                            color="yellow", fontsize=10, fontweight="bold")

        # Morceaux a tracer : p0->p1, p1->p2, ..., (et dernier->p0 si contour ferme)
        if ferme:
            paires = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        else:
            paires = [(points[0], points[1])]

        morceaux = []
        debut = time.perf_counter()
        for i, (depart, arrivee) in enumerate(paires):
            nom_morceau = f"p{i} -> p{(i + 1) % len(points)}"
            ax_img.set_title(f"Morceau {nom_morceau} : calcul du Fast Marching...")
            fig.canvas.draw_idle()
            plt.pause(0.01)

            # 1. Fast Marching depuis le point de depart du morceau
            #    (interdiction de repasser sur les morceaux deja traces, voir chemin.py)
            deja_trace = np.concatenate(morceaux) if morceaux else None
            U = fm.fast_marching(ch.cout_sans_retour(cout, deja_trace, depart, arrivee), depart)

            # 2. Animation : le front grandit jusqu'a atteindre le point d'arrivee
            ax_img.set_title(f"Morceau {nom_morceau} : propagation du front")
            for t in np.linspace(0, U[arrivee], NB_IMAGES_ANIMATION):
                if not _fenetre_ouverte(fig):
                    return contour
                calque[..., 3] = 0.45 * (U <= t)
                zone.set_data(calque)
                plt.pause(PAUSE_ANIMATION)

            # 3. Backtracking : on descend U depuis l'arrivee jusqu'au depart
            chemin = ch.descente(U, arrivee)[::-1]
            morceaux.append(chemin if not morceaux else chemin[1:])
            contour = np.concatenate(morceaux)
            ligne_contour.set_data(contour[:, 1], contour[:, 0])
            print(f"[interactif] {nom_morceau} : {depart} -> {arrivee}, "
                  f"cout du chemin U = {U[arrivee]:.1f}, {len(chemin)} pixels")

            # Panneau de droite : la carte U depuis p0 (premier morceau)
            if i == 0:
                im_droite.set_data(U)
                im_droite.set_cmap("magma")
                im_droite.set_clim(0, U.max())
                barre.set_label("U = cout du chemin le moins cher depuis p0")
                if lignes_niveau is not None:
                    lignes_niveau.remove()
                lignes_niveau = ax_u.contour(U, levels=20, colors="white", linewidths=0.4)
                depart_droite.set_data([depart[1]], [depart[0]])
                ax_u.set_title("Carte U depuis p0 (sombre = proche / pas cher)\n"
                               "lignes blanches = position du front a temps reguliers",
                               fontsize=10)
            chemin_droite.set_data(contour[:, 1], contour[:, 0])

        # Fin : on efface la zone rouge pour ne garder que le contour
        calque[..., 3] = 0
        zone.set_data(calque)
        if ferme:      # on referme visuellement la ligne
            ligne_contour.set_data(np.r_[contour[:, 1], contour[0, 1]],
                                   np.r_[contour[:, 0], contour[0, 0]])
        type_contour = "contour ferme" if ferme else "chemin ouvert"
        ax_img.set_title(f"{type_contour} ({len(points)} points, {len(contour)} pixels, "
                         f"{time.perf_counter() - debut:.1f} s)")
        fig.canvas.draw_idle()
        print(f"[interactif] {type_contour} termine : {len(contour)} pixels.")

        # Figure resultat : image + contour, et contour seul
        fig_res = af.afficher_contour(image, contour, ferme, f"{titre} : {type_contour}")
        if dossier is not None:
            af.enregistrer_figure(fig, dossier / f"{nom}_fast_marching.png")
            af.enregistrer_figure(fig_res, dossier / f"{nom}_contour.png")
            af.sauvegarder_png(dossier / f"{nom}_contour_seul.png",
                               ch.image_contour(contour, image.shape))

        # Etape 5 (IRM seulement) : comparaison avec l'expert (score de Dice)
        if verite is not None and ferme:
            masque = ev.masque_interieur(contour, image.shape)
            score = ev.dice(masque, verite)
            print(f"[evaluation] Dice = {score:.3f}  (ta zone : {masque.sum()} pixels, "
                  f"expert : {verite.sum()} pixels, en commun : {(masque & verite).sum()})")
            fig_eval = af.afficher_evaluation(image, contour, masque, verite, score,
                                              f"{titre} : comparaison avec l'expert", zoom=zoom)
            if dossier is not None:
                af.enregistrer_figure(fig_eval, dossier / f"{nom}_evaluation.png")
            fig_eval.show()
        elif verite is not None:       # chemin ouvert (vaisseau) : part du chemin dans la verite
            score = ev.part_dans_verite(contour, verite)
            print(f"[evaluation] {100 * score:.1f} % du chemin est dans la zone de l'expert")
            fig_eval = af.afficher_evaluation_chemin(image, contour, verite, score,
                                                     f"{titre} : comparaison avec l'expert")
            if dossier is not None:
                af.enregistrer_figure(fig_eval, dossier / f"{nom}_evaluation.png")
            fig_eval.show()

        fig_res.show()                # affiche la fenetre resultat tout de suite
        plt.pause(0.5)
        if dossier is not None:
            print(f"[interactif] Images enregistrees : {nom}_contour.png, {nom}_contour_seul.png, "
                  f"{nom}_fast_marching.png (dossier figures/)")
        print(f"[interactif] Pour un autre contour : clique {nb_points} nouveaux points ; "
              "sinon ferme la fenetre.")
        plt.figure(fig.number)       # on revient sur la fenetre principale pour la suite
    return contour
