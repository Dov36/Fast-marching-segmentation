"""
Module affichage : toutes les fonctions qui affichent ou enregistrent des figures.

Regle du projet : les autres modules CALCULENT et renvoient des tableaux,
ce module les AFFICHE. Aucun calcul de la methode ne doit se trouver ici.

On l'appelle depuis main.py avec :
    import affichage as af
"""

import matplotlib.pyplot as plt
import numpy as np

from donnees_acdc import LABELS, LABEL_VG, coupes_avec_label, normaliser


# ----------------------------------------------------------------------------
# 1. Donnees (etape 1)
# ----------------------------------------------------------------------------
def afficher_infos(donnees):
    """Ecrit dans le terminal les informations principales sur le volume."""
    v, voxel, gt = donnees["volume"], donnees["taille_voxel"], donnees["gt"]
    print(f"Fichier IRM          : {donnees['fichier_image'].name}")
    print(f"Fichier segmentation : {donnees['fichier_gt'].name}")
    print(f"Taille du volume     : {v.shape}  (hauteur, largeur, nombre de coupes)")
    print(f"Taille d'un pixel    : {voxel[0]:.2f} x {voxel[1]:.2f} mm, "
          f"ecart entre coupes : {voxel[2]:.1f} mm")
    print(f"Intensites           : min {v.min():.0f}, max {v.max():.0f}")
    print(f"Coupes avec le ventricule gauche : {coupes_avec_label(gt)}")


def afficher_toutes_les_coupes(donnees, titre="", coupe_choisie=None, avec_contour=False):
    """Affiche toutes les coupes du volume (de la base du coeur jusqu'a la pointe).

    coupe_choisie : numero de la coupe a encadrer en jaune (None = aucune)
    avec_contour  : True pour ajouter le contour du ventricule gauche de l'expert en rouge
    """
    volume, gt = donnees["volume"], donnees["gt"]
    nb = volume.shape[2]
    colonnes = 5
    lignes = int(np.ceil(nb / colonnes))
    fig, axes = plt.subplots(lignes, colonnes, figsize=(2.8 * colonnes, 2.8 * lignes))
    for k, ax in enumerate(np.ravel(axes)):
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= nb:
            ax.axis("off")
            continue
        ax.imshow(normaliser(volume[:, :, k]), cmap="gray")
        if avec_contour and np.any(gt[:, :, k] == LABEL_VG):
            ax.contour(gt[:, :, k] == LABEL_VG, levels=[0.5], colors="r", linewidths=0.8)
        if k == coupe_choisie:
            ax.set_title(f"coupe {k} (choisie)", fontsize=10, color="darkorange",
                         fontweight="bold")
            for bord in ax.spines.values():
                bord.set_edgecolor("gold")
                bord.set_linewidth(4)
        else:
            ax.set_title(f"coupe {k}", fontsize=9)
            for bord in ax.spines.values():
                bord.set_visible(False)
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


def afficher_image(image, titre=""):
    """Affiche une seule image en niveaux de gris, sans contour."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    ax.set_title(titre)
    ax.axis("off")
    fig.tight_layout()
    return fig


def afficher_coupe(image, gt, titre=""):
    """Affiche une coupe : IRM, segmentation de l'expert, et contour du ventricule gauche."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("IRM (coupe choisie)")

    im = axes[1].imshow(gt, cmap="viridis", vmin=0, vmax=3)
    axes[1].set_title("Segmentation de l'expert")
    barre = fig.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3], fraction=0.046)
    barre.ax.set_yticklabels([LABELS[i] for i in range(4)])

    axes[2].imshow(image, cmap="gray")
    axes[2].contour(gt == LABEL_VG, levels=[0.5], colors="r", linewidths=1.5)
    axes[2].set_title("Contour du ventricule gauche (expert)")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 2. Enregistrement
# ----------------------------------------------------------------------------
def enregistrer_figure(fig, fichier, dpi=150):
    """Enregistre une figure matplotlib dans un fichier PNG.

    Si le fichier ne peut pas etre ecrit (par exemple parce qu'il est ouvert dans
    une visionneuse, ou en cours de synchronisation par OneDrive), on affiche un
    avertissement et le programme CONTINUE au lieu de s'arreter.
    """
    try:
        fig.savefig(fichier, dpi=dpi)
    except OSError as erreur:
        print(f"[!] Figure non enregistree ({fichier.name}) : {erreur.strerror}. "
              f"Le fichier est peut-etre ouvert ailleurs ou en cours de synchronisation.")


def sauvegarder_png(fichier, image):
    """Enregistre l'image (valeurs entre 0 et 1) comme une photo PNG en niveaux de gris.
    Meme protection que enregistrer_figure si le fichier est verrouille."""
    try:
        plt.imsave(str(fichier), image, cmap="gray", vmin=0, vmax=1)
    except OSError as erreur:
        print(f"[!] Image non enregistree ({fichier.name}) : {erreur.strerror}.")


# ----------------------------------------------------------------------------
# 3. Carte de cout (etape 2)
# ----------------------------------------------------------------------------
def afficher_comparaison(images, titres, titre="", zoom=(slice(70, 140), slice(95, 170))):
    """Affiche plusieurs images cote a cote : en entier (ligne du haut)
    et zoomees sur le ventricule (ligne du bas).

    images : liste de tableaux 2D (valeurs entre 0 et 1)
    titres : liste des titres, un par image
    zoom   : zone a agrandir (lignes, colonnes) ; par defaut autour du ventricule
             gauche de patient001, coupe 5. (slice(None), slice(None)) = image entiere.
    """
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    for k, (img, nom) in enumerate(zip(images, titres)):
        axes[0, k].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_title(nom)
        axes[1, k].imshow(img[zoom], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, k].set_title(nom + " (zoom)")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


def afficher_gradient(gx, gy, norme, titre="", zoom=(slice(70, 140), slice(95, 170))):
    """Affiche gx, gy et la norme du gradient, en entier (haut) et zoomes (bas).

    gx et gy sont signes : rouge = positif (sombre -> clair), bleu = negatif
    (clair -> sombre), blanc = 0. La norme est en niveaux de gris (blanc = bord fort).
    """
    m = max(np.abs(gx).max(), np.abs(gy).max())          # meme echelle pour gx et gy
    cartes = [(gx, "gx (Sobel horizontal)", "bwr", -m, m),
              (gy, "gy (Sobel vertical)", "bwr", -m, m),
              (norme, "Norme du gradient", "gray", 0, norme.max())]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for k, (carte, nom, cmap, vmin, vmax) in enumerate(cartes):
        im = axes[0, k].imshow(carte, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, k].set_title(nom)
        fig.colorbar(im, ax=axes[0, k], fraction=0.035)
        axes[1, k].imshow(carte[zoom], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[1, k].set_title(nom + " (zoom)")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


def afficher_potentiel(cout, titre=""):
    """Affiche la carte de cout P~ : sombre = bon marche (bords), clair = cher."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cout, cmap="gray")
    ax.set_title("Carte de cout P~ (sombre = bon marche)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.035, label="cout")
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


def afficher_comparaison_couts(couts, titre="", zoom=(slice(70, 140), slice(95, 170))):
    """Compare plusieurs cartes de cout cote a cote (une par lissage), a la meme echelle.

    couts : dictionnaire {nom : carte de cout}, par exemple {"aucun": ..., "gauss": ...}
    zoom  : zone a agrandir (lignes, colonnes) ; (slice(None), slice(None)) = image entiere
    """
    n = len(couts)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    vmin = min(c.min() for c in couts.values())
    vmax = max(c.max() for c in couts.values())
    for ax, (nom, cout) in zip(axes, couts.items()):
        ax.imshow(cout[zoom], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"P~ - lissage : {nom}")
        ax.axis("off")
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 4. Fast Marching et contour (etapes 3 et 4)
#    (la fenetre interactive est dans interactif.py)
# ----------------------------------------------------------------------------
def afficher_contour(image, contour, ferme=True, titre=""):
    """Resultat final : a gauche l'image avec le contour, a droite le contour seul.

    contour : tableau (n, 2) de points (ligne, colonne), par exemple renvoye par
              chemin.contour_ferme
    ferme   : True -> on relie le dernier point au premier
    """
    ys, xs = contour[:, 0], contour[:, 1]
    if ferme:
        ys, xs = np.r_[ys, ys[0]], np.r_[xs, xs[0]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].plot(xs, ys, "-", color="cyan", linewidth=2, label="contour trouve")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("Image + contour")
    axes[1].imshow(np.zeros(image.shape), cmap="gray", vmin=0, vmax=1)
    axes[1].plot(xs, ys, "-", color="white", linewidth=1.5)
    axes[1].set_title("Contour seul")
    for ax in axes:
        ax.set_xlabel("colonne x (pixels)")
        ax.set_ylabel("ligne y (pixels)")
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 5. Evaluation (etape 5) : comparaison avec l'expert
# ----------------------------------------------------------------------------
def afficher_evaluation(image, contour, masque, verite, score, titre="", zoom=None):
    """Compare notre resultat a celui de l'expert (IRM).

    A gauche : l'image avec notre contour (cyan) et le contour de l'expert (rouge).
    A droite : chaque pixel colorie selon le cas :
        vert  = dans les deux zones (bien trouve)
        rouge = chez l'expert seulement (oublie)
        bleu  = chez nous seulement (en trop)
    zoom : zone agrandie (lignes, colonnes) ; None = image entiere
    """
    if zoom is None:
        zoom = (slice(None), slice(None))
    y0 = zoom[0].start or 0
    x0 = zoom[1].start or 0
    couleurs = np.zeros(image.shape + (3,))
    couleurs[masque & verite] = (0.1, 0.8, 0.1)       # vert
    couleurs[verite & ~masque] = (0.9, 0.1, 0.1)      # rouge
    couleurs[masque & ~verite] = (0.1, 0.3, 0.9)      # bleu

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image[zoom], cmap="gray")
    axes[0].plot(np.r_[contour[:, 1], contour[0, 1]] - x0,
                 np.r_[contour[:, 0], contour[0, 0]] - y0, "-", color="cyan", linewidth=2,
                 label="notre contour (Fast Marching)")
    axes[0].contour(verite[zoom], levels=[0.5], colors="red", linewidths=1.5)
    axes[0].plot([], [], "-", color="red", label="contour de l'expert")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("Contours superposes")
    axes[1].imshow(image[zoom], cmap="gray")
    axes[1].imshow(couleurs[zoom], alpha=0.6)
    axes[1].set_title("vert = commun, rouge = oublie, bleu = en trop")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{titre}\nDice = {score:.3f}")
    fig.tight_layout()
    return fig


def afficher_evaluation_chemin(image, chemin, verite, score, titre=""):
    """Compare un chemin ouvert a la zone de l'expert (par exemple les vaisseaux de DRIVE).
    A gauche : l'image et le chemin. A droite : vaisseaux de l'expert en blanc, chemin en
    vert quand il est dans un vaisseau, en rouge quand il en sort."""
    dedans = verite[chemin[:, 0], chemin[:, 1]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    axes[0].imshow(image, cmap="gray")
    axes[0].plot(chemin[:, 1], chemin[:, 0], "-", color="cyan", linewidth=1.5)
    axes[0].set_title("Image + chemin minimal")
    axes[1].imshow(verite, cmap="gray")
    axes[1].plot(chemin[dedans, 1], chemin[dedans, 0], ".", color="lime", markersize=2,
                 label="chemin dans un vaisseau")
    axes[1].plot(chemin[~dedans, 1], chemin[~dedans, 0], ".", color="red", markersize=3,
                 label="chemin hors vaisseau")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title("Vaisseaux de l'expert (blanc)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{titre}\n{100 * score:.1f} % du chemin dans un vaisseau")
    fig.tight_layout()
    return fig
