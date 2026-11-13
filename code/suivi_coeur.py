"""
Suivi du ventricule gauche pendant un battement de coeur (IRM cine) -> animation GIF.

L'IRM "cine" filme le coeur : le fichier patient001_4d.nii.gz contient 30 images d'un
meme battement (pour chaque coupe). On trace le contour du ventricule gauche sur la
premiere image (6 points), puis on le SUIT d'image en image :

    1. Les 6 points de depart sont places sur le bord du ventricule (image 1).
    2. Pour chaque image suivante, chaque point se deplace le long de son RAYON
       (la droite qui part du centre du ventricule et passe par ce point) :
       il va sur le pixel le moins cher (le bord), avec une PENALITE s'il saute trop
       loin de sa position precedente (le coeur bouge peu entre deux images).
    3. Le contour de chaque image est trace par Fast Marching entre ces 6 points
       (exactement comme avec des clics, mais les points bougent tout seuls).

Pourquoi la penalite : sans elle, un point peut sauter d'un bord a un autre (les muscles
papillaires a l'interieur du ventricule) et le contour "tremble" d'une image a l'autre.

Resultats dans figures/gif_coeur/ :
    suivi_coeur.gif         le coeur qui bat, avec le contour qui suit (rouge = expert en ED et ES)
    aire_ventricule.png     aire du ventricule au cours du battement
    comparaison_ED_ES.png   notre contour et celui de l'expert en fin de diastole et de systole

Lancer depuis le dossier Projet avec :
    python code/suivi_coeur.py           (environ 2 minutes)
"""

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import carte_cout as cc
import chemin as ch
import donnees_acdc as da
import evaluation as ev
from parametres import DOSSIER_DONNEES, DOSSIER_FIGURES

PATIENT = "patient001"
COUPE = None            # None = coupe du milieu du ventricule
W, ALPHA = 0.1, 10.0    # carte de cout (memes reglages que main_irm.py)
NB_POINTS = 6           # points sur le contour
RECHERCHE = 6           # un point peut bouger de +/- 6 pixels le long de son rayon
PENALITE = 0.08         # cout ajoute par pixel de deplacement par rapport a l'image precedente
CLIQUER = False         # True : tu cliques les 6 points sur la 1re image (sinon : bord de l'expert)
ZOOM = (slice(60, 150), slice(85, 180))
DOSSIER_GIF = DOSSIER_FIGURES / "gif_coeur"


def points_sur_le_bord(zone, nb):
    """nb points repartis regulierement (en angle) sur le bord d'une zone (comme nb clics)."""
    bord = zone & ev.elargir(~zone, 1)
    ys, xs = np.nonzero(bord)
    cy, cx = ys.mean(), xs.mean()
    angles = np.arctan2(ys - cy, xs - cx)
    points = []
    for a in np.linspace(-np.pi, np.pi, nb, endpoint=False):
        k = np.argmin(np.abs(np.angle(np.exp(1j * (angles - a)))))
        points.append((int(ys[k]), int(xs[k])))
    return points


def suivre(couts, points_depart):
    """Suit les points d'image en image, puis trace le contour de chaque image.

    couts         : liste des cartes de cout (une par image du battement)
    points_depart : les points sur la 1re image
    Renvoie (contours, points) : une liste de contours et une liste de points par image.
    """
    centre = np.mean(points_depart, axis=0)
    rayons = []                                    # direction centre -> point (longueur 1)
    for y, x in points_depart:
        d = np.array([y - centre[0], x - centre[1]], dtype=float)
        rayons.append(d / np.hypot(*d))
    decalage = [0] * len(points_depart)            # position de chaque point sur son rayon
    contours, tous_les_points = [], []
    for t, cout in enumerate(couts):
        if t > 0:
            for i, ((y, x), d) in enumerate(zip(points_depart, rayons)):
                meilleur = None
                for k in range(-RECHERCHE, RECHERCHE + 1):
                    p = (int(round(y + k * d[0])), int(round(x + k * d[1])))
                    score = cout[p] + PENALITE * abs(k - decalage[i])
                    if meilleur is None or score < meilleur[0]:
                        meilleur = (score, k)
                decalage[i] = meilleur[1]
        points = [(int(round(y + decalage[i] * d[0])), int(round(x + decalage[i] * d[1])))
                  for i, ((y, x), d) in enumerate(zip(points_depart, rayons))]
        contour, _ = ch.contour_ferme(cout, points)
        contours.append(contour)
        tous_les_points.append(points)
        print(f"image {t + 1:2d}/{len(couts)} : contour de {len(contour)} pixels", flush=True)
    return contours, tous_les_points


def ferme(contour):
    """Coordonnees (x, y) du contour, avec le dernier point relie au premier."""
    return np.r_[contour[:, 1], contour[0, 1]], np.r_[contour[:, 0], contour[0, 0]]


if __name__ == "__main__":
    if not CLIQUER:
        matplotlib.use("Agg")
    DOSSIER_GIF.mkdir(parents=True, exist_ok=True)

    # 1. Le film du battement et les deux images annotees par l'expert
    dossier = da.obtenir_dossier_patient(DOSSIER_DONNEES, PATIENT, telecharger=False)
    film = np.asarray(nib.load(str(dossier / f"{PATIENT}_4d.nii.gz")).dataobj, dtype=float)
    info = da.lire_info_cfg(dossier)
    t_ed, t_es = int(info["ED"]) - 1, int(info["ES"]) - 1      # numeros d'image (0 = 1re)
    ed, es = da.charger_patient(dossier, "ED"), da.charger_patient(dossier, "ES")
    coupe = COUPE if COUPE is not None else da.coupe_du_milieu(ed["gt"])
    vg_ed = ed["gt"][:, :, coupe] == da.LABEL_VG
    vg_es = es["gt"][:, :, coupe] == da.LABEL_VG
    mm2 = ed["taille_voxel"][0] * ed["taille_voxel"][1]       # surface d'un pixel en mm2
    images = [da.normaliser(film[:, :, coupe, t]) for t in range(film.shape[3])]
    print(f"{PATIENT}, coupe {coupe} : {len(images)} images, ED = image {t_ed + 1}, "
          f"ES = image {t_es + 1}")

    # 2. Cartes de cout et points de depart
    couts = []
    for image in images:
        gx, gy = cc.gradient(image)
        couts.append(cc.potentiel(cc.norme_gradient(gx, gy), W, ALPHA))
    if CLIQUER:
        plt.imshow(images[0], cmap="gray")
        plt.title(f"Clique {NB_POINTS} points sur le bord du ventricule gauche")
        points_depart = [(int(round(y)), int(round(x))) for x, y in plt.ginput(NB_POINTS, timeout=0)]
        plt.close()
    else:
        points_depart = points_sur_le_bord(vg_ed, NB_POINTS)

    # 3. Suivi sur tout le battement
    contours, points = suivre(couts, points_depart)
    masques = [ev.masque_interieur(c, images[0].shape) for c in contours]
    aires = np.array([m.sum() * mm2 for m in masques])
    dice_ed = ev.dice(masques[t_ed], vg_ed)
    dice_es = ev.dice(masques[t_es], vg_es)
    variation = (aires[t_ed] - aires[t_es]) / aires[t_ed]       # meme calcul que l'expert : ED -> ES
    variation_expert = (vg_ed.sum() - vg_es.sum()) / vg_ed.sum()
    print(f"\nDice avec l'expert : ED = {dice_ed:.3f}, ES = {dice_es:.3f}")
    print(f"Variation d'aire entre ED et ES : {100 * variation:.0f} % "
          f"(expert : {100 * variation_expert:.0f} %) ; aire minimale a l'image "
          f"{int(np.argmin(aires)) + 1}")

    # 4. Animation GIF
    y0, x0 = ZOOM[0].start, ZOOM[1].start
    fig, ax = plt.subplots(figsize=(5, 5))

    def dessiner(t):
        ax.clear()
        ax.imshow(images[t][ZOOM], cmap="gray", vmin=0, vmax=1)
        xs, ys = ferme(contours[t])
        ax.plot(xs - x0, ys - y0, "-", color="cyan", linewidth=2)
        ax.plot([p[1] - x0 for p in points[t]], [p[0] - y0 for p in points[t]], "o",
                color="yellow", markersize=4, markeredgecolor="black")
        titre = f"image {t + 1}/{len(images)}  -  aire {aires[t]:.0f} mm2"
        for t_ref, zone, nom in [(t_ed, vg_ed, "ED"), (t_es, vg_es, "ES")]:
            if t == t_ref:
                ax.contour(zone[ZOOM], levels=[0.5], colors="red", linewidths=1.2)
                titre += f"\n{nom} : rouge = expert"
        ax.set_title(titre, fontsize=10)
        ax.axis("off")

    animation = FuncAnimation(fig, dessiner, frames=len(images))
    animation.save(DOSSIER_GIF / "suivi_coeur.gif", writer=PillowWriter(fps=8))
    plt.close(fig)

    # 5. Courbe de l'aire
    fig, ax = plt.subplots(figsize=(8, 4.5))
    numeros = np.arange(1, len(images) + 1)
    ax.plot(numeros, aires, "o-", color="#1f5fa8", label="notre contour (suivi)")
    ax.plot([t_ed + 1, t_es + 1], [vg_ed.sum() * mm2, vg_es.sum() * mm2], "s", color="red",
            markersize=9, label="expert (ED et ES)")
    for t_ref, nom in [(t_ed, "ED : coeur rempli"), (t_es, "ES : coeur contracte")]:
        ax.axvline(t_ref + 1, color="gray", linestyle=":")
        ax.text(t_ref + 1.3, ax.get_ylim()[1] * 0.97, nom, fontsize=9, va="top")
    ax.set_xlabel("image du battement (30 images = 1 battement)")
    ax.set_ylabel("aire du ventricule gauche (mm2)")
    ax.set_title(f"Aire du ventricule au cours du battement ({PATIENT}, coupe {coupe})\n"
                 f"variation entre ED et ES : {100 * variation:.0f} % (expert : {100 * variation_expert:.0f} %)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DOSSIER_GIF / "aire_ventricule.png", dpi=120)
    plt.close(fig)

    # 6. Comparaison avec l'expert en ED et ES
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, t_ref, zone, nom, d in [(axes[0], t_ed, vg_ed, "ED (coeur rempli)", dice_ed),
                                    (axes[1], t_es, vg_es, "ES (coeur contracte)", dice_es)]:
        ax.imshow(images[t_ref][ZOOM], cmap="gray")
        xs, ys = ferme(contours[t_ref])
        ax.plot(xs - x0, ys - y0, "-", color="cyan", linewidth=2, label="notre contour")
        ax.contour(zone[ZOOM], levels=[0.5], colors="red", linewidths=1.5)
        ax.plot([], [], "-", color="red", label="expert")
        ax.set_title(f"{nom} : Dice = {d:.3f}")
        ax.legend(loc="lower right", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Suivi sur le battement : comparaison avec l'expert ({PATIENT}, coupe {coupe})")
    fig.tight_layout()
    fig.savefig(DOSSIER_GIF / "comparaison_ED_ES.png", dpi=120)
    plt.close(fig)
    print(f"[ok] Resultats dans figures/{DOSSIER_GIF.name}/ "
          f"(suivi_coeur.gif, aire_ventricule.png, comparaison_ED_ES.png)")
