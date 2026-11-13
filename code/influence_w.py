"""
Influence du parametre w (sujet, Q4.1) sur toutes les images.

Pour chaque image, on garde les MEMES points de depart et on refait le chemin / contour avec
plusieurs valeurs de w. Seul w change : on voit donc uniquement son effet.

Rappel : P~ = P + w. w est le cout minimal de chaque pixel, donc il penalise la LONGUEUR :
    w petit -> le chemin suit fidelement les bords (meme avec des detours), mais aussi le bruit
    w grand -> le chemin devient court et lisse, il "coupe" les details

Quand une verite terrain existe, on mesure le resultat :
    IRM        -> F1-score (= Dice) avec la zone de l'expert
    oeil       -> F1-score avec le chemin de reference de l'expert
    labyrinthe -> nombre de pixels du chemin dans un mur (doit etre 0)

Resultats dans figures/influence_w/ : une figure par image (influence_w_xxx.png).
Le graphique resume.png (F1 moyen sur les deux bases de donnees) est fait par
influence_w_bases.py.
Lancer depuis le dossier Projet avec :
    python code/influence_w.py          (environ 3 minutes)
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import carte_cout as cc
import chemin as ch
import donnees_acdc as da
import donnees_drive as dd
import donnees_formes as df
import evaluation as ev
import pretraitement as pt
from parametres import DOSSIER_DONNEES, DOSSIER_FIGURES

VALEURS_W = [0.001, 0.01, 0.1, 0.5, 1.0]
DOSSIER_W = DOSSIER_FIGURES / "influence_w"


def cout_gradient(image, w, alpha, sigma=None):
    """Carte de cout 'bords' (Q1.3), avec un lissage gaussien optionnel."""
    if sigma:
        image = pt.lisser_gauss(image, sigma)
    gx, gy = cc.gradient(image)
    return cc.potentiel(cc.norme_gradient(gx, gy), w, alpha)


def charger_cas():
    """Les images testees : image, fonction 'cout(w)', points, zoom, et mesure eventuelle.
    Les points sont fixes (ce sont des points sur le bord, choisis une fois pour toutes)."""
    cas = []

    piece = df.charger_image(DOSSIER_DONNEES / "formes/piece.png")
    cas.append(dict(nom="piece", image=piece, cout=lambda w: cout_gradient(piece, w, 100.0),
                    points=[(177, 72), (101, 148), (177, 224)],
                    zoom=(slice(80, 280), slice(50, 250))))

    engrenage = df.charger_image(DOSSIER_DONNEES / "formes/engrenage.png")
    cas.append(dict(nom="engrenage", image=engrenage,
                    cout=lambda w: cout_gradient(engrenage, w, 100.0),
                    points=[(143, 199), (197, 82), (80, 33), (32, 140)], zoom=None))

    laby = df.reduire(df.charger_image(DOSSIER_DONNEES / "formes/labyrinthe.png"), 4)
    cas.append(dict(nom="labyrinthe", image=laby,
                    cout=lambda w: cc.potentiel_intensite(laby, w, clair=True),
                    points=[(289, 8), (272, 322)], zoom=None,
                    mesure=("pixels dans un mur", lambda c: int((laby[c[:, 0], c[:, 1]] < 0.5).sum()))))

    bc = df.reduire(df.charger_image(DOSSIER_DONNEES / "Paris/ButtesChaumont.png"), 2)
    cas.append(dict(nom="buttes_chaumont", image=bc,
                    cout=lambda w: cout_gradient(bc, w, 100.0, sigma=2.0),
                    points=[(100, 290), (75, 395), (150, 455), (195, 360), (300, 300), (230, 245)],
                    zoom=(slice(30, 340), slice(200, 500))))

    donnees = da.charger_patient(
        da.obtenir_dossier_patient(DOSSIER_DONNEES, "patient001", telecharger=False), "ED")
    irm, gt = da.extraire_coupe(donnees, da.coupe_du_milieu(donnees["gt"]))
    vg = gt == da.LABEL_VG
    cas.append(dict(nom="irm", image=irm, cout=lambda w: cout_gradient(irm, w, 10.0),
                    points=[(104, 107), (84, 118), (85, 141), (104, 152), (123, 141), (123, 118)],
                    zoom=(slice(70, 140), slice(95, 170)), verite=vg,
                    mesure=("F1", lambda c: ev.dice(ev.masque_interieur(c, irm.shape), vg))))

    oeil, vaisseaux, _ = dd.charger_drive(DOSSIER_DONNEES / "DRIVE", 39)
    reference_oeil = ev.chemin_reference(vaisseaux, (151, 167), (275, 75))
    cas.append(dict(nom="oeil_drive_39", image=oeil,
                    cout=lambda w: cc.potentiel_vaisseaux(oeil, w, 100.0),
                    points=[(151, 167), (275, 75)], zoom=(slice(110, 320), slice(30, 230)),
                    verite=vaisseaux,
                    mesure=("F1", lambda c: ev.f1_chemin(c, reference_oeil)[2])))
    return cas


def figure_cas(cas, contours):
    """Une ligne de panneaux : un par valeur de w, meme zoom, memes points."""
    zoom = cas["zoom"] or (slice(None), slice(None))
    y0, x0 = zoom[0].start or 0, zoom[1].start or 0
    ferme = len(cas["points"]) >= 3
    fig, axes = plt.subplots(1, len(VALEURS_W), figsize=(3.6 * len(VALEURS_W), 4.2))
    for ax, w, (contour, valeur) in zip(axes, VALEURS_W, contours):
        ax.imshow(cas["image"][zoom], cmap="gray")
        if cas.get("verite") is not None:
            ax.contour(cas["verite"][zoom], levels=[0.5], colors="red", linewidths=0.8)
        ys, xs = contour[:, 0] - y0, contour[:, 1] - x0
        if ferme:
            ys, xs = np.r_[ys, ys[0]], np.r_[xs, xs[0]]
        ax.plot(xs, ys, "-", color="cyan", linewidth=1.5)
        ax.plot([p[1] - x0 for p in cas["points"]], [p[0] - y0 for p in cas["points"]],
                "o", color="yellow", markersize=5, markeredgecolor="black")
        titre = f"w = {w}"
        if valeur is not None:
            nom_mesure = cas["mesure"][0]
            titre += f"\n{nom_mesure} : " + (f"{valeur:.3f}" if nom_mesure == "F1"
                                              else f"{valeur:.0f}")
        ax.set_title(titre, fontsize=10)
        ax.axis("off")
    legende = "cyan = notre resultat" + (", rouge = expert" if cas.get("verite") is not None else "")
    fig.suptitle(f"{cas['nom']} : influence de w ({legende})")
    fig.tight_layout(rect=(0, 0, 1, 0.9))          # place en haut pour le titre general
    return fig


if __name__ == "__main__":
    matplotlib.use("Agg")
    DOSSIER_W.mkdir(parents=True, exist_ok=True)
    for cas in charger_cas():
        contours = []
        for w in VALEURS_W:
            contour, _ = ch.contour_ferme(cas["cout"](w), cas["points"])
            valeur = cas["mesure"][1](contour) if "mesure" in cas else None
            contours.append((contour, valeur))
            print(f"{cas['nom']:16s} w = {w:<6} : {len(contour):5d} pixels"
                  + (f", {cas['mesure'][0]} = {valeur:.3g}" if valeur is not None else ""),
                  flush=True)
        fig = figure_cas(cas, contours)
        fig.savefig(DOSSIER_W / f"influence_w_{cas['nom']}.png", dpi=110)
        plt.close(fig)

    print(f"[ok] Figures dans figures/{DOSSIER_W.name}/")
