"""
Figure pour le rapport : les cartes de cout utilisees pour chaque application.

Une colonne par application : en haut l'image, en bas sa carte de cout P~
(sombre = bon marche = la ou le chemin a envie de passer).
    engrenage   : gradient (les bords)
    labyrinthe  : intensite (couloirs clairs bon marche)
    parc        : gradient apres lissage gaussien (image aerienne texturee)
    IRM         : gradient (bords du ventricule)
    oeil        : contraste local (vaisseau plus sombre que son voisinage)

Resultats : figures/cartes_cout/cartes_cout.png, engrenage_image_cout.png et irm_cout.png
Lancer depuis le dossier Projet avec :
    python code/figure_cartes_cout.py      (quelques secondes)
"""

import matplotlib
import matplotlib.pyplot as plt

import carte_cout as cc
import donnees_acdc as da
import donnees_drive as dd
import donnees_formes as df
from parametres import DOSSIER_DONNEES, DOSSIER_FIGURES


def cout_gradient(image, w, alpha):
    gx, gy = cc.gradient(image)
    return cc.potentiel(cc.norme_gradient(gx, gy), w, alpha)


if __name__ == "__main__":
    matplotlib.use("Agg")
    engrenage = df.charger_image(DOSSIER_DONNEES / "formes/engrenage.png")
    laby = df.reduire(df.charger_image(DOSSIER_DONNEES / "formes/labyrinthe.png"), 4)
    donnees = da.charger_patient(
        da.obtenir_dossier_patient(DOSSIER_DONNEES, "patient001", telecharger=False), "ED")
    irm, _ = da.extraire_coupe(donnees, da.coupe_du_milieu(donnees["gt"]))
    oeil, _, _ = dd.charger_drive(DOSSIER_DONNEES / "DRIVE", 21)

    zoom_irm = (slice(60, 150), slice(85, 180))
    cas = [
        ("Engrenage\ngradient (bords)", engrenage, cout_gradient(engrenage, 0.01, 100.0), None),
        ("Labyrinthe\nintensite (clair = bon marche)", laby,
         cc.potentiel_intensite(laby, 0.01, clair=True), None),
        ("IRM (ventricule gauche)\ngradient (bords)", irm, cout_gradient(irm, 0.1, 10.0), zoom_irm),
        ("Oeil (DRIVE)\ncontraste local (vaisseaux)", oeil,
         cc.potentiel_vaisseaux(oeil, 0.01, 100.0), (slice(60, 380), slice(150, 470))),
    ]

    fig, axes = plt.subplots(2, len(cas), figsize=(3.2 * len(cas), 6.6))
    for k, (titre, image, cout, zoom) in enumerate(cas):
        zoom = zoom or (slice(None), slice(None))
        axes[0, k].imshow(image[zoom], cmap="gray")
        axes[0, k].set_title(titre, fontsize=9)
        im = axes[1, k].imshow(cout[zoom], cmap="gray")
        fig.colorbar(im, ax=axes[1, k], fraction=0.046, pad=0.03)
        for ax in axes[:, k]:
            ax.axis("off")
    axes[0, 0].text(-0.08, 0.5, "image", transform=axes[0, 0].transAxes, rotation=90,
                    ha="right", va="center", fontsize=10)
    axes[1, 0].text(-0.08, 0.5, "carte de cout P~", transform=axes[1, 0].transAxes, rotation=90,
                    ha="right", va="center", fontsize=10)
    fig.suptitle("Cartes de cout (sombre = bon marche : le chemin y passe)", fontsize=11)
    fig.tight_layout()
    dossier = DOSSIER_FIGURES / "cartes_cout"
    dossier.mkdir(parents=True, exist_ok=True)
    fig.savefig(dossier / "cartes_cout.png", dpi=130)
    print(f"[ok] Figure : figures/cartes_cout/cartes_cout.png")

    # Figure du rapport : l'engrenage seul, image et carte de cout cote a cote
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    axes[0].imshow(engrenage, cmap="gray")
    axes[0].set_title("Image de l'engrenage", fontsize=10)
    im = axes[1].imshow(cas[0][2], cmap="gray")
    axes[1].set_title("Carte de cout P~ (sombre = bon marche)", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(dossier / "engrenage_image_cout.png", dpi=130)
    print(f"[ok] Figure : figures/cartes_cout/engrenage_image_cout.png")

    # Figure du rapport : la carte de cout de l'IRM, avec le meme zoom que les figures de main_irm.py
    zoom_ventricule = (slice(70, 140), slice(95, 170))
    fig, ax = plt.subplots(figsize=(4.5, 4.7))
    ax.imshow(cas[2][2][zoom_ventricule], cmap="gray")
    ax.set_title("Carte de cout P~\n(sombre = bon marche)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(dossier / "irm_cout.png", dpi=130)
    print(f"[ok] Figure : figures/cartes_cout/irm_cout.png")
