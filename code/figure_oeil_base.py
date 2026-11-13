"""
Figure pour le rapport : l'image 21 de la base DRIVE, telle qu'on la recoit.

Quatre panneaux :
    1. l'image de fond d'oeil en couleur (donnee d'entree)
    2. les vaisseaux traces par l'expert (verite terrain, sert seulement a evaluer)
    3. la carte de cout P~ par contraste local (ce que voit le Fast Marching)
    4. trois chemins minimaux entre des points tires au hasard dans les vaisseaux
       (vert = dans un vaisseau de l'expert, rouge = dehors)

Resultat : figures/oeil/oeil_21_base.png
Lancer depuis le dossier Projet avec :
    python code/figure_oeil_base.py      (environ 30 secondes)
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import carte_cout as cc
import chemin as ch
import donnees_drive as dd
import evaluation as ev
from evaluer_drive import paires_au_hasard
from parametres import DOSSIER_DONNEES, DOSSIER_FIGURES

NUMERO = 21
NB_CHEMINS = 3
GRAINE = 3

if __name__ == "__main__":
    matplotlib.use("Agg")
    dossier = DOSSIER_DONNEES / "DRIVE"
    couleur = np.asarray(Image.open(dossier / "images" / f"{dd.nom_image(NUMERO)}.tif"))
    image, vaisseaux, _ = dd.charger_drive(dossier, NUMERO)
    cout = cc.potentiel_vaisseaux(image, 0.01, 100.0)

    rng = np.random.default_rng(GRAINE)
    chemins = [ch.chemin_minimal(cout, a, b)[0]
               for a, b in paires_au_hasard(vaisseaux, NB_CHEMINS, rng)]
    dans_vaisseau = ev.elargir(vaisseaux, 1)       # tolerance de 1 pixel, comme l'evaluation

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    panneaux = [(couleur, None, "Image de fond d'oeil (entree)"),
                (vaisseaux, "gray", "Vaisseaux traces par l'expert"),
                (cout, "gray", "Carte de cout P~ (contraste local)"),
                (image, "gray", "Nos chemins vs expert (vert = juste)")]
    for ax, (donnee, cmap, titre) in zip(axes, panneaux):
        ax.imshow(donnee, cmap=cmap)
        ax.set_title(titre, fontsize=12)
        ax.axis("off")
    for chemin in chemins:
        dedans = dans_vaisseau[chemin[:, 0], chemin[:, 1]]
        axes[3].plot(chemin[dedans, 1], chemin[dedans, 0], ".", color="lime", markersize=2)
        axes[3].plot(chemin[~dedans, 1], chemin[~dedans, 0], ".", color="red", markersize=3)
        axes[3].plot([chemin[0, 1], chemin[-1, 1]], [chemin[0, 0], chemin[-1, 0]], "o",
                     color="yellow", markeredgecolor="black", markersize=5)
        print(f"chemin : {100 * dedans.mean():.1f} % dans un vaisseau")
    fig.tight_layout()
    sortie = DOSSIER_FIGURES / "oeil" / "oeil_21_base.png"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sortie, dpi=110, bbox_inches="tight")
    print(f"[ok] Figure : figures/oeil/oeil_21_base.png")
