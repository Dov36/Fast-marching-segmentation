"""
Evaluation automatique sur toute la base DRIVE (vaisseaux de l'oeil) - robustesse, sujet Q4.3.

Pour chaque image, on tire au hasard des paires de points au coeur d'un vaisseau
(d'apres l'expert), on calcule le chemin minimal entre eux, et on mesure la part du chemin
qui reste dans les vaisseaux de l'expert. On compare plusieurs cartes de cout.

Pas de clic ici : tout est automatique (et reproductible grace a la graine du hasard).
Resultats dans le dossier figures/oeil/ :
    oeil_NN.png           une figure par image : les chemins trouves (vert = dans un vaisseau)
    resultats.csv         tous les chiffres (une ligne par chemin), lisible avec Excel
    resume.png            graphique : moyenne par carte de cout

Lancer depuis le dossier Projet avec :
    python code/evaluer_drive.py
Duree : environ 40 minutes pour les 40 images (reduire IMAGES pour aller plus vite).
"""

import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import carte_cout as cc
import chemin as ch
import donnees_drive as dd
import evaluation as ev
from parametres import DOSSIER_DONNEES, DOSSIER_FIGURES

IMAGES = range(1, 41)        # images testees (1 a 40 : toute la base DRIVE)
NB_PAIRES = 3                # paires de points par image
GRAINE = 0                   # graine du hasard (memes paires a chaque lancement)
DOSSIER_OEIL = DOSSIER_FIGURES / "oeil"
COUT_AFFICHE = "contraste local, a = 100"   # carte de cout dessinee dans les figures


def paires_au_hasard(vaisseaux, nb, rng):
    """Paires de points au coeur d'un vaisseau, distants de 80 a 200 pixels.
    'Au coeur' = le pixel et ses 4 voisins sont des pixels de vaisseau (vaisseau assez epais)."""
    coeur = vaisseaux.copy()
    coeur[1:, :] &= vaisseaux[:-1, :]
    coeur[:-1, :] &= vaisseaux[1:, :]
    coeur[:, 1:] &= vaisseaux[:, :-1]
    coeur[:, :-1] &= vaisseaux[:, 1:]
    ys, xs = np.nonzero(coeur)
    paires = []
    while len(paires) < nb:
        i, j = rng.integers(len(ys), size=2)
        a, b = (ys[i], xs[i]), (ys[j], xs[j])
        if 80 < np.hypot(a[0] - b[0], a[1] - b[1]) < 200:
            paires.append((a, b))
    return paires


def cartes_de_cout(image):
    """Les cartes de cout comparees (meme w = 0.01 pour toutes)."""
    gx, gy = cc.gradient(image)
    return {
        "gradient (bords)":         cc.potentiel(cc.norme_gradient(gx, gy), 0.01, 100.0),
        "intensite (sombre)":       cc.potentiel_intensite(image, 0.01, clair=False),
        "contraste local, a = 30":  cc.potentiel_vaisseaux(image, 0.01, 30.0),
        "contraste local, a = 100": cc.potentiel_vaisseaux(image, 0.01, 100.0),
    }


def figure_image(image, vaisseaux, chemins, scores, titre):
    """Une image de l'oeil : vaisseaux de l'expert en clair, chemins trouves par dessus."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(vaisseaux, cmap="gray", vmin=0, vmax=1.5)
    ax.imshow(image, cmap="gray", alpha=0.35)
    for chemin in chemins:
        dedans = vaisseaux[chemin[:, 0], chemin[:, 1]]
        ax.plot(chemin[dedans, 1], chemin[dedans, 0], ".", color="lime", markersize=2)
        ax.plot(chemin[~dedans, 1], chemin[~dedans, 0], ".", color="red", markersize=3)
        ax.plot([chemin[0, 1], chemin[-1, 1]], [chemin[0, 0], chemin[-1, 0]], "o",
                color="yellow", markersize=5, markeredgecolor="black", linestyle="")
    ax.set_title(f"{titre}\n" + ", ".join(f"{100 * s:.0f} %" for s in scores)
                 + " du chemin dans un vaisseau", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    matplotlib.use("Agg")                     # pas de fenetre : on enregistre seulement
    DOSSIER_OEIL.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(GRAINE)
    lignes = []                               # une ligne par (image, chemin, carte de cout)
    for numero in IMAGES:
        image, vaisseaux, zone = dd.charger_drive(DOSSIER_DONNEES / "DRIVE", numero)
        paires = paires_au_hasard(vaisseaux, NB_PAIRES, rng)
        chemins_affiches, scores_affiches = [], []
        for nom, cout in cartes_de_cout(image).items():
            for k, (depart, arrivee) in enumerate(paires):
                chemin, _ = ch.chemin_minimal(cout, depart, arrivee)
                score = ev.part_dans_verite(chemin, vaisseaux)
                lignes.append([dd.nom_image(numero), k + 1, nom, len(chemin), round(score, 4)])
                if nom == COUT_AFFICHE:
                    chemins_affiches.append(chemin)
                    scores_affiches.append(score)
        fig = figure_image(image, vaisseaux, chemins_affiches, scores_affiches,
                           f"DRIVE {dd.nom_image(numero)} ({COUT_AFFICHE})")
        fig.savefig(DOSSIER_OEIL / f"oeil_{numero:02d}.png", dpi=100)
        plt.close(fig)
        print(f"[ok] image {dd.nom_image(numero)} terminee", flush=True)

    # Tableau de tous les resultats
    with open(DOSSIER_OEIL / "resultats.csv", "w", newline="") as f:
        ecrivain = csv.writer(f, delimiter=";")
        ecrivain.writerow(["image", "chemin", "carte de cout", "longueur (pixels)",
                           "part dans un vaisseau"])
        ecrivain.writerows(lignes)

    # Resume : moyenne et pire cas par carte de cout
    noms = list(dict.fromkeys(l[2] for l in lignes))
    moyennes = [100 * np.mean([l[4] for l in lignes if l[2] == n]) for n in noms]
    pires = [100 * np.min([l[4] for l in lignes if l[2] == n]) for n in noms]
    parfaits = [100 * np.mean([l[4] >= 0.99 for l in lignes if l[2] == n]) for n in noms]
    nb = len(lignes) // len(noms)
    print(f"\nPart du chemin dans les vaisseaux de l'expert ({len(IMAGES)} images x "
          f"{NB_PAIRES} chemins = {nb} chemins par carte de cout) :")
    for n, m, p, q in zip(noms, moyennes, pires, parfaits):
        print(f"    {n:26s} : moyenne {m:5.1f} %, pire cas {p:5.1f} %, "
              f"chemins a plus de 99 % : {q:5.1f} %")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    barres = ax.bar(noms, moyennes, color=["#999999", "#999999", "#5b9bd5", "#1f5fa8"])
    for b, m, p in zip(barres, moyennes, pires):
        ax.text(b.get_x() + b.get_width() / 2, m + 1, f"{m:.1f} %\n(pire {p:.0f} %)",
                ha="center", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("part du chemin dans un vaisseau (%)")
    ax.set_title(f"Base DRIVE : {len(IMAGES)} images, {nb} chemins par carte de cout")
    fig.tight_layout()
    fig.savefig(DOSSIER_OEIL / "resume.png", dpi=120)
    print(f"[ok] Resultats dans figures/{DOSSIER_OEIL.name}/ (resume.png, resultats.csv, "
          f"oeil_NN.png)")
