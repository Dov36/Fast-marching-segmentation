"""
Influence de w (sujet, Q4.1) mesuree sur les deux BASES DE DONNEES, avec la meme mesure : le F1.

    IRM (ACDC, patient001) : toutes les coupes ou l'expert a dessine le ventricule gauche,
                             en fin de diastole (ED) et en fin de systole (ES).
                             F1 = Dice entre notre contour (rempli) et la zone de l'expert.
    Oeil (DRIVE)          : plusieurs images, 2 chemins par image.
                             F1 entre notre chemin et le chemin de reference de l'expert
                             (plus court chemin qui reste dans ses vaisseaux).

Les points sont places AUTOMATIQUEMENT (pour que l'experience soit reproductible) :
    IRM  : 6 points repartis sur le bord de la zone de l'expert (comme 6 clics bien places)
    oeil : 2 points tires au hasard au coeur d'un vaisseau de l'expert
Pour chaque cas, on refait le calcul avec plusieurs w : seul w change.

Resultat : figures/influence_w/resume.png (F1 moyen en fonction de w, pour les deux bases)
Lancer depuis le dossier Projet avec :
    python code/influence_w_bases.py      (environ 15 minutes)
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import carte_cout as cc
import chemin as ch
import donnees_acdc as da
import donnees_drive as dd
import evaluation as ev
from evaluer_drive import paires_au_hasard
from parametres import DOSSIER_DONNEES, DOSSIER_FIGURES

VALEURS_W = [0.001, 0.01, 0.1, 0.5, 1.0]
IMAGES_OEIL = [1, 6, 11, 16, 21, 26, 31, 36]   # 8 images reparties dans la base DRIVE
PAIRES_PAR_IMAGE = 2
GRAINE = 1


def points_sur_le_bord(zone, nb):
    """nb points repartis regulierement (en angle) sur le bord de la zone de l'expert."""
    bord = zone & ev.elargir(~zone, 1)                # pixels de la zone qui touchent le dehors
    ys, xs = np.nonzero(bord)
    cy, cx = ys.mean(), xs.mean()
    angles = np.arctan2(ys - cy, xs - cx)
    points = []
    for a in np.linspace(-np.pi, np.pi, nb, endpoint=False):
        k = np.argmin(np.abs(np.angle(np.exp(1j * (angles - a)))))
        points.append((ys[k], xs[k]))
    return points


def f1_irm():
    """F1 (= Dice) pour chaque coupe de l'IRM et chaque w. Renvoie un tableau (coupes, w)."""
    dossier = da.obtenir_dossier_patient(DOSSIER_DONNEES, "patient001", telecharger=False)
    scores = []
    for instant in ["ED", "ES"]:
        donnees = da.charger_patient(dossier, instant)
        for k in da.coupes_avec_label(donnees["gt"]):
            image, gt = da.extraire_coupe(donnees, k)
            zone = gt == da.LABEL_VG
            if zone.sum() < 50:                           # ventricule trop petit (pointe du coeur)
                continue
            points = points_sur_le_bord(zone, 6)
            gx, gy = cc.gradient(image)
            norme = cc.norme_gradient(gx, gy)
            ligne = []
            for w in VALEURS_W:
                contour, _ = ch.contour_ferme(cc.potentiel(norme, w, 10.0), points)
                ligne.append(ev.dice(ev.masque_interieur(contour, image.shape), zone))
            scores.append(ligne)
            print(f"IRM {instant} coupe {k:2d} : F1 = " + ", ".join(f"{s:.3f}" for s in ligne),
                  flush=True)
    return np.array(scores)


def f1_oeil():
    """F1 de chaque chemin de l'oeil et chaque w. Renvoie un tableau (chemins, w)."""
    rng = np.random.default_rng(GRAINE)
    scores = []
    for numero in IMAGES_OEIL:
        image, vaisseaux, _ = dd.charger_drive(DOSSIER_DONNEES / "DRIVE", numero)
        for depart, arrivee in paires_au_hasard(vaisseaux, PAIRES_PAR_IMAGE, rng):
            reference = ev.chemin_reference(vaisseaux, depart, arrivee)
            ligne = []
            for w in VALEURS_W:
                chemin, _ = ch.chemin_minimal(cc.potentiel_vaisseaux(image, w, 100.0),
                                              depart, arrivee)
                ligne.append(ev.f1_chemin(chemin, reference)[2])
            scores.append(ligne)
            print(f"Oeil {dd.nom_image(numero)} : F1 = " + ", ".join(f"{s:.3f}" for s in ligne),
                  flush=True)
    return np.array(scores)


if __name__ == "__main__":
    matplotlib.use("Agg")
    irm = f1_irm()
    oeil = f1_oeil()

    fig, ax = plt.subplots(figsize=(8, 5))
    for scores, nom, couleur in [(irm, f"IRM cardiaque ACDC ({len(irm)} coupes)", "#c0392b"),
                                 (oeil, f"Oeil DRIVE ({len(oeil)} chemins, "
                                        f"{len(IMAGES_OEIL)} images)", "#1f5fa8")]:
        moyenne, ecart = scores.mean(axis=0), scores.std(axis=0)
        ax.plot(VALEURS_W, moyenne, "o-", color=couleur, label=nom)
        ax.fill_between(VALEURS_W, moyenne - ecart, np.minimum(moyenne + ecart, 1),
                        color=couleur, alpha=0.15)
        for w, m in zip(VALEURS_W, moyenne):
            ax.annotate(f"{m:.3f}", (w, m), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=couleur)
    ax.set_xscale("log")
    ax.set_xlabel("w : parametre de regularisation (penalise la longueur) - echelle log")
    ax.set_ylabel("F1-score moyen (1 = identique a l'expert)")
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("Influence de w sur les bases de donnees (zone coloree = +/- 1 ecart-type)")
    fig.tight_layout()
    dossier = DOSSIER_FIGURES / "influence_w"
    dossier.mkdir(parents=True, exist_ok=True)
    fig.savefig(dossier / "resume.png", dpi=120)

    print("\nF1 moyen (ecart-type) :")
    for nom, scores in [("IRM ", irm), ("Oeil", oeil)]:
        print(f"    {nom} : " + "  ".join(f"w={w}: {m:.3f} ({s:.3f})" for w, m, s in
                                       zip(VALEURS_W, scores.mean(axis=0), scores.std(axis=0))))
    print(f"[ok] Figure : figures/influence_w/resume.png")
