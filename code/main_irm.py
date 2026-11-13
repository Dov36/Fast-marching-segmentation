"""
Programme : IRM cardiaque ACDC (contour du ventricule gauche).

Lancer depuis le dossier Projet avec :
    python code/main_irm.py
(ou depuis VS Code avec le bouton Run). Les figures vont dans figures/irm/.
"""

import matplotlib.pyplot as plt

import affichage as af
import donnees_acdc as da
import traitement as tr
from parametres import DOSSIER_DONNEES

# Choix de l'IRM
PATIENT = "patient001"  # patient001 ... patient100
INSTANT = "ED"          # "ED" = fin de diastole, "ES" = fin de systole
COUPE = None            # numero de coupe ; None = coupe du milieu automatiquement
TELECHARGER = False     # True : telecharge le patient s'il n'est pas dans donnees/

# Zone zoomee autour du ventricule gauche (figures secondaires)
ZOOM_IRM = (slice(70, 140), slice(95, 170))


def lancer(afficher=True):
    """Charge l'IRM, affiche les figures propres a l'IRM, puis applique le traitement commun.
    afficher=False : les figures sont enregistrees puis fermees (pas de fenetres)."""
    dossier_fig = tr.dossier_figures("irm")

    dossier_patient = da.obtenir_dossier_patient(DOSSIER_DONNEES, PATIENT, telecharger=TELECHARGER)
    donnees = da.charger_patient(dossier_patient, INSTANT)
    af.afficher_infos(donnees)

    coupe = COUPE if COUPE is not None else da.coupe_du_milieu(donnees["gt"])
    print(f"Coupe choisie        : {coupe}")
    image, gt = da.extraire_coupe(donnees, coupe)
    nom = f"{PATIENT}_{INSTANT}_coupe{coupe}"
    titre = f"{PATIENT} - {INSTANT} - coupe {coupe}"

    # Figure : toutes les coupes du volume, la coupe choisie encadree en jaune
    fig_coupes = af.afficher_toutes_les_coupes(
        donnees, f"{PATIENT} - {INSTANT} - toutes les coupes (base -> pointe)", coupe_choisie=coupe)
    tr.figure_secondaire(fig_coupes, dossier_fig / f"{PATIENT}_{INSTANT}_toutes_les_coupes.png")

    # Figure : IRM + segmentation de l'expert + contour expert
    fig_expert = af.afficher_coupe(image, gt, titre)
    tr.figure_secondaire(fig_expert, dossier_fig / f"{nom}_verite_terrain.png")

    # Sauvegarde de la coupe (valeurs exactes) dans le dossier du patient
    fichier_coupe = dossier_patient / f"{nom}.npz"
    da.sauvegarder_coupe(fichier_coupe, image, gt, donnees["taille_voxel"][:2])
    print(f"[ok] Coupe sauvegardee dans {dossier_patient.name}/{fichier_coupe.name}")

    # Traitement commun a toutes les images
    # Reglages propres a l'IRM (mesures avec le Dice) : w et alpha de l'article, 6 clics
    tr.traiter_image(image, "irm", nom=nom, titre=titre, zoom=ZOOM_IRM, gt=gt,
                     interactif=afficher, w=0.1, alpha=10.0, lissage="aucun", nb_points=6)
    if not afficher:
        plt.close("all")


if __name__ == "__main__":
    lancer()
    plt.show()
