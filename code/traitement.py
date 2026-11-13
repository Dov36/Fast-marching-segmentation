"""
Module traitement : la chaine de traitement COMMUNE a toutes les images.

Chaque fichier main_xxx.py charge ses images, puis appelle traiter_image().
Ainsi le traitement (lissage, gradient, carte de cout, puis Fast Marching...) n'est
ecrit qu'une seule fois, ici, et s'applique de la meme facon a toutes les images.

    Etape 2 : lissage + carte de cout    -> pretraitement.py, carte_cout.py
    Etapes 3 a 5, dans la fenetre interactive (interactif.py) :
    Etape 3 : Fast Marching              -> fast_marching.py
    Etape 4 : backtracking + contour     -> chemin.py
    Etape 5 : evaluation (Dice, ou part du chemin dans la verite) -> evaluation.py
    Figures                              -> affichage.py
"""

import matplotlib.pyplot as plt

import affichage as af
import carte_cout as cc
from donnees_acdc import LABEL_VG
import interactif as ia
import pretraitement as pt
from parametres import ALPHA as ALPHA_DEFAUT, LISSAGE as LISSAGE_DEFAUT
from parametres import SIGMA as SIGMA_DEFAUT, W as W_DEFAUT
from parametres import (DOSSIER_FIGURES, FIGURES_DETAILLEES, INTERACTIF, TAILLE_MOYENNE,
                        TOUT)


def figure_secondaire(fig, fichier):
    """Enregistre une figure secondaire, puis ferme sa fenetre si FIGURES_DETAILLEES vaut False
    (la figure reste disponible dans le dossier figures/)."""
    af.enregistrer_figure(fig, fichier)
    if not FIGURES_DETAILLEES:
        plt.close(fig)


def dossier_figures(nom_image):
    """Renvoie le dossier figures/<nom_image>/ (et le cree s'il n'existe pas)."""
    dossier = DOSSIER_FIGURES / nom_image
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier


def traiter_image(image, nom_image, nom=None, titre=None, zoom=None, gt=None, interactif=False,
                  type_cout="gradient", nb_points=None, lissage=None, sigma=None,
                  w=None, alpha=None, verite=None):
    """Applique toute la chaine de traitement a une image en niveaux de gris (0 a 1).

    nom_image : nom court de l'image ("piece", "paris", "irm"...) -> dossier figures/<nom_image>/
    nom       : debut du nom des fichiers de figures (par defaut nom_image)
    titre     : titre des figures (par defaut nom_image)
    zoom      : zone a agrandir dans les figures secondaires (lignes, colonnes) ;
                par defaut l'image entiere
    gt        : verite terrain (seulement pour l'IRM), utilisee a l'etape 5 uniquement
    interactif: True -> a la fin, fenetre interactive du Fast Marching (si INTERACTIF = True)
    type_cout : "gradient" -> chemin le long des BORDS (cas general, Q1.3)
                "clair"    -> chemin dans les zones CLAIRES (couloirs d'un labyrinthe)
                "sombre"   -> chemin dans les zones SOMBRES (route sombre, trait noir)
                "vaisseaux"-> chemin le long d'un vaisseau sombre (contraste local, DRIVE)
    nb_points : nombre de clics dans la fenetre interactive (par defaut NB_POINTS)
    lissage, sigma, w, alpha : remplacent LISSAGE, SIGMA, W et ALPHA de parametres.py
                               pour cette image seulement
    verite    : zone de l'expert deja prete (tableau de booleens), pour les images autres que
                l'IRM (ex. vaisseaux de DRIVE). Pour l'IRM, on donne gt (label 3 = ventricule)
    Renvoie un dictionnaire avec les resultats intermediaires.
    """
    nom = nom or nom_image
    LISSAGE = lissage or LISSAGE_DEFAUT          # reglages propres a cette image, sinon parametres.py
    SIGMA = sigma or SIGMA_DEFAUT
    W = w or W_DEFAUT
    ALPHA = alpha or ALPHA_DEFAUT
    titre = titre or nom_image
    dossier_fig = dossier_figures(nom_image)
    hauteur, largeur = image.shape
    if zoom is None:
        zoom = TOUT

    print(f"\n========== {titre} ==========")
    print(f"Image                : {hauteur} x {largeur} pixels, {image.ndim} dimensions "
          f"-> niveaux de gris (1 canal)")
    print(f"Valeurs des pixels   : entre {image.min():.2f} et {image.max():.2f} ({image.dtype})")

    # Figure PRINCIPALE : l'image seule, sans contour
    af.afficher_image(image, titre)
    af.sauvegarder_png(dossier_fig / f"{nom}.png", image)

    # ------------------------------------------------------------------------
    # Etape 2 : carte de cout                              (sujet, partie 1)
    # ------------------------------------------------------------------------
    # Q1.1 - Lissage : comparaison des deux filtres lineaires (bruit granuleux, cf. TP2 ex. 3)
    image_moyenne = pt.lisser_moyenne(image, TAILLE_MOYENNE)
    image_gauss = pt.lisser_gauss(image, SIGMA)

    fig_lissage = af.afficher_comparaison(
        [image, image_moyenne, image_gauss],
        ["Originale", f"Moyenneur {TAILLE_MOYENNE}x{TAILLE_MOYENNE}", f"Gaussien sigma = {SIGMA}"],
        f"{titre} : comparaison des lissages", zoom=zoom)
    figure_secondaire(fig_lissage, dossier_fig / f"{nom}_lissages.png")

    # Image lissee utilisee pour la suite (choix avec le parametre LISSAGE)
    image_lissee = {"gauss": image_gauss, "moyenne": image_moyenne, "aucun": image}[LISSAGE]
    print(f"Lissage utilise      : {LISSAGE}")

    # Q1.2 - Gradient (filtres de Sobel)
    gx, gy = cc.gradient(image_lissee)
    norme = cc.norme_gradient(gx, gy)
    print(f"Norme du gradient    : max {norme.max():.2f}")

    fig_gradient = af.afficher_gradient(
        gx, gy, norme, f"{titre} : gradient (lissage : {LISSAGE})", zoom=zoom)
    figure_secondaire(fig_gradient, dossier_fig / f"{nom}_gradient_{LISSAGE}.png")

    # Q1.3 - Potentiel (carte de cout P~) -> figure PRINCIPALE
    if type_cout == "gradient":
        cout = cc.potentiel(norme, W, ALPHA)
        reglage = f"w = {W}, alpha = {ALPHA}"
    elif type_cout == "vaisseaux":     # contraste local : plus sombre que ses voisins
        cout = cc.potentiel_vaisseaux(image_lissee, W, ALPHA)
        reglage = f"vaisseaux (contraste local), w = {W}, alpha = {ALPHA}"
    else:                    # cout base sur l'intensite (labyrinthe, route...)
        cout = cc.potentiel_intensite(image_lissee, W, clair=(type_cout == "clair"))
        reglage = f"intensite, zones {type_cout}s bon marche, w = {W}"
    print(f"Carte de cout P~     : min {cout.min():.3f}, max {cout.max():.3f} ({reglage})")

    fig_cout = af.afficher_potentiel(
        cout, f"{titre} : carte de cout (lissage : {LISSAGE}, {reglage})")
    af.enregistrer_figure(fig_cout, dossier_fig / f"{nom}_cout_{LISSAGE}.png")

    # Test : carte de cout calculee a partir des trois images (sans lissage, moyenneur, gaussien)
    couts = {}
    for nom_lissage, img in [("aucun", image), ("moyenne", image_moyenne), ("gauss", image_gauss)]:
        gx_l, gy_l = cc.gradient(img)
        couts[nom_lissage] = cc.potentiel(cc.norme_gradient(gx_l, gy_l), W, ALPHA)

    fig_couts = af.afficher_comparaison_couts(
        couts, f"{titre} : carte de cout selon le lissage (w = {W}, alpha = {ALPHA})", zoom=zoom)
    figure_secondaire(fig_couts, dossier_fig / f"{nom}_couts_comparaison.png")

    # ------------------------------------------------------------------------
    # Etapes 3, 4 et 5 : Fast Marching, backtracking, evaluation  (sujet, parties 2 a 4)
    # ------------------------------------------------------------------------
    # Fenetre interactive : clics -> Fast Marching -> descente de U -> contour -> evaluation
    # (si une verite terrain est connue : IRM avec gt, DRIVE avec verite). Voir interactif.py.
    if gt is not None:
        verite = gt == LABEL_VG                   # IRM : zone du ventricule gauche (label 3)
    contour = None
    if interactif and INTERACTIF:
        contour = ia.contour_interactif(image, cout, titre, dossier=dossier_fig, nom=nom,
                                        nb_points=nb_points, verite=verite, zoom=zoom)

    print(f"[ok] Figures enregistrees dans figures/{nom_image}/"
          + ("" if FIGURES_DETAILLEES else " (affichees : image et carte de cout)"))
    return dict(image_lissee=image_lissee, gx=gx, gy=gy, norme=norme, cout=cout,
                contour=contour)
