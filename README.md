# Detection de contour par chemin minimal (Fast Marching)

Projet de traitement d'image (M1) : implementation de la methode de
Cohen & Kimmel (1997), *Global Minimum for Active Contour Models: A Minimal Path Approach*.
On clique des points sur un bord : le Fast Marching calcule la carte U (cout du chemin
le moins cher), puis la descente de U donne le chemin ou le contour ferme.
Applications : formes simples, labyrinthe, vues aeriennes de Paris, IRM cardiaque (ACDC,
avec score de Dice) et vaisseaux de la retine (DRIVE).

## Installation

    python -m pip install -r requirements.txt

## Lancer le projet

Depuis le dossier `Projet` (ou avec le bouton Run de VS Code) :

    python code/main_piece.py           # un fichier main par image : une fenetre interactive
    python code/main_engrenage.py       # s'ouvre, on clique des points sur un bord, et le
    python code/main_labyrinthe.py      # contour est trace (avec l'animation du Fast Marching)
    python code/main_buttes_chaumont.py
    python code/main_irm.py             # IRM : 6 clics autour du ventricule gauche -> Dice
    python code/main_drive.py           # retine : 2 clics sur un vaisseau -> % dans le vaisseau
    python code/main_dessin.py
    python code/main_paris.py
    python code/main_etoile.py

    python code/main.py                 # toutes les images, sans fenetre interactive
    python code/evaluer_drive.py        # oeil : evaluation automatique sur les 40 images DRIVE
                                        #   -> figures/oeil/ (environ 40 minutes)
    python code/suivi_coeur.py          # IRM cine : suivi du ventricule sur un battement
                                        #   -> figures/gif_coeur/ (GIF + aire, environ 2 minutes)
    python code/influence_w.py          # influence de w (Q4.1) sur toutes les images
                                        #   -> figures/influence_w/ (environ 3 minutes)

Fenetre interactive : clic gauche = ajouter un point, clic droit = enlever le dernier ;
le calcul demarre tout seul au dernier clic. 2 points -> chemin ouvert, 3 ou plus -> contour ferme.

Les reglages communs (w, alpha, lissage, nombre de clics...) sont dans `code/parametres.py` ;
certaines images ont leurs propres reglages dans leur `main_xxx.py`.
Les figures sont enregistrees dans `figures/<image>/`.

Tests automatiques (chacun affiche `[ok]` si tout va bien) :

    python code/tests/test_fast_marching.py
    python code/tests/test_chemin.py
    python code/tests/test_evaluation.py
    python code/tests/test_carte_cout.py
    python code/tests/test_pretraitement.py

## Organisation

```
Projet/
|-- code/
|   |-- main.py               lance toutes les images
|   |-- main_xxx.py           un fichier par image (piece, engrenage, labyrinthe, irm, drive...)
|   |-- evaluer_drive.py      evaluation automatique sur la base DRIVE (oeil)
|   |-- influence_w.py        influence du parametre w sur toutes les images (Q4.1)
|   |-- influence_w_bases.py  F1 en fonction de w sur les deux bases (IRM, oeil)
|   |-- suivi_coeur.py        suivi du ventricule pendant un battement (IRM cine, GIF)
|   |-- parametres.py         reglages communs (dossiers, w, alpha, lissage, clics...)
|   |-- traitement.py         chaine de traitement commune a toutes les images
|   |-- donnees_acdc.py       etape 1 : IRM ACDC et choix de la coupe
|   |-- donnees_formes.py     etape 1 : images simples (png, jpg)
|   |-- donnees_drive.py      etape 1 : images de retine DRIVE
|   |-- pretraitement.py      etape 2 : lissage (moyenneur, gaussien), moyenne locale
|   |-- carte_cout.py         etape 2 : gradient, cartes de cout (bords, intensite, vaisseaux)
|   |-- fast_marching.py      etape 3 : carte U (equation eikonale, Fast Marching)
|   |-- chemin.py             etape 4 : descente de U, chemin, contour ferme
|   |-- evaluation.py         etape 5 : Dice, part du chemin dans la verite terrain
|   |-- interactif.py         fenetre interactive (clics, animation, contour, evaluation)
|   |-- affichage.py          toutes les figures
|   `-- tests/                verifications automatiques
|-- donnees/
|   |-- patient001/           IRM ACDC + verite terrain
|   |-- DRIVE/                40 images de retine + vaisseaux de l'expert
|   |-- formes/               piece, engrenage, dessin, labyrinthe
|   `-- Paris/                vues aeriennes de Paris (dont les Buttes-Chaumont)
|-- figures/                  figures generees, un sous-dossier par image,
|                             + influence_w/ (Q4.1), oeil/ (evaluation sur DRIVE)
|                             et gif_coeur/ (battement du coeur)
|-- rapport/                  article, sujet, rapport
|-- README.md
`-- requirements.txt
```

## Donnees

Base ACDC (CREATIS, Lyon) : https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
Citation obligatoire : O. Bernard et al., "Deep Learning Techniques for Automatic MRI
Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?",
IEEE Transactions on Medical Imaging, 2018.

Base DRIVE : https://drive.grand-challenge.org/ (copie : https://github.com/sraashis/deepdyn)
Citation : J. Staal et al., "Ridge-based vessel segmentation in color images of the retina",
IEEE Transactions on Medical Imaging, 23(4), 2004.
"# Fast-marching-segmentation" 
