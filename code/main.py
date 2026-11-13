"""
Programme principal : lance toutes les images, dans l'ordre
    1. Paris          : main_paris.py, main_etoile.py
    2. Objets         : main_piece.py, main_engrenage.py, main_dessin.py
    3. IRM cardiaque  : main_irm.py

Chaque image peut aussi etre lancee seule (python code/main_piece.py, par exemple).
Le traitement commun est dans traitement.py, les reglages dans parametres.py.

Lancer depuis le dossier Projet avec :
    python code/main.py
(ou depuis VS Code avec le bouton Run). Les figures vont dans figures/<image>/.
"""

import matplotlib.pyplot as plt

import main_dessin
import main_drive
import main_engrenage
import main_etoile
import main_irm
import main_buttes_chaumont
import main_labyrinthe
import main_paris
import main_piece

# True  : garde toutes les fenetres de figures ouvertes a la fin (il y en a beaucoup !)
# False : les figures sont seulement enregistrees dans figures/ (recommande ici)
AFFICHER = False

for programme in [main_paris, main_etoile, main_buttes_chaumont,  # 1. Paris
                  main_piece, main_engrenage, main_dessin,  # 2. Objets
                  main_labyrinthe,
                  main_irm, main_drive]:                    # 3. Imagerie medicale
    programme.lancer(afficher=AFFICHER)

print("\n[ok] Tout est termine. Les figures sont dans le dossier figures/.")
if AFFICHER:
    plt.show()
