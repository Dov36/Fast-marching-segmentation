"""
Module donnees_acdc : fonctions pour trouver, ouvrir et afficher les IRM cardiaques ACDC.

Ce fichier ne fait rien tout seul : il contient seulement des fonctions.
Il ne fait aucun affichage (les figures sont dans affichage.py).
On les appelle depuis le programme principal (main.py) avec :
    import donnees_acdc as da

Bibliotheques necessaires :
    pip install nibabel numpy
    pip install huggingface_hub      (seulement pour le telechargement automatique)
"""

from pathlib import Path

import nibabel as nib
import numpy as np

# Valeurs des pixels dans la segmentation de l'expert (fichier *_gt.nii.gz)
LABELS = {0: "fond", 1: "ventricule droit", 2: "myocarde", 3: "ventricule gauche"}
LABEL_VG = 3  # ventricule gauche


# ----------------------------------------------------------------------------
# 1. Trouver ou telecharger les fichiers d'un patient
# ----------------------------------------------------------------------------
def trouver_dossier_patient(dossier_donnees, patient):
    """Cherche le dossier du patient (ex : 'patient001') dans dossier_donnees.

    La recherche se fait dans tous les sous-dossiers, donc cela marche :
      - avec le zip officiel ACDC   (ex : ACDC/database/training/patient001/)
      - avec le miroir Hugging Face (ex : ACDC/train/patient001/)
    Renvoie None si le patient n'est pas trouve.
    """
    dossier_donnees = Path(dossier_donnees)
    if not dossier_donnees.exists():
        return None
    for d in dossier_donnees.rglob(patient):
        if d.is_dir() and any(d.glob("*_frame??.nii.gz")):
            return d
    return None


def telecharger_patient(dossier_donnees, patient):
    """Telecharge un seul patient (environ 10 Mo) depuis le miroir Hugging Face de ACDC."""
    from huggingface_hub import snapshot_download

    print(f"[..] Telechargement de {patient} dans {dossier_donnees} ...")
    snapshot_download(
        repo_id="mathpluscode/ACDC",
        repo_type="dataset",
        local_dir=str(dossier_donnees),
        allow_patterns=[f"train/{patient}/*"],
    )
    return trouver_dossier_patient(dossier_donnees, patient)


def obtenir_dossier_patient(dossier_donnees, patient, telecharger=True):
    """Renvoie le dossier du patient ; le telecharge s'il est absent (si telecharger=True)."""
    d = trouver_dossier_patient(dossier_donnees, patient)
    if d is not None:
        print(f"[ok] {patient} trouve dans {d}")
        return d
    if not telecharger:
        raise FileNotFoundError(f"{patient} introuvable dans {dossier_donnees}")
    d = telecharger_patient(dossier_donnees, patient)
    if d is None:
        raise FileNotFoundError(f"Le telechargement de {patient} a echoue")
    print(f"[ok] {patient} telecharge dans {d}")
    return d


# ----------------------------------------------------------------------------
# 2. Choisir l'instant (ED ou ES) et ouvrir les fichiers NIfTI
# ----------------------------------------------------------------------------
def lire_info_cfg(dossier_patient):
    """Lit le fichier Info.cfg (present dans le zip officiel) : numeros des frames ED et ES."""
    f = Path(dossier_patient) / "Info.cfg"
    if not f.exists():
        return None
    info = {}
    for ligne in f.read_text().splitlines():
        if ":" in ligne:
            cle, valeur = ligne.split(":", 1)
            info[cle.strip()] = valeur.strip()
    return info


def trouver_fichiers(dossier_patient, instant="ED"):
    """Renvoie (fichier IRM, fichier segmentation expert) pour l'instant 'ED' ou 'ES'.

    ED = fin de diastole (coeur rempli, le plus grand).
    ES = fin de systole  (coeur contracte, le plus petit).
    Chaque patient a deux fichiers patientXXX_frameNN.nii.gz.
    Si Info.cfg existe on l'utilise, sinon : plus petit numero = ED, plus grand = ES.
    """
    dossier_patient = Path(dossier_patient)
    images = sorted(dossier_patient.glob("*_frame??.nii.gz"))
    if not images:
        raise FileNotFoundError(f"Aucune IRM dans {dossier_patient}")

    image = None
    info = lire_info_cfg(dossier_patient)
    if info and instant in info:
        numero = int(info[instant])
        candidats = [f for f in images if f.name.endswith(f"_frame{numero:02d}.nii.gz")]
        if candidats:
            image = candidats[0]
    if image is None:
        image = images[0] if instant == "ED" else images[-1]

    gt = image.with_name(image.name.replace(".nii.gz", "_gt.nii.gz"))
    if not gt.exists():
        raise FileNotFoundError(f"Segmentation introuvable : {gt.name}")
    return image, gt


def charger_volume(fichier):
    """Ouvre un fichier .nii.gz.

    Renvoie (volume, taille_voxel) :
      volume       : tableau numpy de taille (hauteur, largeur, nombre de coupes)
      taille_voxel : (mm en x, mm en y, mm entre deux coupes)
    """
    nii = nib.load(str(fichier))
    volume = np.asarray(nii.get_fdata())
    taille_voxel = tuple(float(z) for z in nii.header.get_zooms()[:3])
    return volume, taille_voxel


def charger_patient(dossier_patient, instant="ED"):
    """Ouvre l'IRM et la segmentation d'un patient pour un instant donne.

    Renvoie un dictionnaire avec : 'volume', 'gt', 'taille_voxel', 'fichier_image', 'fichier_gt'.
    """
    f_image, f_gt = trouver_fichiers(dossier_patient, instant)
    volume, voxel = charger_volume(f_image)
    gt, _ = charger_volume(f_gt)
    return {
        "volume": volume,
        "gt": np.rint(gt).astype(np.uint8),
        "taille_voxel": voxel,
        "fichier_image": f_image,
        "fichier_gt": f_gt,
    }


# ----------------------------------------------------------------------------
# 3. Choisir et preparer une coupe 2D
# ----------------------------------------------------------------------------
def coupes_avec_label(gt, label=LABEL_VG):
    """Liste des numeros de coupes ou l'expert a dessine la structure 'label'."""
    return [k for k in range(gt.shape[2]) if np.any(gt[:, :, k] == label)]


def coupe_du_milieu(gt, label=LABEL_VG):
    """Numero de la coupe au milieu du ventricule gauche (ni la base ni la pointe)."""
    coupes = coupes_avec_label(gt, label)
    if not coupes:
        return gt.shape[2] // 2
    return coupes[len(coupes) // 2]


def normaliser(image):
    """Ramene les intensites entre 0 et 1 (on ignore les 1 % de valeurs extremes)."""
    bas, haut = np.percentile(image, [1, 99])
    return np.clip((image - bas) / (haut - bas + 1e-12), 0, 1)


def extraire_coupe(donnees, numero):
    """Renvoie (image normalisee entre 0 et 1, segmentation expert) pour la coupe 'numero'."""
    image = normaliser(donnees["volume"][:, :, numero])
    gt = donnees["gt"][:, :, numero]
    return image, gt


def sauvegarder_coupe(fichier, image, gt, taille_pixel):
    """Sauvegarde une coupe dans un fichier .npz (a recharger avec charger_coupe)."""
    np.savez(fichier, image=image, gt=gt, taille_pixel=np.asarray(taille_pixel))


def charger_coupe(fichier):
    """Recharge une coupe sauvegardee : renvoie (image, gt, taille_pixel)."""
    d = np.load(fichier)
    return d["image"], d["gt"], d["taille_pixel"]
