#!/usr/bin/env python
import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image

from pretraitement_des_images import Pretraitement
from pretraitement_des_attributs import preprocess_attributes


N_IMAGES = 202599  # nombre total d'images CelebA

 
HERE = os.path.dirname(os.path.abspath(__file__))  # Projet_MLA/data
ROOT = os.path.dirname(HERE)                      # Projet_MLA/

# Dossier contenant CelebA
CELEBA_DIR = os.path.join(ROOT, "CelebA", "CelebA") 

# Dossier images brutes
RAW_IMG_DIR = os.path.join(CELEBA_DIR, "images") 

# Dossier images prétraitées
PREPROC_IMG_DIR = 'Data_preprocessed/Images_Preprocessed' 
 


def preprocess_all_images():
    
  

    os.makedirs(PREPROC_IMG_DIR, exist_ok=True)

    preproc = Pretraitement()

    # Liste des images brutes
    files = sorted([
        f for f in os.listdir(RAW_IMG_DIR)
        if f.lower().endswith(".jpg")
    ])

    print(f"{len(files)} images trouvées dans {RAW_IMG_DIR}")
    print("Début du prétraitement...")

    for i, fname in enumerate(files[:N_IMAGES]):

        img_path = os.path.join(RAW_IMG_DIR, fname)

        # Charger l'image
        img_np = np.array(Image.open(img_path).convert("RGB"))

        # Prétraitement complet (méthode preprocess_image de la classe pretraitement)
        img_tensor = preproc.preprocess_image(img_np)

        # Sauvegarde des images prétraité dans le dossier /images_preprocessed
        out_path = os.path.join(PREPROC_IMG_DIR, fname)
        save_image(img_tensor, out_path)

        if (i + 1) % 10000 == 0: # c'est juste pour afficher le nombre d'images prétraité
            print(f"{i + 1} images prétraitées...")

    print("\n Prétraitement images terminé.")
    print("Images prétraitées dans :", PREPROC_IMG_DIR)


def main():
    
    # Prétraiter les images  
    preprocess_all_images()

    # Prétraiter les attributs
    preprocess_attributes()

    print("\n PRÉTRAITEMENT COMPLET TERMINÉ ")
    print("Images prétraitées :", PREPROC_IMG_DIR)
    print("Attributs prétraités : attributes.pth")


if __name__ == "__main__":
    main()
