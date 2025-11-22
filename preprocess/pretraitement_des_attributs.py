#!/usr/bin/env python
import os
import numpy as np
import torch

N_IMAGES = 202599
ATTR_PATH = 'Data_preprocessed/attributes.pth'
HERE = os.path.dirname(os.path.abspath(__file__))

ROOT = os.path.dirname(HERE)   
ATTR_TXT_PATH = os.path.join(ROOT, "CelebA","CelebA", "Anno", "list_attr_celeba.txt")

def preprocess_attributes():
    """Prétraitement des attributs - Même style que votre code original"""
    
    if os.path.isfile(ATTR_PATH):
        print("%s exists, nothing to do." % ATTR_PATH)
        return

    print("Reading attributes from list_attr_celeba.txt...")
    
    # Lecture du fichier d'attributs (comme dans votre code original)
    attr_lines = [line.rstrip() for line in open(ATTR_TXT_PATH, 'r')]
    assert len(attr_lines) == N_IMAGES + 2

    # Extraction des noms d'attributs
    attr_keys = attr_lines[1].split()
    
    # Sélection des attributs importants pour Fader Networks
    selected_attributes = [
        'Smiling', 'Male', 'Young', 'Eyeglasses', 'Mustache', 
        'Bald', 'Wavy_Hair', 'Pale_Skin', 'Heavy_Makeup',
        'Blond_Hair', 'Black_Hair', 'Brown_Hair', 'Gray_Hair'
    ]
    
    # Garder seulement les attributs sélectionnés
    selected_indices = [i for i, key in enumerate(attr_keys) if key in selected_attributes]
    selected_keys = [attr_keys[i] for i in selected_indices]
    
    print(f"Selected {len(selected_keys)} attributes: {selected_keys}")
    
    # Création du dictionnaire d'attributs (comme votre code original)
    attributes = {k: np.zeros(N_IMAGES, dtype=np.bool) for k in selected_keys}

    # Remplissage des attributs
    for i, line in enumerate(attr_lines[2:]):
        image_id = i + 1
        split = line.split()
        assert len(split) == 41
        assert split[0] == ('%06i.jpg' % image_id)
        assert all(x in ['-1', '1'] for x in split[1:])
        
        for j, attr_index in enumerate(selected_indices):
            value = split[1:][attr_index]
            attributes[selected_keys[j]][i] = value == '1'

    # Conversion en tenseur PyTorch et normalisation
    print("Converting to PyTorch tensor and normalizing...")
    
    # Créer un tenseur avec tous les attributs
    attr_tensor = np.zeros((N_IMAGES, len(selected_keys)), dtype=np.float32)
    for j, key in enumerate(selected_keys):
        attr_tensor[:, j] = attributes[key].astype(np.float32)
    
    # Normalisation (mean=0, std=1)
    mean = attr_tensor.mean(axis=0)
    std = attr_tensor.std(axis=0)
    normalized_attrs = (attr_tensor - mean) / std
    
    # Convertir en tenseur PyTorch
    normalized_attrs_tensor = torch.from_numpy(normalized_attrs)
    
    # Sauvegarde avec toutes les informations (comme votre code original)
    torch.save({
        'attributes': normalized_attrs_tensor,
        'attribute_names': selected_keys,
        'mean': torch.from_numpy(mean),
        'std': torch.from_numpy(std),
        'original_bool_attributes': attributes
    }, ATTR_PATH)

    print("Saving attributes to %s ..." % ATTR_PATH)
    
    # Afficher les statistiques
    print("\nAttribute statistics:")
    for j, key in enumerate(selected_keys):
        positive_count = attributes[key].sum()
        percentage = (positive_count / N_IMAGES) * 100
        print(f"  {key}: {positive_count} positives ({percentage:.1f}%)")

if __name__ == '__main__':
    print("Starting attributes preprocessing...")
    preprocess_attributes()
    print("Attributes preprocessing completed!")