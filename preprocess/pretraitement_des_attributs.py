import os
import numpy as np
import torch

# Nombre d'images dans CelebA
N_IMAGES = 202599

 
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LIST_ATTR_PATH = os.path.join(ROOT, "CelebA", "CelebA", "Anno", "list_attr_celeba.txt")
ATTR_PATH = os.path.join(ROOT, "Data_preprocessed", "attributes.pth")


def preprocess_attributes():
    """
    Prétraitement des attributs :

    - lit list_attr_celeba.txt
    - crée un dict {attr_name: bool[N_IMAGES]}
    - sauvegarde dans attributes.pth
    """

    if os.path.isfile(ATTR_PATH):
        print(f"{ATTR_PATH} exists, nothing to do.")
        return

    print(f"Reading attributes from: {LIST_ATTR_PATH}")
    attr_lines = [line.rstrip() for line in open(LIST_ATTR_PATH, "r")]
    assert len(attr_lines) == N_IMAGES + 2

    # Ligne 1 : noms des 40 attributs CelebA
    attr_keys = attr_lines[1].split()
    assert len(attr_keys) == 40

    # Dictionnaire d'attributs : un bool array par attribut
    attributes = {k: np.zeros(N_IMAGES, dtype=np.bool_) for k in attr_keys}

    # Remplissage
    for i, line in enumerate(attr_lines[2:]):
        image_id = i + 1
        split = line.split()
        assert split[0] == ("%06i.jpg" % image_id)
        values = split[1:]
        assert len(values) == 40

        for j, value in enumerate(values):
            assert value in ["-1", "1"]
            attributes[attr_keys[j]][i] = (value == "1")

    # Sauvegarde
    os.makedirs(os.path.dirname(ATTR_PATH), exist_ok=True)
    print(f"Saving attributes to {ATTR_PATH} ...")
    torch.save(attributes, ATTR_PATH)
    print("Done.")


if __name__ == "__main__":
    preprocess_attributes()
