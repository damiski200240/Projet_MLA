import os 
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class CelebADataset(Dataset):
    """
    Dataset CelebA basé sur :
      - un dossier d'images prétraitées (.jpg, 256x256)
      - un fichier attributes.pth au format Fader (dict: attr_name -> bool[N])

    On reconstruit à partir de ce dict un tensor (N, 13) en 0/1
    pour un sous-ensemble d'attributs.
    """

    def __init__(
        self,
        img_dir: str,
        attr_path: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()

        assert split in ["train", "valid", "test", "all"], \
            "split doit être 'train', 'valid', 'test' ou 'all'"

        self.img_dir = img_dir
        self.transform = transform

        # Charger le dictionnaire d'attributs  
         
        attributes_dict = torch.load(attr_path, weights_only=False)

        # Attributs que l'on souhaite conserver dans un premier temps (13 parmi les 40 de CelebA)
        selected_attr_names = [
            "Bald",
            "Black_Hair",
            "Blond_Hair",
            "Brown_Hair",
            "Eyeglasses",
            "Gray_Hair",
            "Heavy_Makeup",
            "Male",
            "Mustache",
            "Pale_Skin",
            "Smiling",
            "Wavy_Hair",
            "Young",
        ]

        # Construire un tensor (N_IMAGES, 13) en 0/1
        attr_tensors = []
        for name in selected_attr_names:
            assert name in attributes_dict, f"Attribut {name} manquant dans attributes.pth"
            arr = attributes_dict[name].astype(np.float32)   # bool -> 0.0 / 1.0
            t = torch.from_numpy(arr).unsqueeze(1)          # (N,1)
            attr_tensors.append(t)

        attributes = torch.cat(attr_tensors, dim=1)          # (N, 13)

        self.attributes = attributes
        self.attr_names = selected_attr_names

        n_images = self.attributes.size(0)

        # Split officiel CelebA  train valid et test 
        if split == "train":
            start, end = 0, 162770
        elif split == "valid":
            start, end = 162770, 162770 + 19867
        elif split == "test":
            start, end = 162770 + 19867, n_images
        else:  # "all"
            start, end = 0, n_images

        self.indices = list(range(start, end))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne :
          - image tensor : [3, 256, 256], normalisée selon self.transform
          - attributs : tensor [13] avec valeurs 0 ou 1
        """
        real_idx = self.indices[idx]               # index global 0..N-1
        img_name = f"{real_idx + 1:06d}.jpg"       # 000001.jpg, 000002.jpg, ...
        img_path = os.path.join(self.img_dir, img_name)

        # Charger les images prétraitées
        img = Image.open(img_path).convert("RGB")

        # Appliquer les transforms (ToTensor + Normalize) 
        if self.transform is not None:
            img = self.transform(img)

        attrs = self.attributes[real_idx]          # vecteur (13,)
        return img, attrs


def get_default_transforms():
    """
    Transformations appliquées aux images au moment du chargement.

    Les images dans Images_Preprocessed sont déjà :
      - recadrées
      - redimensionnées en 256x256
      - avec des valeurs 0–255

    Ici on fait seulement :
      - ToTensor() -> [0,1]
      - Normalize(0.5, 0.5, 0.5)  [-1,1]
    """
    return transforms.Compose([
        transforms.ToTensor(),                         # [0,255]  to  [0,1]
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),        #  [-1,1]
    ])


def get_dataloaders(
    root_dir: str = ".",
    batch_size: int = 64,
    num_workers: int = 0,
):
    """
    Crée trois DataLoader : train, valid, test.

     """

    img_dir = os.path.join(root_dir, "Data_preprocessed", "Images_Preprocessed")
    attr_path = os.path.join(root_dir, "Data_preprocessed", "attributes.pth")

    transform = get_default_transforms()

    train_set = CelebADataset(
        img_dir=img_dir,
        attr_path=attr_path,
        split="train",
        transform=transform,
    )
    valid_set = CelebADataset(
        img_dir=img_dir,
        attr_path=attr_path,
        split="valid",
        transform=transform,
    )
    test_set = CelebADataset(
        img_dir=img_dir,
        attr_path=attr_path,
        split="test",
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader
