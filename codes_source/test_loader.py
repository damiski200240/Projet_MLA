import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from loader import get_dataloaders 
from loader import CelebADataset  



def denormalize(img_tensor):
    """
    img_tensor : [3, H, W] normalisé avec mean=0.5, std=0.5
    On le remet en [0,1] pour l'affichage.
    """
    img = img_tensor * 0.5 + 0.5       # [-1,1]  [0,1]
    return img.clamp(0, 1)


def show_image(img_tensor, title=None):
    """
    Affiche une image (3,H,W) avec matplotlib.
    """
    img = denormalize(img_tensor).permute(1, 2, 0).cpu()  # [C,H,W] -> [H,W,C]
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    # 1) Récupérer les DataLoaders
    print("Chargement des DataLoaders...")
    train_loader, valid_loader, test_loader = get_dataloaders(
        root_dir='C:/Users/amira/Desktop/Projet_MLA',      # racine du projet (dossier Projet_MLA)
        batch_size=32,      # petit batch pour test
        num_workers=0,     # sous Windows, 0 pour éviter les soucis
    )

    # 2) Récupérer un batch du train
    print("Récupération d'un batch d'entraînement...")
    images, attrs = next(iter(train_loader))

    print(f"Shape des images : {images.shape}")    
    print(f"Shape des attributs : {attrs.shape}")  

    print("Valeurs min / max des pixels (après normalisation) :",
          images.min().item(), images.max().item())

    # 3) Afficher les attributs de la première image
    ds = CelebADataset(
        img_dir="Data_preprocessed/Images_Preprocessed",
        attr_path="Data_preprocessed/attributes.pth",
        split="train",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ]),
    )

    attr_names = ds.attr_names

    print("\nAttributs pour la première image du batch :")
    for name, value in zip(attr_names, attrs[0].tolist()):
        print(f"  {name:15s} : {value:.3f}")

    # 4) Afficher la première image du batch
    print("\nAffichage de la première image du batch...")
    show_image(images[0], title="Image 0 (train_batch)")


if __name__ == "__main__":
    main()
