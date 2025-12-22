import torch
from Encoder import Encoder
from Discriminator import Discriminator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)

    # Paramètres
    B = 4          # batch size
    n_attr = 3     # nombre d'attributs

    # images RGB 256x256 
    x = torch.randn(B, 3, 256, 256, device=device)

    # On instancie l'encodeur et le discriminateur
    encoder = Encoder(in_channels=3, z_channels=512).to(device)
    discriminator = Discriminator(n_attr=n_attr, z_channels=512, hid_dim=512).to(device)

    # Passage dans l'encodeur pour obtenir le code latent z
    z = encoder(x)   # [B, 512, 2, 2]

    #  Passage dans le discriminateur pour prédire les attributs
    y_pred = discriminator(z)  # [B, n_attr]

    #  Affichage des shapes
    print("x shape      :", x.shape)       
    print("z shape      :", z.shape)       
    print("y_pred shape :", y_pred.shape)  

    print("y_pred (quelques valeurs) :")
    print(y_pred[:2])  


if __name__ == "__main__":
    main()
