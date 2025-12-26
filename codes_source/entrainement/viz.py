 
"""
Outils de visualisation pendant l'entraînement.

On sauvegarde des grilles d'images pour suivre la progression :
- originaux
- reconstructions
- swap (attribut inversé)

Toutes les images sont supposées être normalisées dans [-1, 1].
"""

from __future__ import annotations
from pathlib import Path

import torch
from torchvision.utils import save_image


@torch.no_grad()
def save_monitor_grid(
    encoder,
    decoder,
    x: torch.Tensor,
    y: torch.Tensor,
    out_path: str | Path,
) -> None:
    """
    Sauvegarde une grille d'images : [orig | recon | swap].

    Paramètres
    ----------
    encoder, decoder : modèles PyTorch
        Encoder E et Decoder D.
    x : Tensor
        Batch d'images, shape [B, 3, H, W], valeurs dans [-1, 1].
    y : Tensor
        Attributs batch, shape [B, 1], valeurs 0/1.
    out_path : str | Path
        Chemin du fichier PNG à écrire.

    Sortie
    ------
    Un PNG contenant 3 lignes (ou 3 blocs) de B images :
        - x (originaux)
        - D(E(x), y) (recon)
        - D(E(x), 1-y) (swap)
    """
    encoder.eval()
    decoder.eval()

    z = encoder(x)
    x_rec = decoder(z, y)
    x_swap = decoder(z, 1.0 - y)

    # concaténation : 3B images
    grid = torch.cat([x, x_rec, x_swap], dim=0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # images sont en [-1,1] -> normalize=True avec value_range permet de sauver correctement
    save_image(
        grid,
        str(out_path),
        nrow=x.size(0),
        normalize=True,
        value_range=(-1, 1),
    )
