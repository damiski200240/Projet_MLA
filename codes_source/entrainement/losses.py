"""
Losses pour l'entraînement Fader.

On utilise :
- MSE pour la reconstruction (AutoEncoder).
- BCEWithLogits pour le discriminateur latent (classification binaire).

Le papier Fader Networks utilise pour l'encodeur une loss adversariale
qui pousse le discriminateur à prédire (1 - y).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class FaderLosses:
    """
    - MSELoss : reconstruction
    - BCEWithLogitsLoss : classification (logits -> cible 0/1)
    """

    def __init__(self):
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def recon_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss : ||x_hat - x||^2
        x_hat et x sont des images normalisées [-1,1].
        """
        return self.mse(x_hat, x)

    def dis_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Loss du discriminateur latent.

        logits : sortie du discriminateur Dis(z), shape [B,1]
        y      : attribut cible, shape [B,1], valeurs 0/1

        Objectif : Dis(z) -> y
        """
        return self.bce(logits, y)

    def adv_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Loss adversariale côté encodeur (Fader).

        L'encodeur veut rendre l'attribut imprédictible dans z.
        Dans le papier, on considère une "erreur" quand Dis prédit (1 - y),
        et l'encodeur maximise la probabilité de cette erreur.
        On implémente cela avec BCE(logits, 1 - y).

        Objectif encodeur : Dis(z) -> (1 - y)
        """
        return self.bce(logits, 1.0 - y)


def discriminator_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Accuracy binaire du discriminateur latent sur un batch.

    logits : [B,1]
    y      : [B,1] en 0/1

    Retourne un float (0..1).
    """
    with torch.no_grad():
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = (pred == (y > 0.5).float()).float().mean().item()
    return float(acc)
