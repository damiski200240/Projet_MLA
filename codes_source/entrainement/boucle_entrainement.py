 
"""
Boucle d'entraînement Fader  

Ce fichier contient la "logique d'entraînement":
- comment extraire y (attribut) depuis attrs
- comment entraîner le discriminateur latent (Dis)
- comment entraîner l'autoencoder (Encoder + Decoder) 
avec l'adversarial

 
"""

from __future__ import annotations
from typing import Dict, Any, Tuple

import torch
from torch.nn.utils import clip_grad_norm_

from .schedule import lambda_schedule
from .losses import FaderLosses, discriminator_accuracy


def make_attr_tensor(attrs: torch.Tensor, attr_idx: int, device: str) -> torch.Tensor:
    """
    Convertit attrs -> y pour UN attribut.

    attrs : Tensor [B, n_attr_total] (ex: 13 attributs)
    attr_idx : index de l'attribut à apprendre (Male, Young, ...)
    Retour :
      y : Tensor [B,1] en float, valeurs 0/1

    Remarque:
    - Si tes attrs sont déjà 0/1 => OK.
    - Si tes attrs sont -1/1 => (attrs > 0) convertit en 0/1.
    """
    y = attrs[:, attr_idx]

    # convertit en 0/1 si jamais c'est -1/1
    y01 = (y > 0).float()

    # reshape en [B,1]
    y01 = y01.unsqueeze(1).to(device)

    return y01


def train_one_step(
    x: torch.Tensor,
    attrs: torch.Tensor,
    attr_idx: int,
    encoder,
    decoder,
    discriminator,
    opt_ae,
    opt_dis,
    losses: FaderLosses,
    step: int,
    lambda_lat: float,
    lambda_warmup: int,
    n_dis_steps: int = 1,
    clip_grad: float = 5.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Effectue un "step" complet de Fader sur un batch:
      1) entraîne Dis(z) -> y  (discriminator latent)
      2) entraîne AE:
           - reconstruction MSE
           - adversarial: encoder veut que Dis(z) prédise 1-y
           - pondération par lambda(t) (warmup)

    Retourne des métriques utiles pour le logging.
    """
    encoder.train()
    decoder.train()
    discriminator.train()

    x = x.to(device)
    y = make_attr_tensor(attrs, attr_idx, device=device)  # [B,1]

  
    # (1) Train Discriminator
    
    loss_dis_val = 0.0
    acc_dis_val = 0.0

    for _ in range(max(1, n_dis_steps)):
        # on ne veut pas backprop dans l'encodeur pendant l'update du dis
        with torch.no_grad():
            z = encoder(x)

        logits = discriminator(z.detach())  # [B,1]
        loss_dis = losses.dis_loss(logits, y)

        opt_dis.zero_grad(set_to_none=True)
        loss_dis.backward()

        if clip_grad and clip_grad > 0:
            clip_grad_norm_(discriminator.parameters(), clip_grad)

        opt_dis.step()

        loss_dis_val = float(loss_dis.item())
        acc_dis_val = discriminator_accuracy(logits, y)

     
    # (2) Train AutoEncoder (E + D)
 
    z = encoder(x)
    x_hat = decoder(z, y)
    loss_rec = losses.recon_loss(x_hat, x)

    logits_adv = discriminator(z)  # attention: sans detach => gradients vers l'encodeur
    loss_adv = losses.adv_loss(logits_adv, y)

    lam = lambda_schedule(step=step, lambda_max=lambda_lat, warmup_steps=lambda_warmup)
    loss_total = loss_rec + lam * loss_adv

    opt_ae.zero_grad(set_to_none=True)
    loss_total.backward()

    if clip_grad and clip_grad > 0:
        clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), clip_grad)

    opt_ae.step()

    # métriques
    metrics = {
        "loss_rec": float(loss_rec.item()),
        "loss_dis": float(loss_dis_val),
        "loss_adv": float(loss_adv.item()),
        "lambda": float(lam),
        "acc_dis": float(acc_dis_val),
        "batch_size": int(x.size(0)),
    }
    return metrics


@torch.no_grad()
def validate_reconstruction(
    valid_loader,
    attr_idx: int,
    encoder,
    decoder,
    losses: FaderLosses,
    device: str = "cuda",
) -> float:
    """
    Validation simple: MSE reconstruction sur le set valid.

    On ne mesure ici que la reconstruction (indicateur stabilité et qualité AE).
    """
    encoder.eval()
    decoder.eval()

    rec_losses = []

    for x, attrs in valid_loader:
        x = x.to(device)
        y = make_attr_tensor(attrs, attr_idx, device=device)  # [B,1]

        z = encoder(x)
        x_hat = decoder(z, y)

        rec = losses.recon_loss(x_hat, x).item()
        rec_losses.append(rec)

    if len(rec_losses) == 0:
        return float("inf")

    return float(sum(rec_losses) / len(rec_losses))
