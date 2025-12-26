 
"""
I/O training:
- création des dossiers de run
- sauvegarde / chargement de checkpoints (reprise)
- export final du modèle entraîné (male.pth, young.pth, ...)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class RunPaths:
    """
    Contient tous les chemins d'un run d'entraînement.

    Exemple pour attr="Male":
      modeles_entraines/Male_256/
        checkpoints/
        logs/
        samples/
        exports/
    """
    run_dir: Path
    checkpoints: Path
    logs: Path
    samples: Path
    exports: Path


def make_run_paths(base_out_dir: str | Path, run_name: str) -> RunPaths:
    """
    Crée les dossiers d'un run.
    base_out_dir = "modeles_entraines"
    run_name     = "Male_256"

    Retourne un objet RunPaths avec les chemins prêts.
    """
    base_out_dir = Path(base_out_dir)
    run_dir = base_out_dir / run_name

    checkpoints = run_dir / "checkpoints"
    logs = run_dir / "logs"
    samples = run_dir / "samples"
    exports = run_dir / "exports"

    for p in [checkpoints, logs, samples, exports]:
        p.mkdir(parents=True, exist_ok=True)

    return RunPaths(run_dir=run_dir, checkpoints=checkpoints, logs=logs, samples=samples, exports=exports)


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    """
    Sauvegarde un checkpoint complet.
    payload doit contenir (au minimum) :
      - encoder / decoder / discriminator state_dict
      - optimizers state_dict
      - epoch / step
      - config
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Charge un checkpoint.
    Retourne le dict sauvegardé.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable: {path}")
    return torch.load(str(path), map_location=map_location)


def export_trained_model(
    path: str | Path,
    attr_name: str,
    encoder,
    decoder,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export final minimal pour le test/interpolation plus tard.

    On exporte:
      - encoder.state_dict()
      - decoder.state_dict()
      - meta : infos utiles (attr, alpha_min/max, config, epoch/step...)

    Exemple : modeles_entraines/Male_256/exports/male.pth
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "attr_name": attr_name,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "meta": meta or {},
    }
    torch.save(payload, str(path))
