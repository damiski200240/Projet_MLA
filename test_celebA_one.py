# ============================================================
# test_celebA_one.py  (VERSION FINALE - sans alpha_after)
# ------------------------------------------------------------
# Objectif :
#   Tester un modèle Fader pré-entraîné (male.pth, eyeglasses.pth, ...)
#   sur UNE image CelebA déjà prétraitée.
#
# Sortie :
#   results/one/<model>/<img_id>/
#       grid.png            -> Original | Recon | Interpolations...
#       before.png          -> image originale
#       after.png           -> image transformée (alpha = alpha_max)
#       before_after.png    -> 2 colonnes (before | after)
#       meta.txt            -> infos
#
# Usage (PowerShell / CMD) :
#   python test_celebA_one.py --model_pth models/eyeglasses.pth --img_id 182638
# ============================================================

import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

# Normalisation [-1,1] (cohérente avec Fader)
TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

def resolve_fname(img_dir: str, img_id: int) -> str:
    """Trouve le fichier image correspondant à l'ID (plusieurs formats possibles)."""
    candidates = [
        f"{img_id}.jpg", f"{img_id:06d}.jpg", f"{img_id:07d}.jpg", f"{img_id:08d}.jpg",
        f"{img_id}.png", f"{img_id:06d}.png", f"{img_id:07d}.png", f"{img_id:08d}.png",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(img_dir, c)):
            return c
    raise FileNotFoundError(f"Image id {img_id} introuvable dans {img_dir}.")

def load_preprocessed_image(img_dir: str, fname: str, device: str):
    """Charge une image et la convertit en tensor (1,3,H,W) normalisé [-1,1]."""
    img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
    return TFM(img).unsqueeze(0).to(device)

def extract_attr_name(ae_attr_item):
    """ae.attr peut être ['Male'] ou [('Male', 2)] -> on récupère juste 'Male'."""
    if isinstance(ae_attr_item, (tuple, list)) and len(ae_attr_item) >= 1:
        return ae_attr_item[0]
    return ae_attr_item

def onehot_single_attr(attributes: dict, img_id: int, attr_name: str, device: str):
    """
    Construit y one-hot (1,2) depuis attributes.pth.
    Dans CelebA, l’index = img_id - 1.
    """
    idx = img_id - 1
    v = bool(attributes[attr_name][idx])         # True / False
    y = torch.tensor([[0.,1.]] if v else [[1.,0.]], device=device)
    return y, v

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pth", required=True)
    ap.add_argument("--img_id", type=int, required=True)

    # Chemins par défaut (racine du projet)
    ap.add_argument("--img_dir", default="Data_preprocessed/Images_Preprocessed")
    ap.add_argument("--attr_pth", default="Data_preprocessed/attributes.pth")

    # Slider
    ap.add_argument("--n_interpolations", type=int, default=10)
    ap.add_argument("--alpha_min", type=float, default=2.0)
    ap.add_argument("--alpha_max", type=float, default=1.0)

    ap.add_argument("--save_before_after", type=int, default=1,
                    help="1: sauvegarde before/after en plus de la grille")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ae = torch.load(args.model_pth, map_location=device, weights_only=False).eval()
    assert hasattr(ae, "attr") and len(ae.attr) == 1, "Script prévu pour modèles à 1 attribut."
    attr_name = extract_attr_name(ae.attr[0])

    # Load attributes
    attributes = torch.load(args.attr_pth, weights_only=False)
    if attr_name not in attributes:
        raise KeyError(f"Attribut '{attr_name}' absent de attributes.pth.")

    # Load image
    fname = resolve_fname(args.img_dir, args.img_id)
    x = load_preprocessed_image(args.img_dir, fname, device)

    # True attribute (one-hot)
    y_true, v = onehot_single_attr(attributes, args.img_id, attr_name, device)

    # Output folder
    model_name = os.path.splitext(os.path.basename(args.model_pth))[0]
    out_dir = os.path.join("results", "one", model_name, str(args.img_id))
    os.makedirs(out_dir, exist_ok=True)

    print("Model:", args.model_pth)
    print("Attr :", attr_name)
    print("Image:", fname, "true_attr=", v)
    print("Saving to:", out_dir)

    # Encode / recon
    enc = ae.encode(x)
    x_rec = ae.decode(enc, y_true)[-1]

    # Interpolations (comme l'officiel) : alpha in [1-alpha_min, alpha_max]
    alphas = np.linspace(1 - args.alpha_min, args.alpha_max, args.n_interpolations)

    outs = [x, x_rec]
    for a in alphas:
        ya = torch.tensor([[1-a, a]], device=device, dtype=torch.float32)
        outs.append(ae.decode(enc, ya)[-1])

    # Save grid
    grid = torch.cat(outs, dim=0)
    grid_vis = (grid + 1) / 2
    grid_img = make_grid(grid_vis, nrow=len(outs))
    save_image(grid_img, os.path.join(out_dir, "grid.png"))

    # Save before / after
    if args.save_before_after == 1:
        before = (x + 1) / 2

        # AFTER = extrême de la plage -> alpha = alpha_max
        a = float(args.alpha_max)
        ya_after = torch.tensor([[1-a, a]], device=device, dtype=torch.float32)
        after = (ae.decode(enc, ya_after)[-1] + 1) / 2

        save_image(before, os.path.join(out_dir, "before.png"))
        save_image(after, os.path.join(out_dir, "after.png"))

        ba = torch.cat([before, after], dim=0)
        ba_grid = make_grid(ba, nrow=2)
        save_image(ba_grid, os.path.join(out_dir, "before_after.png"))

    # meta
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"model={args.model_pth}\n")
        f.write(f"attr={attr_name}\n")
        f.write(f"img_id={args.img_id}\n")
        f.write(f"filename={fname}\n")
        f.write(f"true_attr={v}\n")
        f.write(f"alpha_min={args.alpha_min}\n")
        f.write(f"alpha_max={args.alpha_max}\n")
        f.write(f"n_interpolations={args.n_interpolations}\n")
        f.write("alphas=" + ",".join([str(x) for x in alphas]) + "\n")

    print("Done. Files: grid.png (+ before/after if enabled)")

if __name__ == "__main__":
    main()
