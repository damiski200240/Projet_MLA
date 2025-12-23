"""
============================================================
test_external_one.py
------------------------------------------------------------
Test d'un modèle Fader pré-entraîné sur UNE image externe
(non issue de CelebA).

Ce script :
- charge une image externe (jpg / png)
- applique un prétraitement compatible CelebA
- encode l'image avec le Fader
- génère :
    * before.png
    * after.png
    * before_after.png
    * grid.png (interpolations)
    * meta.txt

Usage (PowerShell) :
python test_external_one.py --model_pth models\\eyeglasses.pth --image External_images\\img1.jpg

============================================================
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.utils import make_grid, save_image


# ------------------------------------------------------------
# Prétraitement image externe (imite CelebA)
# ------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),                 # même taille que CelebA
    transforms.ToTensor(),                         # [0,1]
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))            # [-1,1]
])


def load_external_image(img_path: str, device: str):
    """
    Charge une image externe et la convertit en Tensor (1,3,256,256)
    normalisé dans [-1,1].
    """
    img = Image.open(img_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(device)
    return x


def tensor_to_pil(x: torch.Tensor):
    """
    Convertit un Tensor [-1,1] -> PIL Image
    """
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1) / 2.0
    x = (x * 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def draw_border(img: Image.Image, thickness=5):
    """
    Dessine un cadre vert autour d'une image (pour marquer l'original).
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for i in range(thickness):
        draw.rectangle([i, i, w-1-i, h-1-i], outline=(0, 255, 0))
    return out


# ------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Test Fader sur image externe")
    parser.add_argument("--model_pth", required=True,
                        help="Chemin du modèle Fader (.pth)")
    parser.add_argument("--image", required=True,
                        help="Chemin de l'image externe")
    parser.add_argument("--out_dir", default="results/external",
                        help="Dossier de sortie")
    parser.add_argument("--alpha_min", type=float, default=1.2,
                        help="alpha_min (plage: [1-alpha_min, alpha_max])")
    parser.add_argument("--alpha_max", type=float, default=1.0,
                        help="alpha_max")
    parser.add_argument("--n_interpolations", type=int, default=10,
                        help="Nombre d'interpolations")
    parser.add_argument("--assumed_attr", type=int, default=0,
                        help="Attribut supposé de l'image (0 ou 1)")
    parser.add_argument("--cpu", action="store_true",
                        help="Forcer CPU")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------
    assert os.path.isfile(args.model_pth), "Modèle introuvable"
    assert os.path.isfile(args.image), "Image introuvable"

    # --------------------------------------------------------
    # Charger le modèle
    # --------------------------------------------------------
    ae = torch.load(args.model_pth,
                    map_location=device,
                    weights_only=False).eval()

    assert hasattr(ae, "attr") and len(ae.attr) == 1, \
        "Ce script suppose un modèle avec un seul attribut"

    attr_name = ae.attr[0] if not isinstance(ae.attr[0], (list, tuple)) else ae.attr[0][0]

    print("Modèle :", args.model_pth)
    print("Attribut :", attr_name)
    print("Image externe :", args.image)

    # --------------------------------------------------------
    # Charger et prétraiter l'image externe
    # --------------------------------------------------------
    x = load_external_image(args.image, device)

    # --------------------------------------------------------
    # Construire le label d'entrée (choisi manuellement)
    # --------------------------------------------------------
    if args.assumed_attr == 0:
        y_true = torch.tensor([[1., 0.]], device=device)
    else:
        y_true = torch.tensor([[0., 1.]], device=device)

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------
    enc = ae.encode(x)

    # Reconstruction
    recon = ae.decode(enc, y_true)[-1]

    # AFTER = alpha_max
    a_after = float(args.alpha_max)
    y_after = torch.tensor([[1 - a_after, a_after]],
                           device=device,
                           dtype=torch.float32)
    after = ae.decode(enc, y_after)[-1]

    # --------------------------------------------------------
    # Interpolations
    # --------------------------------------------------------
    alphas = np.linspace(1 - args.alpha_min,
                         args.alpha_max,
                         args.n_interpolations)

    inters = []
    for a in alphas:
        ya = torch.tensor([[1 - a, a]],
                          device=device,
                          dtype=torch.float32)
        inters.append(ae.decode(enc, ya)[-1])

    # --------------------------------------------------------
    # Sauvegarde des résultats
    # --------------------------------------------------------
    model_name = os.path.splitext(os.path.basename(args.model_pth))[0]
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(args.out_dir, model_name, img_name)
    os.makedirs(out_path, exist_ok=True)

    # Convertir en images
    before_pil = tensor_to_pil(x)
    recon_pil = tensor_to_pil(recon)
    after_pil = tensor_to_pil(after)

    before_pil.save(os.path.join(out_path, "before.png"))
    after_pil.save(os.path.join(out_path, "after.png"))

    # before_after
    W, H = before_pil.size
    ba = Image.new("RGB", (2*W, H))
    ba.paste(before_pil, (0, 0))
    ba.paste(after_pil, (W, 0))
    ba.save(os.path.join(out_path, "before_after.png"))

    # Grid interpolations
    grid_tensors = [x, recon] + inters
    grid = make_grid(torch.cat(grid_tensors, dim=0),
                     nrow=len(grid_tensors),
                     normalize=True,
                     value_range=(-1, 1))
    save_image(grid, os.path.join(out_path, "grid.png"))

    # Meta
    with open(os.path.join(out_path, "meta.txt"), "w") as f:
        f.write(f"model = {args.model_pth}\n")
        f.write(f"image = {args.image}\n")
        f.write(f"attribute = {attr_name}\n")
        f.write(f"assumed_attr = {args.assumed_attr}\n")
        f.write(f"alpha_min = {args.alpha_min}\n")
        f.write(f"alpha_max = {args.alpha_max}\n")
        f.write(f"n_interpolations = {args.n_interpolations}\n")
        f.write("alphas = " + ",".join([str(a) for a in alphas]))

    print("Résultats sauvegardés dans :", out_path)
    print("Fichiers : before.png | after.png | before_after.png | grid.png | meta.txt")


if __name__ == "__main__":
    main()
