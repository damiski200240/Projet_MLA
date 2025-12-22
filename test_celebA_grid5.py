# ============================================================
# test_celebA_grid5.py  (VERSION FINALE - sans alpha_after)
# ------------------------------------------------------------
# Objectif (CelebA prétraité + IDs fixes) :
#
# 1) before_after.png :
#       pour chaque image (une ligne)
#       BEFORE (original) | AFTER (transformé, alpha = alpha_max)
#
# 2) interpolations.png :
#       pour chaque image (une ligne)
#       ORIGINAL (entouré) | RECON | alpha1 | alpha2 | ... | alphaK
#
# Sortie :
#   results/grid5/<model>/<ids>/
#       before_after.png
#       interpolations.png
#       meta.txt
#
# Usage :
#   python test_celebA_grid5.py --model_pth models/male.pth --img_ids 182638 190012 195555 200001 202599
# ============================================================

import os
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

# Normalisation [-1,1]
TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

def extract_attr_name(ae_attr_item):
    if isinstance(ae_attr_item, (tuple, list)) and len(ae_attr_item) >= 1:
        return ae_attr_item[0]
    return ae_attr_item

def resolve_fname(img_dir: str, img_id: int) -> str:
    candidates = [
        f"{img_id}.jpg", f"{img_id:06d}.jpg", f"{img_id:07d}.jpg", f"{img_id:08d}.jpg",
        f"{img_id}.png", f"{img_id:06d}.png", f"{img_id:07d}.png", f"{img_id:08d}.png",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(img_dir, c)):
            return c
    raise FileNotFoundError(f"Image id {img_id} introuvable dans {img_dir}.")

def load_preprocessed_image(img_dir: str, fname: str, device: str) -> torch.Tensor:
    img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
    return TFM(img).unsqueeze(0).to(device)  # (1,3,H,W) en [-1,1]

def onehot_single_attr(attributes: dict, img_id: int, attr_name: str, device: str):
    idx = img_id - 1
    v = bool(attributes[attr_name][idx])
    y = torch.tensor([[0.,1.]] if v else [[1.,0.]], device=device)
    return y, v

def tensor01_to_pil(x01: torch.Tensor) -> Image.Image:
    if x01.dim() == 4:
        x01 = x01[0]
    x01 = x01.clamp(0, 1)
    arr = (x01 * 255).byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")

def draw_border(img: Image.Image, color=(0,255,0), thickness=5) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for t in range(thickness):
        d.rectangle([t, t, w-1-t, h-1-t], outline=color)
    return out

def make_rows_image(rows, bg=(0,0,0)):
    W = rows[0].size[0]
    H = sum(r.size[1] for r in rows)
    canvas = Image.new("RGB", (W, H), bg)
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.size[1]
    return canvas

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Grid5 CelebA : before/after + interpolations (2 images séparées)")
    ap.add_argument("--model_pth", required=True)
    ap.add_argument("--img_ids", nargs="+", type=int, required=True)

    ap.add_argument("--img_dir", default="Data_preprocessed/Images_Preprocessed")
    ap.add_argument("--attr_pth", default="Data_preprocessed/attributes.pth")

    ap.add_argument("--n_interpolations", type=int, default=10)
    ap.add_argument("--alpha_min", type=float, default=2.0)
    ap.add_argument("--alpha_max", type=float, default=1.0)

    ap.add_argument("--border_color", type=str, default="0,255,0")
    ap.add_argument("--border_thickness", type=int, default=5)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ae = torch.load(args.model_pth, map_location=device, weights_only=False).eval()
    assert hasattr(ae, "attr") and len(ae.attr) == 1, "Script prévu pour modèles 1 attribut."
    attr_name = extract_attr_name(ae.attr[0])

    # Load attributes
    attributes = torch.load(args.attr_pth, weights_only=False)
    if attr_name not in attributes:
        raise KeyError(f"Attribut '{attr_name}' absent de attributes.pth.")

    # Output folder
    model_name = os.path.splitext(os.path.basename(args.model_pth))[0]
    ids_str = "-".join(str(i) for i in args.img_ids)
    out_dir = os.path.join("results", "grid5", model_name, ids_str)
    os.makedirs(out_dir, exist_ok=True)

    bc = tuple(int(x) for x in args.border_color.split(","))
    bt = args.border_thickness

    # Interpolation alphas (officiel): [1-alpha_min, alpha_max]
    alphas = np.linspace(1 - args.alpha_min, args.alpha_max, args.n_interpolations)

    print("Model:", args.model_pth)
    print("Attr :", attr_name)
    print("IDs  :", args.img_ids)
    print("Saving to:", out_dir)

    rows_before_after = []
    rows_interpolations = []
    meta_lines = []

    for img_id in args.img_ids:
        fname = resolve_fname(args.img_dir, img_id)
        x = load_preprocessed_image(args.img_dir, fname, device)  # [-1,1]
        y_true, v = onehot_single_attr(attributes, img_id, attr_name, device)

        enc = ae.encode(x)

        # BEFORE (original)
        before01 = (x + 1) / 2
        before_pil = tensor01_to_pil(before01)

        # AFTER = alpha_max (extrême)
        a_after = float(args.alpha_max)
        y_after = torch.tensor([[1 - a_after, a_after]], device=device, dtype=torch.float32)
        after01 = (ae.decode(enc, y_after)[-1] + 1) / 2
        after_pil = tensor01_to_pil(after01)

        # before_after row
        W, H = before_pil.size
        row_ba = Image.new("RGB", (2 * W, H), (0, 0, 0))
        row_ba.paste(before_pil, (0, 0))
        row_ba.paste(after_pil, (W, 0))
        rows_before_after.append(row_ba)

        # recon (vrai label)
        recon01 = (ae.decode(enc, y_true)[-1] + 1) / 2
        recon_pil = tensor01_to_pil(recon01)

        # original entouré
        before_bordered = draw_border(before_pil, color=bc, thickness=bt)

        # interpolations
        inter_pils = []
        for a in alphas:
            ya = torch.tensor([[1 - a, a]], device=device, dtype=torch.float32)
            out01 = (ae.decode(enc, ya)[-1] + 1) / 2
            inter_pils.append(tensor01_to_pil(out01))

        cells = [before_bordered, recon_pil] + inter_pils
        ncols = len(cells)

        row_int = Image.new("RGB", (ncols * W, H), (0, 0, 0))
        for j, cell in enumerate(cells):
            row_int.paste(cell, (j * W, 0))
        rows_interpolations.append(row_int)

        meta_lines.append(f"{img_id},{fname},true_attr={v}")
        print(f"  - {img_id} ({fname}) true_attr={v}")

    before_after_img = make_rows_image(rows_before_after)
    interpolations_img = make_rows_image(rows_interpolations)

    out_ba = os.path.join(out_dir, "before_after.png")
    out_int = os.path.join(out_dir, "interpolations.png")

    before_after_img.save(out_ba)
    interpolations_img.save(out_int)

    print("Saved:", out_ba)
    print("Saved:", out_int)

    # meta
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"model={args.model_pth}\n")
        f.write(f"attr={attr_name}\n")
        f.write(f"img_ids={args.img_ids}\n")
        f.write(f"alpha_min={args.alpha_min}\n")
        f.write(f"alpha_max={args.alpha_max}\n")
        f.write(f"n_interpolations={args.n_interpolations}\n")
        f.write(f"border_color={bc}\n")
        f.write(f"border_thickness={bt}\n")
        f.write("alphas=" + ",".join([str(a) for a in alphas]) + "\n")
        f.write("\nper_image:\n")
        for line in meta_lines:
            f.write(line + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
