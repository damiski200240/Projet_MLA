import os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam

from loader import get_dataloaders
from Encoder import Encoder
from Decoder import Decoder
from Discriminator import Discriminator

ATTR_NAMES = [
    "Bald","Black_Hair","Blond_Hair","Brown_Hair","Eyeglasses","Gray_Hair",
    "Heavy_Makeup","Male","Mustache","Pale_Skin","Smiling","Wavy_Hair","Young"
]
NAME_TO_ID = {n:i for i,n in enumerate(ATTR_NAMES)}

def denorm_to_01(x):
    return (x.clamp(-1, 1) + 1) / 2

def get_lambda(step, warmup_steps=500_000, lambda_max=1e-4):
    t = min(1.0, step / float(warmup_steps)) if warmup_steps > 0 else 1.0
    return lambda_max * t

@torch.no_grad()
def save_alpha_grid(enc, dec, valid_loader, device, epoch, attr="Smiling",
                    alphas=(0,0.25,0.5,0.75,1.0), out_dir="plots", max_rows=6):
    enc.eval(); dec.eval()
    os.makedirs(out_dir, exist_ok=True)
    a = NAME_TO_ID[attr]

    x, y = next(iter(valid_loader))
    x = x.to(device)[:max_rows]
    y = y.to(device)[:max_rows].float()

    z = enc(x)
    x_rec = dec(z, y)

    cols = 2 + len(alphas)
    plt.figure(figsize=(3*cols, 3*max_rows))

    def plot_col(imgs, col, title):
        imgs = denorm_to_01(imgs).cpu()
        for r in range(imgs.size(0)):
            ax = plt.subplot(max_rows, cols, r*cols + col + 1)
            ax.imshow(imgs[r].permute(1,2,0).numpy())
            ax.axis("off")
            if r == 0: ax.set_title(title)

    plot_col(x, 0, "x")
    plot_col(x_rec, 1, "recon")

    for j, alpha in enumerate(alphas):
        y2 = y.clone()
        y2[:, a] = float(alpha)
        x2 = dec(z, y2)
        plot_col(x2, 2+j, f"{attr}={alpha}")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"alpha_{attr}_epoch_{epoch:03d}.png"))
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, valid_loader, _ = get_dataloaders(root_dir=".", batch_size=64, num_workers=2)
    _, y0 = next(iter(train_loader))
    n_attr = y0.shape[1]
    print("n_attr =", n_attr)

    enc = Encoder().to(device)
    dec = Decoder(n_attr=n_attr).to(device)
    dis = Discriminator(n_attr=n_attr).to(device)

    opt_ed = Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-4, betas=(0.5, 0.999))
    opt_dis = Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # PAPER / REPO: MSE reconstruction
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    num_epochs = 200
    warmup_steps = 500_000
    lambda_max = 1e-4

    global_step = 0
    best_val = float("inf")

    # stats à la manière du repo (moyenne sur 25 it)
    rec_costs, dis_costs, adv_costs = [], [], []

    for epoch in range(1, num_epochs + 1):
        enc.train(); dec.train(); dis.train()
        t0 = time.time()

        for it, (x, y) in enumerate(train_loader, 1):
            global_step += 1
            lam = get_lambda(global_step, warmup_steps, lambda_max)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            # -------- lat_dis_step (comme repo) --------
            enc.eval()
            dis.train()
            with torch.no_grad():
                z_det = enc(x)
            pred = dis(z_det.detach())
            loss_dis = bce(pred, y)

            opt_dis.zero_grad()
            loss_dis.backward()
            opt_dis.step()

            # -------- autoencoder_step (comme repo) --------
            enc.train(); dec.train()
            for p in dis.parameters():
                p.requires_grad = False

            z = enc(x)
            xhat = dec(z, y)

            # REPO: lambda_ae * MSE
            loss_rec = mse(xhat, x)

            # REPO: encoder loss via latent dis (get_attr_loss(..., True))
            # papier: -log P(1-y|z)  => BCE(pred, 1-y)
            pred_adv = dis(z)
            loss_adv = bce(pred_adv, 1.0 - y)

            loss = loss_rec + lam * loss_adv

            opt_ed.zero_grad()
            loss.backward()
            opt_ed.step()

            for p in dis.parameters():
                p.requires_grad = True

            rec_costs.append(loss_rec.item())
            dis_costs.append(loss_dis.item())
            adv_costs.append(loss_adv.item())

            # log style repo toutes les 25 it
            if len(rec_costs) >= 25:
                print(
                    f"{global_step:06d} - "
                    f"Reconstruction: {np.mean(rec_costs):.5f} / "
                    f"Latent dis: {np.mean(dis_costs):.5f} / "
                    f"Adv: {np.mean(adv_costs):.5f} / "
                    f"lambda: {lam:.2e}"
                )
                rec_costs.clear(); dis_costs.clear(); adv_costs.clear()

        # -------- validation recon (best_rec comme repo) --------
        enc.eval(); dec.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).float()
                z = enc(x)
                xhat = dec(z, y)
                val_sum += mse(xhat, x).item()
        val_loss = val_sum / len(valid_loader)

        # save latest
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "dis": dis.state_dict(),
                    "epoch": epoch, "global_step": global_step, "val_mse": val_loss},
                   "checkpoints/fader_latest.pt")

        # save best_rec (comme repo)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "dis": dis.state_dict(),
                        "epoch": epoch, "global_step": global_step, "val_mse": val_loss},
                       "checkpoints/best_rec.pt")
            print(f"[BEST_REC] val_mse={best_val:.6f} @ epoch={epoch}")

        # save periodic (comme repo)
        if epoch % 5 == 0:
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "dis": dis.state_dict(),
                        "epoch": epoch, "global_step": global_step, "val_mse": val_loss},
                       f"checkpoints/periodic_{epoch:03d}.pt")

        # visuals (alpha grids, comme figures)
        for attr in ["Male", "Eyeglasses", "Smiling"]:
            save_alpha_grid(enc, dec, valid_loader, device, epoch, attr=attr)

        print(f"Epoch {epoch}/{num_epochs} done | val_mse={val_loss:.6f} | time={(time.time()-t0)/60:.1f} min")

    print("Done. Checkpoints in checkpoints/ and visuals in plots/")

if __name__ == "__main__":
    main()
