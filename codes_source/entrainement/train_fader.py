"""
Script principal d'entraînement Fader.

Lancement recommandé depuis la racine du projet :
    python -m codes_source.entrainement.train_fader --attr Male --root_dir . --out_dir modeles_entraines

Sorties:
- modeles_entraines/<Attr>_256/checkpoints/  (reprise)
- modeles_entraines/<Attr>_256/logs/         (train.log + tensorboard optionnel)
- modeles_entraines/<Attr>_256/samples/      (images orig/recon/swap)
- modeles_entraines/<Attr>_256/exports/<attr>.pth  (modèle final pour test/interpolation)
"""

from __future__ import annotations
import time
import argparse

import torch
from torch.optim import Adam
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_OK = True
except Exception:
    TENSORBOARD_OK = False

from codes_source.Encoder import Encoder
from codes_source.Decoder import Decoder
from codes_source.Discriminator import Discriminator
from codes_source.loader import get_dataloaders


from .losses import FaderLosses
from .boucle_entrainement import train_one_step, validate_reconstruction
from .io_utils import make_run_paths, save_checkpoint, export_trained_model
from .viz import save_monitor_grid


def main():
    ap = argparse.ArgumentParser()

    # --- Data / paths ---
    ap.add_argument("--root_dir", type=str, default=".", help="Doit contenir Data_preprocessed/ .")
    ap.add_argument("--attr", type=str, required=True, help='Attribut à entraîner: "Male", "Young", ...')
    ap.add_argument("--out_dir", type=str, default="modeles_entraines", help="Dossier de sorties training")
    ap.add_argument("--num_workers", type=int, default=4)

    # --- Training hyperparams ---
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.999))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--epoch_size", type=int, default=50000, help="Nb d'images vues par epoch (approx)")
    ap.add_argument("--clip_grad", type=float, default=5.0)

    # --- Fader specific ---
    ap.add_argument("--lambda_lat", type=float, default=1e-4)
    ap.add_argument("--lambda_warmup", type=int, default=500000)  # ✅ en steps (batches)
    ap.add_argument("--n_dis_steps", type=int, default=1)

    # --- Monitoring / save ---
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)  # ✅ en steps (batches)
    ap.add_argument("--ckpt_every", type=int, default=5000)  # ✅ en steps (batches)
    ap.add_argument("--tensorboard", action="store_true")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data ---
    train_loader, valid_loader, _ = get_dataloaders(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Trouver l'index de l'attribut demandé
    attr_names = train_loader.dataset.attr_names
    if args.attr not in attr_names:
        raise ValueError(f"Attribut {args.attr} absent. Dispo: {attr_names}")
    attr_idx = attr_names.index(args.attr)

    # --- Run paths ---
    run_name = f"{args.attr}_256"
    paths = make_run_paths(args.out_dir, run_name)

    # S'assure que le dossier checkpoints existe
    paths.checkpoints.mkdir(parents=True, exist_ok=True)

    log_file = paths.logs / "train.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"RUN={run_name} device={device} attr_idx={attr_idx} batch={args.batch_size}\n")

    # --- Models (mono-attribut => n_attr=1) ---
    encoder = Encoder().to(device)
    decoder = Decoder(n_attr=1).to(device)
    discriminator = Discriminator(n_attr=1).to(device)

    # --- Losses ---
    losses = FaderLosses()

    # --- Optimizers ---
    opt_ae = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        betas=tuple(args.betas),
    )
    opt_dis = Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
    )

    # --- TensorBoard ---
    writer = None
    if args.tensorboard:
        if not TENSORBOARD_OK:
            print("TensorBoard non disponible (torch.utils.tensorboard introuvable).")
        else:
            writer = SummaryWriter(log_dir=str(paths.logs))

    # --- Fixed batch pour monitoring (recon + swap) ---
    fixed_x, fixed_attrs = next(iter(valid_loader))
    fixed_x = fixed_x.to(device)
    fixed_y = (fixed_attrs[:, attr_idx] > 0).float().unsqueeze(1).to(device)

    
    global_step = 0
    best_val_rec = float("inf")

     
    next_save = args.save_every
    next_ckpt = args.ckpt_every

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(args.epochs):
        t0 = time.time()

        # nombre de batches par epoch  
        steps_in_epoch = max(1, args.epoch_size // args.batch_size)
        it = iter(train_loader)

        pbar = tqdm(
            range(steps_in_epoch),
            desc=f"epoch {epoch}/{args.epochs-1}",
            dynamic_ncols=True,
        )

        for _ in pbar:
            try:
                x, attrs = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, attrs = next(it)

            metrics = train_one_step(
                x=x,
                attrs=attrs,
                attr_idx=attr_idx,
                encoder=encoder,
                decoder=decoder,
                discriminator=discriminator,
                opt_ae=opt_ae,
                opt_dis=opt_dis,
                losses=losses,
                step=global_step,                 
                lambda_lat=args.lambda_lat,
                lambda_warmup=args.lambda_warmup,  
                n_dis_steps=args.n_dis_steps,
                clip_grad=args.clip_grad,
                device=device,
            )

            # logging console + fichier
            if global_step % args.log_every == 0:
                pbar.set_postfix({
                    "rec": f"{metrics['loss_rec']:.4f}",
                    "dis": f"{metrics['loss_dis']:.4f}",
                    "adv": f"{metrics['loss_adv']:.4f}",
                    "lam": f"{metrics['lambda']:.2e}",
                    "accD": f"{metrics['acc_dis']:.2f}",
                })

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"step={global_step} rec={metrics['loss_rec']:.6f} dis={metrics['loss_dis']:.6f} "
                        f"adv={metrics['loss_adv']:.6f} lam={metrics['lambda']:.6g} accD={metrics['acc_dis']:.4f}\n"
                    )

                if writer is not None:
                    writer.add_scalar("train/rec", metrics["loss_rec"], global_step)
                    writer.add_scalar("train/dis", metrics["loss_dis"], global_step)
                    writer.add_scalar("train/adv", metrics["loss_adv"], global_step)
                    writer.add_scalar("train/lambda", metrics["lambda"], global_step)
                    writer.add_scalar("train/acc_dis", metrics["acc_dis"], global_step)

            # ✅ images monitoring (seuil >=)
            if global_step >= next_save:
                out_img = paths.samples / f"monitor_step_{global_step:07d}.png"
                save_monitor_grid(encoder, decoder, fixed_x, fixed_y, out_img)
                next_save += args.save_every

            # ✅ checkpoint (seuil >=)
            if global_step >= next_ckpt:
                ckpt = {
                    "epoch": epoch,
                    "step": global_step,
                    "attr_name": args.attr,
                    "attr_idx": attr_idx,
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "opt_ae": opt_ae.state_dict(),
                    "opt_dis": opt_dis.state_dict(),
                    "args": vars(args),
                }
                save_checkpoint(paths.checkpoints / f"ckpt_{global_step:07d}.pth", ckpt)
                save_checkpoint(paths.checkpoints / "last.pth", ckpt)
                next_ckpt += args.ckpt_every

           
            global_step += 1

       
        # Validation reconstruction
          
        val_rec = validate_reconstruction(
            valid_loader=valid_loader,
            attr_idx=attr_idx,
            encoder=encoder,
            decoder=decoder,
            losses=losses,
            device=device,
        )

        epoch_time = (time.time() - t0) / 60.0
        msg = f"[VAL] epoch={epoch} val_rec={val_rec:.6f} time_min={epoch_time:.1f}"
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if writer is not None:
            writer.add_scalar("val/rec", val_rec, epoch)

        # -----------------------------
        # Save best + export final
        # -----------------------------
        if val_rec < best_val_rec:
            best_val_rec = val_rec

            best_ckpt = {
                "epoch": epoch,
                "step": global_step,
                "attr_name": args.attr,
                "attr_idx": attr_idx,
                "best_val_rec": best_val_rec,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_ae": opt_ae.state_dict(),
                "opt_dis": opt_dis.state_dict(),  
                "args": vars(args),
            }
            save_checkpoint(paths.checkpoints / "best_rec.pth", best_ckpt)

            export_name = f"{args.attr.lower()}.pth"
            export_trained_model(
                paths.exports / export_name,
                attr_name=args.attr,
                encoder=encoder,
                decoder=decoder,
                meta={
                    "best_val_rec": best_val_rec,
                    "epoch": epoch,
                    "step": global_step,
                    "alpha_min": -2.0,
                    "alpha_max": 2.0,
                },
            )

    if writer is not None:
        writer.close()

    print(f"Training terminé. Modèle exporté dans: {paths.exports / (args.attr.lower() + '.pth')}")


if __name__ == "__main__":
    main()
