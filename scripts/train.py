# =============================================================================
# KAO: Kernel-Adaptive Optimization in Diffusion for Satellite Image
# =============================================================================
# Author  : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Paper   : "KAO: Kernel-Adaptive Optimization in Diffusion for Satellite Image"
# Journal : IEEE Transactions on Geoscience and Remote Sensing (IF: 8.9)
# DOI     : https://doi.org/10.1109/TGRS.2025.3621738
# Project : https://kaopanboonyuen.github.io/KAO/
# ORCID   : https://orcid.org/0000-0001-8464-4476
# =============================================================================
# This file implements the training procedure described in:
#   Appendix — Training Setup:
#     • 1000 diffusion timesteps
#     • Batch size 16, single NVIDIA A40 GPU
#     • AdamW, lr = 5e-5, weight decay = 0.01
#     • Linear warmup (first 10% of iterations)
#     • Cosine decay schedule
#     • 250,000 total training iterations
# =============================================================================
# MIT License — Copyright (c) 2025 Teerapong Panboonyuen
# =============================================================================

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# KAO imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from kao.model     import KAO
from kao.diffusion import KAODiffusion
from datasets.satellite import build_dataloader

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(asctime)s] %(levelname)s — %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("KAO-Train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="KAO Training — IEEE TGRS 2025"
    )
    # Dataset
    p.add_argument("--dataset",     type=str, default="massachusetts",
                   choices=["massachusetts", "deepglobe", "generic"],
                   help="Dataset name (Section IV-B)")
    p.add_argument("--data_root",   type=str, required=True,
                   help="Path to dataset root directory")
    p.add_argument("--image_size",  type=int, default=256,
                   help="Spatial resolution for training")
    p.add_argument("--mask_type",   type=str, default="random",
                   choices=["random", "irregular"],
                   help="Mask generation strategy (Appendix)")

    # Model
    p.add_argument("--latent_channels", type=int, default=256,
                   help="Latent feature channels")
    p.add_argument("--tpt_depth",       type=int, default=4,
                   help="TPT transformer depth (Section III-D)")
    p.add_argument("--tpt_heads",       type=int, default=8,
                   help="TPT attention heads")
    p.add_argument("--tpt_pyramid",     type=int, default=3,
                   help="TPT pyramid levels")
    p.add_argument("--num_ep",          type=int, default=2,
                   choices=[1, 2],
                   help="Number of Explicit Propagation modules (Ablation Table II)")
    p.add_argument("--kernel_sigma",    type=float, default=1.0,
                   help="Initial RBF kernel bandwidth (Section III-C)")

    # Diffusion
    p.add_argument("--num_timesteps",   type=int,   default=1000,
                   help="Diffusion steps T (Appendix training setup)")
    p.add_argument("--beta_schedule",   type=str,   default="cosine",
                   choices=["linear", "cosine"])

    # Training  (Appendix training setup)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=5e-5,
                   help="Initial learning rate (Appendix: 5e-5)")
    p.add_argument("--weight_decay",    type=float, default=0.01,
                   help="AdamW weight decay (Appendix: 0.01)")
    p.add_argument("--total_iters",     type=int,   default=250_000,
                   help="Total training iterations (Appendix: 250k)")
    p.add_argument("--warmup_ratio",    type=float, default=0.10,
                   help="Fraction of iters for LR warmup (Appendix: 10%)")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--log_every",       type=int,   default=100)
    p.add_argument("--save_every",      type=int,   default=5_000)
    p.add_argument("--output_dir",      type=str,   default="./checkpoints")
    p.add_argument("--resume",          type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device : {device}")
    log.info(f"Dataset: {args.dataset} @ {args.data_root}")

    # -----------------------------------------------------------------------
    # Data  (Section IV-B)
    # -----------------------------------------------------------------------
    train_loader = build_dataloader(
        dataset_name = args.dataset,
        root         = args.data_root,
        split        = "train",
        image_size   = args.image_size,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        mask_type    = args.mask_type,
    )
    log.info(f"Train batches per epoch: {len(train_loader)}")

    # -----------------------------------------------------------------------
    # Model  (Section III-D)
    # -----------------------------------------------------------------------
    model = KAO(
        in_channels      = 3,
        latent_channels  = args.latent_channels,
        tpt_depth        = args.tpt_depth,
        tpt_heads        = args.tpt_heads,
        tpt_pyramid_lvls = args.tpt_pyramid,
        num_ep_modules   = args.num_ep,
        kernel_sigma     = args.kernel_sigma,
        image_size       = args.image_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(f"KAO model: {n_params:.2f} M trainable parameters")

    # -----------------------------------------------------------------------
    # Diffusion  (Section III-A)
    # -----------------------------------------------------------------------
    diffusion = KAODiffusion(
        num_timesteps = args.num_timesteps,
        schedule      = args.beta_schedule,
    ).to(device)

    # -----------------------------------------------------------------------
    # Optimiser & LR schedule  (Appendix training setup)
    # AdamW, lr=5e-5, weight_decay=0.01
    # Linear warmup (10%) + cosine decay
    # -----------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
    )

    warmup_iters = int(args.total_iters * args.warmup_ratio)
    warmup_sched = LinearLR(optimizer,
                            start_factor = 1e-6 / args.lr,
                            end_factor   = 1.0,
                            total_iters  = warmup_iters)
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max = args.total_iters - warmup_iters,
                                     eta_min = args.lr * 0.01)
    scheduler    = SequentialLR(optimizer,
                                schedulers  = [warmup_sched, cosine_sched],
                                milestones  = [warmup_iters])

    # -----------------------------------------------------------------------
    # Optionally resume
    # -----------------------------------------------------------------------
    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = ckpt.get("iteration", 0)
        log.info(f"Resumed from {args.resume} (iter {start_iter})")

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training loop  (Algorithm 1 — outer training iteration)
    # -----------------------------------------------------------------------
    model.train()
    iteration  = start_iter
    epoch      = 0
    total_loss = 0.0
    t0         = time.time()

    log.info(f"Starting training — {args.total_iters:,} iterations")
    log.info(f"Warmup: {warmup_iters:,} iters  |  "
             f"Batch: {args.batch_size}  |  LR: {args.lr}")

    while iteration < args.total_iters:
        epoch += 1
        for batch in train_loader:
            if iteration >= args.total_iters:
                break

            images = batch["image"].to(device)  # (B, 3, H, W) in [-1,1]
            masks  = batch["mask"].to(device)   # (B, 1, H, W) binary

            # KAO-modulated KL loss (Section III-C, Eq. 9)
            loss = diffusion.kao_loss(model, images, masks, model.kernel)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            iteration  += 1

            # Logging
            if iteration % args.log_every == 0:
                avg_loss = total_loss / args.log_every
                lr_now   = optimizer.param_groups[0]["lr"]
                elapsed  = time.time() - t0
                its_sec  = args.log_every / max(elapsed, 1e-8)
                log.info(
                    f"Iter {iteration:7d}/{args.total_iters:,} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_now:.2e} | "
                    f"Epoch: {epoch} | "
                    f"{its_sec:.1f} it/s"
                )
                total_loss = 0.0
                t0 = time.time()

            # Checkpoint
            if iteration % args.save_every == 0 or iteration == args.total_iters:
                ckpt_path = os.path.join(
                    args.output_dir, f"kao_{args.dataset}_iter{iteration:07d}.pt"
                )
                torch.save({
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "iteration":  iteration,
                    "args":       vars(args),
                }, ckpt_path)
                log.info(f"  ✓ Checkpoint saved: {ckpt_path}")

    log.info("Training complete ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
