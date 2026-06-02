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
# This file implements inference and evaluation:
#   Section IV-C  — Quantitative results (Table I)
#   Section V     — Ablation Study (Table II)
#   Appendix      — Inference Process
#
#   Usage:
#     python scripts/evaluate.py \
#       --checkpoint checkpoints/kao_massachusetts_iter0250000.pt \
#       --dataset massachusetts \
#       --data_root /data/massachusetts_roads \
#       --output_dir results/
# =============================================================================
# MIT License — Copyright (c) 2026 Teerapong Panboonyuen
# =============================================================================

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torchvision.utils as vutils
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from kao.model     import KAO
from kao.diffusion import KAODiffusion
from kao.metrics   import KAOEvaluator
from datasets.satellite import build_dataloader

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(asctime)s] %(levelname)s — %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("KAO-Eval")


def parse_args():
    p = argparse.ArgumentParser(description="KAO Evaluation — IEEE TGRS 2025")
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--dataset",      type=str, default="massachusetts",
                   choices=["massachusetts", "deepglobe", "generic"])
    p.add_argument("--data_root",    type=str, required=True)
    p.add_argument("--split",        type=str, default="val")
    p.add_argument("--image_size",   type=int, default=256)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_timesteps",type=int, default=1000)
    p.add_argument("--output_dir",   type=str, default="./results")
    p.add_argument("--save_images",  action="store_true",
                   help="Save inpainted images to output_dir")
    p.add_argument("--num_samples",  type=int, default=None,
                   help="Evaluate on subset (None = full dataset)")
    return p.parse_args()


@torch.no_grad()
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device    : {device}")
    log.info(f"Checkpoint: {args.checkpoint}")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    ckpt  = torch.load(args.checkpoint, map_location=device)
    cfg   = ckpt.get("args", {})

    model = KAO(
        in_channels      = 3,
        latent_channels  = cfg.get("latent_channels", 256),
        tpt_depth        = cfg.get("tpt_depth",        4),
        tpt_heads        = cfg.get("tpt_heads",        8),
        tpt_pyramid_lvls = cfg.get("tpt_pyramid",      3),
        num_ep_modules   = cfg.get("num_ep",           2),
        image_size       = args.image_size,
    ).to(device).eval()

    model.load_state_dict(ckpt["model"])
    log.info("Model loaded ✓")

    diffusion = KAODiffusion(num_timesteps=args.num_timesteps).to(device)

    # -----------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------
    loader = build_dataloader(
        dataset_name = args.dataset,
        root         = args.data_root,
        split        = args.split,
        image_size   = args.image_size,
        batch_size   = args.batch_size,
        num_workers  = 2,
        augment      = False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    evaluator = KAOEvaluator(device=device)

    total_samples = 0
    log.info("Running KAO inference loop (Algorithm 1)...")

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        images = batch["image"].to(device)   # (B,3,H,W)
        masks  = batch["mask"].to(device)    # (B,1,H,W)
        masked = batch["masked"].to(device)  # (B,3,H,W)
        names  = batch["filename"]

        # -----------------------------------------------------------------
        # Appendix — Inference Process:
        # "Given a masked image x_cond and binary mask m, the model
        #  predicts the missing regions x_inf by iterating through the
        #  diffusion process."
        # -----------------------------------------------------------------
        inpainted = diffusion.p_sample_loop(
            model    = model,
            x_cond   = masked,
            mask     = masks,
            shape    = images.shape,
            device   = device,
            verbose  = False,
        )

        evaluator.update(inpainted, images)

        # Save qualitative results
        if args.save_images:
            for i in range(images.shape[0]):
                # Grid: masked | inpainted | ground truth
                grid = vutils.make_grid(
                    torch.stack([masked[i], inpainted[i], images[i]]),
                    nrow=3, normalize=True, value_range=(-1, 1)
                )
                stem = Path(names[i]).stem
                out_path = os.path.join(args.output_dir, f"{stem}_kao.png")
                vutils.save_image(grid, out_path)

        total_samples += images.shape[0]
        if args.num_samples and total_samples >= args.num_samples:
            break

    evaluator.print_results(dataset_name=args.dataset.upper())

    results = evaluator.compute()
    # Write summary
    summary_path = os.path.join(args.output_dir, "metrics.txt")
    with open(summary_path, "w") as f:
        f.write(f"KAO Evaluation — {args.dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Samples   : {total_samples}\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")
    log.info(f"Results saved to {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
