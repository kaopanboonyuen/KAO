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
# This file reproduces the ablation study from:
#   Section V — Ablation Study and Analysis (Table II)
#
#   Configurations evaluated:
#     1. KAO w/o Resampling               LPIPS=0.528  FID=13.28
#     2. KAO w/ Latent Space Cond. only   LPIPS=0.297  FID=11.44
#     3. KAO w/ Single Propagation Module LPIPS=0.118  FID=8.93
#     4. KAO w/ Two Propagation Modules   LPIPS=0.059  FID=6.13  (Full)
#
#   Section V-A: "Evaluating the Role of Kernel Resampling and
#                 Latent Conditioning"
# =============================================================================
# MIT License — Copyright (c) 2025 Teerapong Panboonyuen
# =============================================================================

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from kao.model     import KAO, LatentSpaceConditioning, ExplicitPropagation
from kao.metrics   import LPIPSMetric, compute_psnr, compute_ssim


# ---------------------------------------------------------------------------
# Ablation variant factory
# Implements the four configurations from Table II
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    name:            str
    use_resampling:  bool = True
    use_lsc:         bool = True
    num_ep_modules:  int  = 2       # 0, 1, or 2
    # Expected results from Table II
    expected_lpips:  float = 0.0
    expected_fid:    float = 0.0


ABLATION_CONFIGS = [
    AblationConfig(
        name            = "KAO w/o Resampling",
        use_resampling  = False,
        use_lsc         = False,
        num_ep_modules  = 0,
        expected_lpips  = 0.528,
        expected_fid    = 13.28,
    ),
    AblationConfig(
        name            = "KAO w/ Latent Space Conditioning only",
        use_resampling  = True,
        use_lsc         = True,
        num_ep_modules  = 0,
        expected_lpips  = 0.297,
        expected_fid    = 11.44,
    ),
    AblationConfig(
        name            = "KAO w/ Single Propagation Module",
        use_resampling  = True,
        use_lsc         = True,
        num_ep_modules  = 1,
        expected_lpips  = 0.118,
        expected_fid    = 8.93,
    ),
    AblationConfig(
        name            = "KAO w/ Two Propagation Modules (Full Model)",
        use_resampling  = True,
        use_lsc         = True,
        num_ep_modules  = 2,
        expected_lpips  = 0.059,
        expected_fid    = 6.13,
    ),
]


# ---------------------------------------------------------------------------
# Ablation-mode KAO forward
# We reuse KAO but optionally disable LSC / EP components
# ---------------------------------------------------------------------------

class KAOAblation(KAO):
    """
    KAO with configurable component ablations.

    Implements the ablation configurations described in Table II
    (Section V — Ablation Study and Analysis).

    Args:
        use_resampling : If False, skip kernel-weighted denoising
        use_lsc        : If False, skip Latent Space Conditioning
        num_ep_modules : 0 = no EP, 1 = single EP, 2 = full model
    """

    def __init__(
        self,
        use_resampling: bool = True,
        use_lsc:        bool = True,
        num_ep_modules: int  = 2,
        **kwargs,
    ):
        # Always init full KAO; we'll selectively bypass components
        super().__init__(num_ep_modules=num_ep_modules, **kwargs)
        self.use_resampling = use_resampling
        self.use_lsc        = use_lsc
        # Override EP modules based on ablation
        if num_ep_modules == 0:
            self.ep_modules = nn.ModuleList([])

    def single_step(self, x_t, x_cond, mask, t):
        """Ablation-aware forward pass."""
        B = x_t.shape[0]
        device = x_t.device

        inp    = torch.cat([x_t, x_cond, mask], dim=1)
        h_infr = self.encoder(inp)
        t_emb  = self._timestep_embedding(t)
        h_infr = h_infr + t_emb[:, :, None, None]
        h_infr = self.tpt(h_infr)

        inp_c  = torch.cat([x_cond, x_cond, mask], dim=1)
        h_cond = self.encoder(inp_c)
        h_cond = self.tpt(h_cond)

        # --- Latent Space Conditioning (ablation: skip if not use_lsc) ---
        if self.use_lsc:
            h_star = self.lsc(h_infr, h_cond, mask)
        else:
            h_star = h_infr    # no conditioning

        # --- Explicit Propagation (ablation: num_ep_modules controls depth) ---
        for ep in self.ep_modules:
            h_hat  = ep(h_star, mask)
            h_star = h_star + h_hat

        x_pred = self.decoder(h_star)

        # --- Resampling / blending (ablation: skip kernel weighting) ---
        if self.use_resampling:
            x_pred = x_pred * (1 - mask) + x_cond * mask
            k_weight = self.kernel(x_t, x_pred)
        else:
            # No resampling: raw decoder output, uniform kernel weights
            k_weight = torch.ones_like(x_pred[:, :1])

        return x_pred, k_weight


# ---------------------------------------------------------------------------
# Run ablation study
# ---------------------------------------------------------------------------

def run_ablation_study(
    num_samples: int  = 64,
    image_size:  int  = 64,
    device:      str  = "cpu",
):
    """
    Run and print the KAO ablation study (Table II).

    Args:
        num_samples : synthetic images to evaluate on
        image_size  : spatial resolution
        device      : 'cuda' or 'cpu'

    Returns:
        dict of {config_name: {LPIPS, PSNR, SSIM}}
    """
    lpips_fn = LPIPSMetric(device=device)

    # Synthetic data (replace with real loader for paper-level results)
    torch.manual_seed(42)
    images_gt = torch.randn(num_samples, 3, image_size, image_size,
                            device=device).clamp(-1, 1)
    masks     = (torch.rand(num_samples, 1, image_size, image_size,
                             device=device) > 0.5).float()
    masked    = images_gt * masks

    results = {}
    header  = f"{'Configuration':<50} {'LPIPS↓':>8} {'PSNR↑':>8} {'SSIM↑':>8}"
    sep     = "─" * len(header)

    print("\n" + "=" * len(header))
    print("  KAO Ablation Study — Table II (Section V)")
    print("  IEEE TGRS 2025 | Teerapong Panboonyuen")
    print("=" * len(header))
    print(header)
    print(sep)

    for cfg in ABLATION_CONFIGS:
        model = KAOAblation(
            in_channels     = 3,
            latent_channels = 64,
            tpt_depth       = 1,
            tpt_heads       = 4,
            tpt_pyramid_lvls= 2,
            use_resampling  = cfg.use_resampling,
            use_lsc         = cfg.use_lsc,
            num_ep_modules  = cfg.num_ep_modules,
        ).to(device).eval()

        lpips_vals, psnr_vals, ssim_vals = [], [], []

        batch_size = 8
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                imgs   = images_gt[i:i + batch_size]
                msk    = masks    [i:i + batch_size]
                mskd   = masked   [i:i + batch_size]
                t_rand = torch.randint(1, 100, (imgs.shape[0],), device=device)

                pred, _ = model(imgs + 0.1 * torch.randn_like(imgs), mskd, msk, t_rand)
                pred    = pred.clamp(-1, 1)

                lpips_vals.append(lpips_fn(pred, imgs))
                psnr_vals.append(compute_psnr(pred, imgs))
                ssim_vals.append(compute_ssim(pred, imgs))

        import numpy as np
        mean_lpips = np.mean(lpips_vals)
        mean_psnr  = np.mean(psnr_vals)
        mean_ssim  = np.mean(ssim_vals)

        results[cfg.name] = {
            "LPIPS": mean_lpips,
            "PSNR":  mean_psnr,
            "SSIM":  mean_ssim,
            "expected_lpips": cfg.expected_lpips,
            "expected_fid":   cfg.expected_fid,
        }

        marker = " ← Full Model" if cfg.num_ep_modules == 2 and cfg.use_lsc else ""
        print(f"  {cfg.name:<48} {mean_lpips:>8.4f} {mean_psnr:>8.2f} {mean_ssim:>8.4f}{marker}")

    print(sep)
    print("  Note: FID requires full dataset; run evaluate.py for paper-level numbers.")
    print("  Paper Table II expected (LPIPS): 0.528 → 0.297 → 0.118 → 0.059")
    print("=" * len(header) + "\n")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="KAO Ablation Study — Table II")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--image_size",  type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_ablation_study(args.num_samples, args.image_size, device)
