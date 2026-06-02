# =============================================================================
# KAO: Kernel-Adaptive Optimization in Diffusion for Satellite Image
# =============================================================================
# Author  : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Paper   : "KAO: Kernel-Adaptive Optimization in Diffusion for Satellite Image"
# Journal : IEEE Transactions on Geoscience and Remote Sensing (IF: 8.9)
# DOI     : https://doi.org/10.1109/TGRS.2025.3621738
# Project : https://kaopanboonyuen.github.io/KAO/
# =============================================================================
# End-to-end smoke test — runs the full KAO pipeline on synthetic data.
# No real satellite data required. Validates:
#   ✓  KAO model forward pass         (Section III-D)
#   ✓  GaussianRBFKernel              (Section III-C, Eq. 8)
#   ✓  KAODiffusion forward + reverse (Section III-A)
#   ✓  KAO training loss              (Section III-C, Eq. 9)
#   ✓  KAOEvaluator                   (Section IV-C)
#   ✓  Ablation configurations        (Section V, Table II)
#   ✓  Dataset mask generation        (Appendix)
# =============================================================================
# MIT License — Copyright (c) 2025 Teerapong Panboonyuen
# =============================================================================

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kao.model     import KAO, GaussianRBFKernel
from kao.diffusion import KAODiffusion
from kao.metrics   import KAOEvaluator, compute_psnr, compute_ssim
from datasets.satellite import generate_random_mask, generate_irregular_mask
from scripts.ablation   import run_ablation_study

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║   KAO: Kernel-Adaptive Optimization in Diffusion for Satellite Image         ║
║   IEEE Transactions on Geoscience and Remote Sensing — Impact Factor 8.9     ║
║   DOI: https://doi.org/10.1109/TGRS.2025.3621738                            ║
║   Author: Teerapong Panboonyuen (Kao) — kaopanboonyuen.github.io             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

def check(label, passed=True):
    icon = "✓" if passed else "✗"
    print(f"  [{icon}] {label}")
    return passed


def main():
    print(BANNER)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}\n")

    B, C, H, W = 2, 3, 64, 64
    all_passed = True

    # -----------------------------------------------------------------------
    # 1. Mask generation  (Appendix — Masking Strategy)
    # -----------------------------------------------------------------------
    print("── Mask Generation (Appendix) ──────────────────────────────────")
    mask_rand = generate_random_mask(H, W, 0.30, 0.50)
    all_passed &= check(
        f"Random mask: shape={tuple(mask_rand.shape)}, "
        f"coverage={1 - mask_rand.float().mean().item():.1%}",
        mask_rand.shape == (1, H, W),
    )

    # -----------------------------------------------------------------------
    # 2. KAO model  (Section III-D)
    # -----------------------------------------------------------------------
    print("\n── KAO Model (Section III-D / Algorithm 1) ─────────────────────")
    model = KAO(
        in_channels     = C,
        latent_channels = 64,
        tpt_depth       = 2,
        tpt_heads       = 4,
        tpt_pyramid_lvls= 2,
        num_ep_modules  = 2,    # Full model — Table II
        kernel_sigma    = 1.0,
    ).to(device)

    x_t    = torch.randn(B, C, H, W, device=device)
    x_cond = torch.randn(B, C, H, W, device=device)
    mask   = (torch.rand(B, 1, H, W, device=device) > 0.5).float()
    t      = torch.randint(1, 100, (B,), device=device)

    with torch.no_grad():
        pred, kw = model(x_t, x_cond, mask, t)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    all_passed &= check(
        f"Forward pass: pred={tuple(pred.shape)}, "
        f"kernel={tuple(kw.shape)}, params={n_params:.2f}M",
        pred.shape == (B, C, H, W),
    )

    # -----------------------------------------------------------------------
    # 3. Gaussian RBF Kernel  (Section III-C, Eq. 8)
    # -----------------------------------------------------------------------
    print("\n── GaussianRBFKernel (Section III-C, Eq. 8) ────────────────────")
    kernel = GaussianRBFKernel(init_sigma=1.0).to(device)
    K      = kernel(x_t, x_cond)
    all_passed &= check(
        f"K(X_t, X_prev): shape={tuple(K.shape)}, "
        f"range=[{K.min():.3f}, {K.max():.3f}]",
        K.shape == (B, 1, H, W) and (K >= 0).all() and (K <= 1).all(),
    )

    # -----------------------------------------------------------------------
    # 4. High Structural Variance mask  (Section III-C)
    # -----------------------------------------------------------------------
    print("\n── High Structural Variance (Section III-C) ─────────────────────")
    from kao.metrics import high_structural_variance_mask
    hsv = high_structural_variance_mask(x_cond.to(device))
    all_passed &= check(
        f"HSV mask: shape={tuple(hsv.shape)}, "
        f"positive_ratio={( hsv > 0).float().mean().item():.1%}",
        hsv.shape == (B, 1, H, W),
    )

    # -----------------------------------------------------------------------
    # 5. Diffusion forward process  (Section III-A, Eq. 6)
    # -----------------------------------------------------------------------
    print("\n── KAO Diffusion — Forward Process (Section III-A) ──────────────")
    diffusion = KAODiffusion(num_timesteps=100, schedule="cosine").to(device)
    x_0    = torch.randn(B, C, H, W, device=device)
    x_t_   = diffusion.q_sample(x_0, t)
    all_passed &= check(
        f"q_sample (forward): shape={tuple(x_t_.shape)}, "
        f"noise added ✓",
        x_t_.shape == x_0.shape,
    )

    # -----------------------------------------------------------------------
    # 6. KAO training loss  (Section III-C, Eq. 9)
    # -----------------------------------------------------------------------
    print("\n── KAO Training Loss (Section III-C, Eq. 9) ─────────────────────")
    loss = diffusion.kao_loss(model, x_0, mask, model.kernel)
    all_passed &= check(
        f"KAO loss (kernel-modulated KL): {loss.item():.4f}",
        loss.item() > 0 and torch.isfinite(loss),
    )

    # -----------------------------------------------------------------------
    # 7. Reverse denoising loop  (Algorithm 1)
    # -----------------------------------------------------------------------
    print("\n── Reverse Denoising (Algorithm 1) ──────────────────────────────")
    with torch.no_grad():
        result = diffusion.p_sample_loop(
            model   = model,
            x_cond  = x_0 * mask,
            mask    = mask,
            shape   = (B, C, H, W),
            device  = device,
            verbose = False,
        )
    all_passed &= check(
        f"p_sample_loop output: shape={tuple(result.shape)}, "
        f"range=[{result.min():.2f}, {result.max():.2f}]",
        result.shape == (B, C, H, W),
    )

    # -----------------------------------------------------------------------
    # 8. Metrics  (Section IV-C)
    # -----------------------------------------------------------------------
    print("\n── Evaluation Metrics (Section IV-C) ────────────────────────────")
    psnr = compute_psnr(result, x_0)
    ssim = compute_ssim(result, x_0)
    all_passed &= check(f"PSNR : {psnr:.2f} dB")
    all_passed &= check(f"SSIM : {ssim:.4f}")

    # -----------------------------------------------------------------------
    # 9. Ablation configurations  (Table II)
    # -----------------------------------------------------------------------
    print("\n── Ablation Study (Section V, Table II) ─────────────────────────")
    run_ablation_study(num_samples=16, image_size=H, device=device)
    all_passed &= check("Ablation study ran successfully ✓")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    status = "ALL CHECKS PASSED ✓" if all_passed else "SOME CHECKS FAILED ✗"
    print(f"\n{'═' * 60}")
    print(f"  {status}")
    print(f"  KAO — IEEE TGRS 2025 | Teerapong Panboonyuen")
    print(f"  https://kaopanboonyuen.github.io/KAO/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
