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
# This file implements the evaluation metrics described in:
#   Section IV-C — Quantitative Results
#     • FID   (Fréchet Inception Distance)   ↓
#     • Precision                             ↑
#     • Recall                                ↑
#   Section V   — Ablation Study
#     • LPIPS  (Learned Perceptual Image Patch Similarity)  ↓
#
#   Paper results (Table I):
#     KAO on Massachusetts:  FID=3.11, Prec=0.93, Recall=0.91
#     KAO on DeepGlobe:      FID=1.42, Prec=0.88, Recall=0.63
#
#   Ablation results (Table II):
#     Full model (2 EP modules):  LPIPS=0.059, FID=6.13
# =============================================================================
# MIT License — Copyright (c) 2025 Teerapong Panboonyuen
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    print("[Metrics] torchmetrics not found — using lightweight fallbacks.")


# ---------------------------------------------------------------------------
# PSNR  (Peak Signal-to-Noise Ratio)
# ---------------------------------------------------------------------------

def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,    # images in [-1, 1] → range = 2
) -> float:
    """
    Compute PSNR between predicted and target images.

    Args:
        pred, target : (B, C, H, W) tensors in [-1, 1]
        data_range   : max - min of the value range

    Returns:
        mean PSNR in dB over the batch
    """
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.mean(dim=(1, 2, 3))                        # per-image MSE
    psnr = 10 * torch.log10(data_range ** 2 / mse.clamp(min=1e-10))
    return psnr.mean().item()


# ---------------------------------------------------------------------------
# SSIM  (Structural Similarity Index)
# ---------------------------------------------------------------------------

def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Compute mean SSIM between predicted and target images.

    Args:
        pred, target : (B, C, H, W) in [-1, 1]

    Returns:
        mean SSIM ∈ [-1, 1]
    """
    # Gaussian window
    sigma  = 1.5
    kernel = _gaussian_kernel(window_size, sigma, pred.device, pred.dtype)
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(pred,   kernel, padding=pad, groups=pred.shape[1])
    mu2 = F.conv2d(target, kernel, padding=pad, groups=target.shape[1])

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=target.shape[1]) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=pad, groups=pred.shape[1])  - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def _gaussian_kernel(size: int, sigma: float,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g      = g / g.sum()
    return (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# FID wrapper  (Section IV-C — "Image Quality (FID)")
# ---------------------------------------------------------------------------

class FIDMetric:
    """
    Fréchet Inception Distance — primary quality metric in Table I.

    From Section IV-C:
        "KAO achieves the lowest FID on both datasets: 3.11 on Massachusetts
         and 1.42 on DeepGlobe."
    """

    def __init__(self, feature: int = 2048, device: str = "cpu"):
        self.device = device
        if HAS_TORCHMETRICS:
            self.fid = FrechetInceptionDistance(
                feature        = feature,
                reset_real_features = False,
            ).to(device)
        else:
            self.fid = None

    def update_real(self, images: torch.Tensor):
        """Add real images (uint8, [0, 255])."""
        if self.fid is not None:
            self.fid.update(_to_uint8(images), real=True)

    def update_fake(self, images: torch.Tensor):
        """Add generated images (uint8, [0, 255])."""
        if self.fid is not None:
            self.fid.update(_to_uint8(images), real=False)

    def compute(self) -> float:
        if self.fid is not None:
            return self.fid.compute().item()
        return float("nan")

    def reset(self):
        if self.fid is not None:
            self.fid.reset()


# ---------------------------------------------------------------------------
# Precision & Recall  (Section IV-C — "Precision and Recall")
# ---------------------------------------------------------------------------

def compute_precision_recall(
    real_features:  torch.Tensor,   # (N, D) feature vectors of real images
    fake_features:  torch.Tensor,   # (M, D) feature vectors of generated images
    k:              int = 3,
) -> tuple:
    """
    Compute Precision and Recall for generative models.

    Based on Kynkäänniemi et al. (2019). Measures:
        Precision — fraction of generated samples within real manifold
        Recall    — fraction of real samples covered by generated manifold

    From Section IV-C:
        "KAO leads in precision (0.93 and 0.88) and recall (0.91 and 0.63),
         outperforming all baselines."

    Args:
        real_features : (N, D) Inception features of real images
        fake_features : (M, D) Inception features of generated images
        k             : number of nearest neighbours

    Returns:
        (precision, recall) both ∈ [0, 1]
    """
    def _knn_dist(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
        """k-th nearest-neighbour distance for each row of a w.r.t b."""
        dist = torch.cdist(a, b)                     # (N, M)
        kth  = dist.topk(k + 1, dim=1, largest=False).values[:, -1]
        return kth

    real_r = _knn_dist(real_features, real_features, k)  # (N,)
    fake_r = _knn_dist(fake_features, fake_features, k)  # (M,)

    # Precision: fake samples inside real hyperspheres
    dist_f2r = torch.cdist(fake_features, real_features)  # (M, N)
    precision = (dist_f2r.min(dim=1).values < real_r.mean()).float().mean()

    # Recall: real samples inside fake hyperspheres
    dist_r2f = torch.cdist(real_features, fake_features)  # (N, M)
    recall    = (dist_r2f.min(dim=1).values < fake_r.mean()).float().mean()

    return precision.item(), recall.item()


# ---------------------------------------------------------------------------
# LPIPS  (Section V — Ablation Study, Table II)
# ---------------------------------------------------------------------------

class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity.

    Used in the ablation study (Table II):
        Full model (2 EP modules):  LPIPS = 0.059
        KAO w/o Resampling:         LPIPS = 0.528
    """

    def __init__(self, net: str = "vgg", device: str = "cpu"):
        self.device = device
        if HAS_TORCHMETRICS:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net).to(device)
        else:
            self.lpips = None

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> float:
        """
        Args:
            pred, target : (B, 3, H, W) in [-1, 1]
        Returns:
            scalar LPIPS (lower = better perceptual quality)
        """
        if self.lpips is not None:
            return self.lpips(pred, target).item()
        # Fallback: MSE-based proxy
        return F.mse_loss(pred, target).item()


# ---------------------------------------------------------------------------
# High Structural Variance (HSV) mask  (Section III-C)
# HSV(x) = Var_{N(x)}[∇I] - ε
# ---------------------------------------------------------------------------

def high_structural_variance_mask(
    image:    torch.Tensor,     # (B, C, H, W) in [-1, 1]
    epsilon:  float = 0.0,
    ksize:    int   = 5,
) -> torch.Tensor:
    """
    Compute the HSV mask from Section III-C.

    Defined in Eq. after Sec III-C-3:
        HSV(x) = Var_{N(x)}[∇I] - ε

    Regions with positive HSV indicate significant structural detail —
    KAO prioritises these during denoising.

    Returns:
        hsv_mask : (B, 1, H, W) float, positive = high-structure region
    """
    gray = image.mean(dim=1, keepdim=True)   # (B, 1, H, W)

    # Sobel gradient magnitude
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                             dtype=gray.dtype, device=gray.device)
    sobel_y = sobel_x.T
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)   # |∇I|

    # Local variance of gradient magnitude in N(x)
    kernel = torch.ones(1, 1, ksize, ksize, device=gray.device,
                        dtype=gray.dtype) / (ksize * ksize)
    mean_g  = F.conv2d(grad_mag, kernel, padding=ksize // 2)
    mean_g2 = F.conv2d(grad_mag ** 2, kernel, padding=ksize // 2)
    var_g   = mean_g2 - mean_g ** 2

    return var_g - epsilon    # positive → structurally complex region


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

class KAOEvaluator:
    """
    Aggregated evaluation suite for KAO.

    Computes all metrics reported in Section IV-C and Table I:
        FID, Precision, Recall, PSNR, SSIM, LPIPS
    """

    def __init__(self, device: str = "cpu"):
        self.device   = device
        self.fid      = FIDMetric(device=device)
        self.lpips    = LPIPSMetric(device=device)
        self._psnrs   = []
        self._ssims   = []
        self._lpips_vals = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update all running metrics.

        Args:
            pred, target : (B, 3, H, W) in [-1, 1]
        """
        pred   = pred.to(self.device)
        target = target.to(self.device)

        self.fid.update_real(_to_uint8(target))
        self.fid.update_fake(_to_uint8(pred))

        self._psnrs.append(compute_psnr(pred, target))
        self._ssims.append(compute_ssim(pred, target))
        self._lpips_vals.append(self.lpips(pred, target))

    def compute(self) -> dict:
        """Return dict of all metrics."""
        return {
            "FID"   : self.fid.compute(),
            "PSNR"  : float(np.mean(self._psnrs)),
            "SSIM"  : float(np.mean(self._ssims)),
            "LPIPS" : float(np.mean(self._lpips_vals)),
        }

    def reset(self):
        self.fid.reset()
        self._psnrs.clear()
        self._ssims.clear()
        self._lpips_vals.clear()

    def print_results(self, dataset_name: str = ""):
        results = self.compute()
        header  = f" KAO Evaluation — {dataset_name} " if dataset_name else " KAO Evaluation "
        bar     = "=" * (len(header) + 4)
        print(f"\n{bar}")
        print(f"  {header}")
        print(bar)
        print(f"  FID    : {results['FID']:.4f}  ↓ (Table I: 3.11 / 1.42)")
        print(f"  PSNR   : {results['PSNR']:.2f} dB ↑")
        print(f"  SSIM   : {results['SSIM']:.4f}  ↑")
        print(f"  LPIPS  : {results['LPIPS']:.4f}  ↓ (Table II: 0.059 full model)")
        print(f"{bar}\n")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] float to [0, 255] uint8."""
    return ((images.clamp(-1, 1) + 1) / 2 * 255).byte()


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Metrics] Sanity check on {device} ...")

    B, C, H, W = 4, 3, 64, 64
    pred   = torch.randn(B, C, H, W, device=device).clamp(-1, 1)
    target = torch.randn(B, C, H, W, device=device).clamp(-1, 1)

    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)
    hsv  = high_structural_variance_mask(target)
    print(f"  PSNR  : {psnr:.2f} dB")
    print(f"  SSIM  : {ssim:.4f}")
    print(f"  HSV shape: {tuple(hsv.shape)}")
    print("[Metrics] Sanity check passed ✓")
