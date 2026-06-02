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
# This file implements the core KAO model architecture described in:
#   Section III-C  — Kernel-Adaptive Optimization in Diffusion
#   Section III-D  — KAO Theoretical Foundations (Algorithm 1)
#   Appendix       — Latent Space Conditioning & Explicit Propagation
# =============================================================================
# MIT License
# Copyright (c) 2025 Teerapong Panboonyuen
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Gaussian RBF Kernel  (Section III-C, Eq. 8 in the paper)
# K(X_t, X_{t-1}) = exp(-||X_t - X_{t-1}||^2 / 2σ^2)
# ---------------------------------------------------------------------------
class GaussianRBFKernel(nn.Module):
    """
    Adaptive Gaussian Radial Basis Function Kernel.

    Implements Equation (8) from Section III-C:
        K(X_t, X_{t-1}) = exp(-||X_t - X_{t-1}||^2 / (2 * sigma^2))

    The bandwidth sigma is learnable, enabling the kernel to adapt to the
    complexity of the satellite image regions (high-structural-variance areas
    receive stronger gradient updates during denoising).
    """

    def __init__(self, init_sigma: float = 1.0):
        super().__init__()
        # Learnable bandwidth parameter (log-parameterised for positivity)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)))

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(
        self, x_t: torch.Tensor, x_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t   : latent at timestep t,   shape (B, C, H, W)
            x_prev: latent at timestep t-1, shape (B, C, H, W)
        Returns:
            kernel weights, shape (B, 1, H, W)  ∈ (0, 1]
        """
        diff_sq = (x_t - x_prev).pow(2).sum(dim=1, keepdim=True)  # (B,1,H,W)
        return torch.exp(-diff_sq / (2.0 * self.sigma ** 2))


# ---------------------------------------------------------------------------
# Token Pyramid Transformer (TPT)  (Section III-D & Appendix Section IV-C)
# Reference: Zhang et al., "TopFormer: Token Pyramid Transformer for Mobile
#            Semantic Segmentation", CVPR 2022.
# ---------------------------------------------------------------------------
class TPTBlock(nn.Module):
    """
    Single Token Pyramid Transformer block.

    Implements the hierarchical attention used in KAO's latent refinement step
    (Algorithm 1, Step 3 — "Token-wise Adaptive Conditioning via TPT").

    TPT is preferred over U-Net / HRNet because it:
      • Preserves token-level granularity without downsampling/upsampling
      • Is computationally efficient for VHR satellite images
      • Naturally aligns with KAO's kernel-adaptive latent updates
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                           batch_first=True)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : token sequence, shape (B, N, C)
        Returns:
            refined tokens, shape (B, N, C)
        """
        # Self-attention with pre-norm (standard ViT style)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TokenPyramidTransformer(nn.Module):
    """
    Multi-scale Token Pyramid Transformer for KAO.

    Implements the TPT component referenced in Algorithm 1 and
    Appendix Section IV-C ("KAO Theoretical Foundations and Integration
    with TPT").

    At each pyramid level the spatial tokens are pooled to a coarser
    resolution, processed by self-attention, and the refined context is
    broadcast back — enabling both local and global feature modelling.
    """

    def __init__(self, in_channels: int, token_dim: int = 256,
                 num_heads: int = 8, depth: int = 4,
                 pyramid_levels: int = 3):
        super().__init__()
        self.pyramid_levels = pyramid_levels

        # Project spatial features → token dim
        self.proj_in = nn.Conv2d(in_channels, token_dim, 1)

        # Independent transformer stacks per pyramid level
        self.levels = nn.ModuleList([
            nn.Sequential(*[TPTBlock(token_dim, num_heads) for _ in range(depth)])
            for _ in range(pyramid_levels)
        ])

        # Fuse multi-scale outputs back to in_channels
        self.proj_out = nn.Conv2d(token_dim, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : feature map, shape (B, C, H, W)
        Returns:
            refined feature map, shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        feat = self.proj_in(x)               # (B, token_dim, H, W)
        accumulated = torch.zeros_like(feat)

        for lvl, block in enumerate(self.levels):
            scale = 2 ** lvl                 # 1, 2, 4
            h_s, w_s = max(H // scale, 1), max(W // scale, 1)

            # Pool to pyramid level
            pooled = F.adaptive_avg_pool2d(feat, (h_s, w_s))

            # Flatten to token sequence (B, h_s*w_s, token_dim)
            tokens = rearrange(pooled, 'b c h w -> b (h w) c')
            tokens = block(tokens)
            tokens = rearrange(tokens, 'b (h w) c -> b c h w', h=h_s, w=w_s)

            # Upsample back and accumulate
            accumulated += F.interpolate(tokens, size=(H, W),
                                         mode='bilinear', align_corners=False)

        out = self.proj_out(accumulated / self.pyramid_levels)
        return x + out   # residual connection


# ---------------------------------------------------------------------------
# Latent Space Conditioning  (Appendix — "Latent Space Conditioning")
# h* = h_inf ⊙ (1 - D(m)) + h_cond ⊙ D(m)         [Eq. in Appendix]
# ---------------------------------------------------------------------------
class LatentSpaceConditioning(nn.Module):
    """
    Merges inferred and conditioned latent representations using a
    (possibly fractional / multi-resolution) mask.

    Implements the LSC equation from the Appendix:
        h* = h_inf ⊙ (1 - D(m)) + h_cond ⊙ D(m)

    where D(m) is the mask downsampled via average pooling to match the
    current latent resolution.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Lightweight refinement after merging
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        h_inf:  torch.Tensor,   # (B, C, H, W) — from denoising branch
        h_cond: torch.Tensor,   # (B, C, H, W) — from condition branch
        mask:   torch.Tensor,   # (B, 1, H_orig, W_orig) binary
    ) -> torch.Tensor:
        """
        Returns merged latent h*, shape (B, C, H, W).
        """
        # Downsample mask to latent resolution (average pooling → not binary)
        _, _, H, W = h_inf.shape
        d_mask = F.adaptive_avg_pool2d(mask.float(), (H, W))  # (B,1,H,W) ∈[0,1]

        h_star = h_inf * (1.0 - d_mask) + h_cond * d_mask
        return self.refine(h_star)


# ---------------------------------------------------------------------------
# Explicit Propagation (EP)  (Appendix — "Explicit Propagation")
# ĥ = γ⁻¹[φ(ω; γ(D(m), h_cond))]
# ---------------------------------------------------------------------------
class ExplicitPropagation(nn.Module):
    """
    Explicit Propagation module — propagates information from conditioned
    to inferred latent regions.

    Implements the EP equation from the Appendix:
        ĥ = γ⁻¹[ φ(ω ;  γ(D(m), h_cond)) ]

    where:
        γ     — mask-wise max-pooling operation
        φ(ω)  — learned non-linear transformation  φ : R^C → R^C
        γ⁻¹   — mask-wise unpooling (nearest-neighbour upsample)

    Trainable parameters are < 1% of the full diffusion model (paper claim).
    """

    def __init__(self, channels: int, pool_size: int = 4):
        super().__init__()
        self.pool_size = pool_size

        # φ(ω): lightweight channel-mixing MLP
        self.phi = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(
        self,
        h_cond: torch.Tensor,   # (B, C, H, W)
        mask:   torch.Tensor,   # (B, 1, H_orig, W_orig)
    ) -> torch.Tensor:
        """
        Returns propagated latent ĥ, same shape as h_cond.
        """
        B, C, H, W = h_cond.shape

        # γ: mask-aware max-pooling
        d_mask = F.adaptive_avg_pool2d(mask.float(), (H, W))
        pool_h = max(H // self.pool_size, 1)
        pool_w = max(W // self.pool_size, 1)

        pooled = F.adaptive_max_pool2d(h_cond * d_mask, (pool_h, pool_w))

        # φ(ω): non-linear feature transform in channel space
        tokens = rearrange(pooled, 'b c h w -> b (h w) c')
        tokens = self.phi(tokens)
        pooled = rearrange(tokens, 'b (h w) c -> b c h w',
                           h=pool_h, w=pool_w)

        # γ⁻¹: upsample back to original latent resolution
        h_hat = F.interpolate(pooled, size=(H, W), mode='nearest')
        return h_hat


# ---------------------------------------------------------------------------
# KAO Core Model  (Section III — Algorithm 1 in the paper)
# ---------------------------------------------------------------------------
class KAO(nn.Module):
    """
    KAO: Kernel-Adaptive Optimization model for satellite image inpainting.

    Implements the full Algorithm 1 described in Section III-D of the paper:

        Step 1  — Kernel-Adaptive Denoising (p-sample)
        Step 2  — Adaptive Noisy Condition Estimation (q-sample)
        Step 3  — Post-Conditioning via KAO in Latent Space (TPT + LSC + EP)
        Step 4  — Kernel-Blended Reconstruction

    Key design choices:
      • Gaussian RBF kernel weights gradient updates by image complexity
        (Section III-C, Eq. 8)
      • TPT replaces U-Net skip connections for multi-scale token refinement
        (Appendix Section IV-C)
      • LSC + EP together form the post-conditioning block that blends
        inference and condition branches (Appendix Eqs.)
    """

    def __init__(
        self,
        in_channels:       int   = 3,
        latent_channels:   int   = 256,
        tpt_depth:         int   = 4,
        tpt_heads:         int   = 8,
        tpt_pyramid_lvls:  int   = 3,
        num_ep_modules:    int   = 2,      # ablation: 1 or 2 (Table II)
        kernel_sigma:      float = 1.0,
        image_size:        int   = 256,
    ):
        super().__init__()

        self.latent_channels  = latent_channels
        self.num_ep_modules   = num_ep_modules

        # --- Encoder: image → latent space ---
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, 64,  3, padding=1), nn.SiLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1),         nn.SiLU(),
            nn.Conv2d(128, latent_channels, 3, stride=2, padding=1), nn.SiLU(),
        )

        # --- Token Pyramid Transformer (TPT) — Section III-D / Appendix ---
        self.tpt = TokenPyramidTransformer(
            in_channels  = latent_channels,
            token_dim    = latent_channels,
            num_heads    = tpt_heads,
            depth        = tpt_depth,
            pyramid_levels = tpt_pyramid_lvls,
        )

        # --- Latent Space Conditioning — Appendix ---
        self.lsc = LatentSpaceConditioning(latent_channels)

        # --- Explicit Propagation modules — Appendix ---
        # num_ep_modules=2 → "Full Model" in ablation Table II
        self.ep_modules = nn.ModuleList([
            ExplicitPropagation(latent_channels)
            for _ in range(num_ep_modules)
        ])

        # --- Gaussian RBF Kernel — Section III-C ---
        self.kernel = GaussianRBFKernel(init_sigma=kernel_sigma)

        # --- Timestep embedding (sinusoidal + MLP) ---
        self.t_emb_dim = latent_channels
        self.t_proj = nn.Sequential(
            nn.Linear(latent_channels, latent_channels * 4),
            nn.SiLU(),
            nn.Linear(latent_channels * 4, latent_channels),
        )

        # --- Decoder: latent → inpainted image ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64,          4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    # ------------------------------------------------------------------
    # Sinusoidal timestep embedding (standard DDPM convention)
    # ------------------------------------------------------------------
    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.t_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args  = t[:, None].float() * freqs[None]
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.t_proj(emb)              # (B, latent_channels)

    # ------------------------------------------------------------------
    # Algorithm 1 — single denoising step
    # ------------------------------------------------------------------
    def single_step(
        self,
        x_t:    torch.Tensor,   # (B, C, H, W) noisy image at step t
        x_cond: torch.Tensor,   # (B, C, H, W) conditioned (known) regions
        mask:   torch.Tensor,   # (B, 1, H, W) binary mask (1=known)
        t:      torch.Tensor,   # (B,) integer timestep indices
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs one forward step of KAO (Algorithm 1, Steps 1-4).

        Returns:
            x_pred   : predicted clean image x_0,  shape (B, C, H, W)
            k_weight : kernel weights,              shape (B, 1, H, W)
        """
        # ----------------------------------------------------------------
        # Step 1: Kernel-Adaptive Denoising (p-sample path)
        # ----------------------------------------------------------------
        # Concatenate noisy image + conditioned image + mask
        inp     = torch.cat([x_t, x_cond, mask], dim=1)   # (B, 2C+1, H, W)
        h_infr  = self.encoder(inp)                         # (B, L, h, w)

        # Inject timestep embedding (broadcast over spatial dims)
        t_emb   = self._timestep_embedding(t)               # (B, L)
        h_infr  = h_infr + t_emb[:, :, None, None]

        # TPT-based token refinement (Algorithm 1 — Step 3, TPT line)
        h_infr  = self.tpt(h_infr)

        # ----------------------------------------------------------------
        # Step 2: Adaptive Noisy Condition Estimation (q-sample path)
        # ----------------------------------------------------------------
        inp_c   = torch.cat([x_cond, x_cond, mask], dim=1)
        h_cond  = self.encoder(inp_c)
        h_cond  = self.tpt(h_cond)

        # ----------------------------------------------------------------
        # Step 3: Post-Conditioning via KAO in Latent Space
        # ----------------------------------------------------------------
        # 3a — Latent Space Conditioning
        h_star  = self.lsc(h_infr, h_cond, mask)

        # 3b — Explicit Propagation (applied num_ep_modules times)
        for ep in self.ep_modules:
            h_hat  = ep(h_star, mask)
            h_star = h_star + h_hat          # residual accumulation

        # ----------------------------------------------------------------
        # Step 4: Kernel-Blended Reconstruction
        # x_{t-1} = x_infr ⊙ (1 - m) + x_cond ⊙ m   [Eq. in Sec III-D]
        # ----------------------------------------------------------------
        x_pred  = self.decoder(h_star)                      # (B, C, H, W)
        x_pred  = x_pred * (1 - mask) + x_cond * mask      # blend

        # Kernel weight for loss modulation (Section III-C, Eq. 7)
        k_weight = self.kernel(x_t, x_pred)

        return x_pred, k_weight

    def forward(
        self,
        x_t:    torch.Tensor,
        x_cond: torch.Tensor,
        mask:   torch.Tensor,
        t:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.single_step(x_t, x_cond, mask, t)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[KAO] Running model sanity check on {device} ...")

    model = KAO(
        in_channels     = 3,
        latent_channels = 128,
        tpt_depth       = 2,
        tpt_heads       = 4,
        num_ep_modules  = 2,     # Full model (ablation Table II)
    ).to(device)

    B, C, H, W = 2, 3, 64, 64
    x_t    = torch.randn(B, C, H, W, device=device)
    x_cond = torch.randn(B, C, H, W, device=device)
    mask   = (torch.rand(B, 1, H, W, device=device) > 0.5).float()
    t      = torch.randint(0, 1000, (B,), device=device)

    x_pred, kw = model(x_t, x_cond, mask, t)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  x_pred  : {tuple(x_pred.shape)}")
    print(f"  k_weight: {tuple(kw.shape)}")
    print(f"  Total params: {total_params:.2f} M")
    print("[KAO] Sanity check passed ✓")
