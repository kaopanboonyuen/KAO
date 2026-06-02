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
# This file implements the diffusion process described in:
#   Section III-A — Diffusion Models for Satellite Image Inpainting
#   Section III-B — Illustration of the Inpainting Process
#
#   Key equations implemented:
#     Forward process  : q(x_t | x_{t-1}) = N(sqrt(α_t)*x_{t-1}, (1-α_t)*I)
#     Reverse process  : p(x_{0:T}) = p(x_T) ∏ p_θ(x_{t-1}|x_t)
#     ELBO objective   : Section III-A (multi-line eq.)
#     KAO-modulated KL : Sec III-C, Eq. (7)
# =============================================================================
# MIT License — Copyright (c) 2026 Teerapong Panboonyuen
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Tuple


class KAODiffusion(nn.Module):
    """
    Diffusion process for KAO satellite image inpainting.

    Implements the forward (noising) and reverse (denoising) diffusion
    processes described in Section III-A of the paper.

    Forward process (Eq. 3):
        q(x_{1:T} | x_0) = ∏_{t=1}^{T} q(x_t | x_{t-1})

    Forward step (Eq. 6):
        q(x_t | x_{t-1}) = N(x_t ; sqrt(α_t)*x_{t-1}, (1-α_t)*I)

    Reverse process (Eq. 7):
        p(x_{0:T}) = p(x_T) ∏_{t=1}^{T} p_θ(x_{t-1}|x_t)

    KAO-modulated training objective (Eq. 9 / Section III-C):
        θ* = argmin_θ E[D_KL(q||p_θ) · K(X_t, X_{t-1})]
    """

    def __init__(
        self,
        num_timesteps:  int   = 1000,      # T in the paper (Sec III-A)
        beta_start:     float = 1e-4,
        beta_end:       float = 0.02,
        schedule:       str   = "cosine",  # 'linear' or 'cosine'
        device:         str   = "cpu",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.device        = device

        # --- Noise schedule (β_t) ---
        betas = self._make_schedule(num_timesteps, beta_start, beta_end, schedule)

        # Pre-compute α and ᾱ quantities (standard DDPM notation)
        alphas          = 1.0 - betas
        alphas_cumprod  = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not learnable, but moved with .to(device))
        self.register_buffer("betas",               betas)
        self.register_buffer("alphas",              alphas)
        self.register_buffer("alphas_cumprod",      alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",
                             torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod",
                             torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod",
                             torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod",
                             torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0) — used in KL divergence (Sec III-A)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev)
                             / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas)
                             / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------
    # Noise schedule builders
    # ------------------------------------------------------------------
    @staticmethod
    def _make_schedule(T, b_start, b_end, kind):
        if kind == "linear":
            return torch.linspace(b_start, b_end, T)
        elif kind == "cosine":
            # Nichol & Dhariwal 2021 cosine schedule
            steps  = torch.arange(T + 1, dtype=torch.float64)
            f      = torch.cos(((steps / T) + 0.008) / 1.008 * np.pi / 2) ** 2
            f      = f / f[0]
            betas  = torch.clip(1 - f[1:] / f[:-1], 0.0001, 0.9999)
            return betas.float()
        else:
            raise ValueError(f"Unknown schedule: {kind}")

    # ------------------------------------------------------------------
    # Forward process helpers
    # ------------------------------------------------------------------
    def _extract(self, a: torch.Tensor, t: torch.Tensor,
                 shape: Tuple) -> torch.Tensor:
        """Gather values at timestep t and reshape to broadcast."""
        out = a.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t:   torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward process: sample x_t from x_0 at timestep t.

        Implements Eq. (6):
            q(x_t | x_{t-1}) = N(x_t ; sqrt(ᾱ_t)*x_0, (1-ᾱ_t)*I)

        (Using the reparameterisation that marginalises over the chain.)

        Args:
            x_0   : clean satellite image, (B, C, H, W)
            t     : integer timesteps,      (B,)
            noise : optional pre-sampled Gaussian noise

        Returns:
            x_t : noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_acp  = self._extract(self.sqrt_alphas_cumprod,           t, x_0.shape)
        sqrt_1macp = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_acp * x_0 + sqrt_1macp * noise

    def q_posterior_mean_variance(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance: q(x_{t-1} | x_t, x_0).
        Used in the KL divergence objective (Section III-A).
        """
        c1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean  = c1 * x_0 + c2 * x_t
        var   = self._extract(self.posterior_variance,            t, x_t.shape)
        log_v = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_v

    # ------------------------------------------------------------------
    # KAO Training Loss  (Eq. 9 — kernel-modulated KL)
    # θ* = argmin E[ D_KL(q||p_θ) · K(X_t, X_{t-1}) ]
    # ------------------------------------------------------------------
    def kao_loss(
        self,
        model:        nn.Module,
        x_0:          torch.Tensor,
        mask:         torch.Tensor,
        kernel_fn:    Callable,
        noise:        Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the KAO training loss with kernel-modulated KL divergence.

        Implements Eq. (7) from Section III-C:
            θ* = argmin_θ E_{t~U(1,T)}[D_KL(q(X_{t-1}|X_t,X_0)‖p_θ(X_{t-1}|X_t))
                                        · K(X_t, X_{t-1})]

        Args:
            model     : KAO model (Section III-D)
            x_0       : clean satellite image batch, (B, C, H, W)
            mask      : binary inpainting mask,      (B, 1, H, W)
            kernel_fn : GaussianRBFKernel instance
            noise     : optional pre-sampled noise

        Returns:
            scalar loss
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps uniformly (Sec III-A Eq. 5)
        t = torch.randint(1, self.num_timesteps, (B,), device=device)

        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward process: get x_t
        x_t    = self.q_sample(x_0, t, noise=noise)
        x_cond = x_0 * mask          # known (conditioned) regions

        # Model prediction
        x_pred, k_weight = model(x_t, x_cond, mask, t)

        # Posterior targets (for KL)
        mu_q, var_q, _ = self.q_posterior_mean_variance(x_0, x_t, t)

        # Simple denoising objective (MSE proxy for KL divergence)
        # Weighted by the kernel K(X_t, X_{t-1}) — Eq. (7)
        recon_loss = F.mse_loss(x_pred, x_0, reduction='none')       # (B,C,H,W)
        recon_loss = (recon_loss * k_weight).mean()

        # Additional perceptual loss on masked region only
        mask_loss  = F.l1_loss(
            x_pred * (1 - mask),
            x_0    * (1 - mask),
        )

        return recon_loss + 0.1 * mask_loss

    # ------------------------------------------------------------------
    # Reverse process  (Algorithm 1 — full denoising loop)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample_loop(
        self,
        model:       nn.Module,
        x_cond:      torch.Tensor,   # (B, C, H, W) — known image regions
        mask:        torch.Tensor,   # (B, 1, H, W) — 1 = known
        shape:       Tuple,
        device:      str = "cpu",
        verbose:     bool = True,
    ) -> torch.Tensor:
        """
        Full reverse diffusion loop: Algorithm 1 (Section III-D).

        Iterates t = T → 1, running one KAO denoising step each time:
            x_{t-1} = x_infr ⊙ (1-m) + x_cond ⊙ m

        Args:
            model   : trained KAO model
            x_cond  : conditioned satellite image
            mask    : inpainting mask
            shape   : output shape (B, C, H, W)

        Returns:
            x_0 : final inpainted image
        """
        B = shape[0]
        # Start from pure Gaussian noise: p(x_T) = N(0, I)
        x_t = torch.randn(shape, device=device)

        iterator = range(self.num_timesteps - 1, 0, -1)
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="KAO Denoising", unit="step")

        for t_idx in iterator:
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # Steps 1-4 from Algorithm 1
            x_pred, _  = model(x_t, x_cond, mask, t_batch)

            # Posterior mean for x_{t-1} (Section III-A reverse process)
            mu_q, _, log_var = self.q_posterior_mean_variance(x_pred, x_t, t_batch)

            # Add noise for t > 1 (no noise at final step)
            noise = torch.randn_like(x_t) if t_idx > 1 else torch.zeros_like(x_t)
            x_t   = mu_q + (0.5 * log_var).exp() * noise

            # Enforce known regions from condition at every step (RePaint-style)
            noise_cond = torch.randn_like(x_cond) if t_idx > 1 else torch.zeros_like(x_cond)
            alpha_cp   = self._extract(self.alphas_cumprod, t_batch, x_cond.shape)
            x_cond_t   = torch.sqrt(alpha_cp) * x_cond + torch.sqrt(1 - alpha_cp) * noise_cond
            x_t        = x_t * (1 - mask) + x_cond_t * mask

        return x_t.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from kao.model import KAO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[KAODiffusion] Sanity check on {device} ...")

    diffusion = KAODiffusion(num_timesteps=100, schedule="cosine").to(device)
    model     = KAO(in_channels=3, latent_channels=64, tpt_depth=1,
                    tpt_heads=4, num_ep_modules=2).to(device)

    B, C, H, W = 2, 3, 32, 32
    x_0  = torch.randn(B, C, H, W, device=device)
    mask = (torch.rand(B, 1, H, W, device=device) > 0.5).float()

    loss = diffusion.kao_loss(model, x_0, mask, model.kernel)
    print(f"  KAO loss: {loss.item():.4f}")

    x_out = diffusion.p_sample_loop(model, x_0 * mask, mask,
                                     shape=(B, C, H, W), device=device,
                                     verbose=False)
    print(f"  Output shape : {tuple(x_out.shape)}")
    print("[KAODiffusion] Sanity check passed ✓")
