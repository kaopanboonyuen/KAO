# =============================================================================
# KAO: Kernel-Adaptive Optimization in Diffusion for Satellite Image
# Author: Teerapong Panboonyuen (aka Kao Panboonyuen)
# IEEE Transactions on Geoscience and Remote Sensing — IF: 8.9
# DOI: https://doi.org/10.1109/TGRS.2025.3621738
# =============================================================================

from .model     import KAO, GaussianRBFKernel, TokenPyramidTransformer
from .model     import LatentSpaceConditioning, ExplicitPropagation
from .diffusion import KAODiffusion
from .metrics   import KAOEvaluator, compute_psnr, compute_ssim

__version__  = "1.0.0"
__author__   = "Teerapong Panboonyuen"
__email__    = "teerapong.panboonyuen@gmail.com"
__paper__    = "KAO: Kernel-Adaptive Optimization in Diffusion for Satellite Image"
__journal__  = "IEEE Transactions on Geoscience and Remote Sensing"
__doi__      = "https://doi.org/10.1109/TGRS.2025.3621738"
__project__  = "https://kaopanboonyuen.github.io/KAO/"

__all__ = [
    "KAO",
    "KAODiffusion",
    "KAOEvaluator",
    "GaussianRBFKernel",
    "TokenPyramidTransformer",
    "LatentSpaceConditioning",
    "ExplicitPropagation",
    "compute_psnr",
    "compute_ssim",
]