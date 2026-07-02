"""Core contributions of PRICoD.

- ``PixelRegionInstanceContrast`` (PRI): multi-granularity cross-network contrast.
- ``ClassConditionedRefinement`` (CRM): class-conditioned logit refinement.
- ``urc_loss`` (URC): uncertainty-rectified cross-network consistency.
"""

from .pri import PixelRegionInstanceContrast, PRI
from .crm import ClassConditionedRefinement, CRM
from .urc import urc_loss, uncertainty_loss

__all__ = [
    "PixelRegionInstanceContrast",
    "PRI",
    "ClassConditionedRefinement",
    "CRM",
    "urc_loss",
    "uncertainty_loss",
]
