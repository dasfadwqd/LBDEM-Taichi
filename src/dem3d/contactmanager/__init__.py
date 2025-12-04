"""
Contact model registry for DEM simulations.

Provides a unified interface to different contact force models, including:
- Linear spring-dashpot model
- Hertz-Mindlin nonlinear elastic model
"""

from .contactmodel import ContactModel
from .hertz import HertzMindlinContactModel
from .linear import LinearContactModel

__all__ = [
    "ContactModel",
    "HertzMindlinContactModel",
    "LinearContactModel"
]