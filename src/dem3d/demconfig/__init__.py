"""
Core configuration and data types for DEM simulations.
"""

# Base classes
from .contact_model import ContactModelConfig, HertzContactConfig, LinearContactConfig
from .types import DomainBounds, ParticleProperties, WallProperties
from .demconfig import DEMSolverConfig

__all__ = [
    "ContactModelConfig",
    "HertzContactConfig",
    "LinearContactConfig",
    "DomainBounds",
    "ParticleProperties",
    "WallProperties",
    "DEMSolverConfig",
]