"""
Module responsible for broad-phase collision detection in 3D DEM simulations.

Provides spatial hashing with Morton codes (BPCD) and parallel prefix sum utilities
to efficiently generate collision candidates.
"""

# Base classes
from .bpcd import BPCD
from .prefixsum import PrefixSumExecutor
from .utils import *


__all__ = [
    "BPCD",
    "PrefixSumExecutor",

]