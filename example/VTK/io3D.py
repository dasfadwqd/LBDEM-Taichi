"""
Base IO3D class for TaichiLB

This module defines the base class for 3D input/output operations.
Following the OpenLB pattern, this provides a common interface for different
types of data writers (VTK, etc.).
"""

from abc import ABC
from .ioBase import IOBase


class IO3D(IOBase):
    """
    Base class for 3D input/output operations
    
    This abstract base class defines the interface for 3D data I/O operations
    extending the common IOBase functionality with 3D-specific attributes.
    """
    
    def __init__(self, nx: int, ny: int, nz: int, outputDir: str, prefix: str = "taichilb"):
        """
        Initialize 3D IO class
        
        Args:
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            nz: Number of grid points in z direction
            outputDir: Base output directory for all data files
            prefix: Prefix for output files
        """
        super().__init__(outputDir, prefix)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
    def getDimensions(self) -> tuple:
        """
        Get 3D grid dimensions
        
        Returns:
            Tuple of (nx, ny, nz)
        """
        return (self.nx, self.ny, self.nz)
        
    def getNumCells(self) -> int:
        """
        Get total number of cells in 3D grid
        
        Returns:
            Total number of cells (nx * ny * nz)
        """
        return self.nx * self.ny * self.nz
