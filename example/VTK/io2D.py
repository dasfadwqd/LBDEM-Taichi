"""
Base IO2D class for TaichiLB

This module defines the base class for 2D input/output operations.
Following the OpenLB pattern, this provides a common interface for different
types of data writers (images, VTK, etc.).
"""

from abc import ABC
from .ioBase import IOBase


class IO2D(IOBase):
    """
    Base class for 2D input/output operations
    
    This abstract base class defines the interface for 2D data I/O operations
    extending the common IOBase functionality with 2D-specific attributes.
    """
    
    def __init__(self, nx: int, ny: int, outputDir: str, prefix: str = "taichilb"):
        """
        Initialize 2D IO class
        
        Args:
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            outputDir: Base output directory for all data files
            prefix: Prefix for output files
        """
        super().__init__(outputDir, prefix)
        self.nx = nx
        self.ny = ny
        
    def getDimensions(self) -> tuple:
        """
        Get 2D grid dimensions
        
        Returns:
            Tuple of (nx, ny)
        """
        return (self.nx, self.ny)
        
    def getNumCells(self) -> int:
        """
        Get total number of cells in 2D grid
        
        Returns:
            Total number of cells (nx * ny)
        """
        return self.nx * self.ny
