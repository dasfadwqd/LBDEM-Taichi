"""
Image Writer for 2D data

This module provides functionality to save 2D scalar fields as images
using matplotlib for visualization and analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import inspect
from typing import Optional
from .io2D import IO2D


class ImageWriter2D(IO2D):
    """
    Writer class for saving 2D scalar fields as images
    
    This class handles the conversion of Taichi fields to images using matplotlib,
    with support for different colormaps and image formats. The class automatically
    calculates appropriate figure sizes with the larger dimension fixed at 8 inches
    to maintain proper aspect ratios. Font sizes are optimized for this standard size.
    """
    
    def __init__(self, nx: int, ny: int, outputDir: Optional[str] = None, prefix: str = "taichilb", 
                 subDir: str = "images", colormap: str = "viridis", 
                 imageFormat: str = "png", dpi: int = 300):
        """
        Initialize image writer
        
        Args:
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            outputDir: Base output directory. If None, defaults to 'tmp' directory 
                      in the same directory as the calling script
            prefix: Prefix for output files
            subDir: Subdirectory for images within outputDir
            colormap: Matplotlib colormap name
            imageFormat: Image format (png, jpg, etc.)
            dpi: Image resolution in dots per inch
        """
        # Auto-detect output directory if not provided
        if outputDir is None:
            # Get the directory of the script that called this constructor
            callerFrame = inspect.stack()[1]
            callerFile = callerFrame.filename
            callerDir = os.path.dirname(os.path.abspath(callerFile))
            outputDir = os.path.join(callerDir, "tmp")
        
        super().__init__(nx, ny, outputDir, prefix)
        self.subDir = subDir
        self.colormap = colormap
        self.imageFormat = imageFormat
        self.dpi = dpi
        
        # Calculate appropriate figure size based on grid dimensions
        self.figsize = self._calculateFigureSize()
        
        # Calculate appropriate font sizes based on figure dimensions
        self.fontSizes = self._calculateFontSizes()
        
        # Create output directory
        self.outputPath = self.createOutputDir(self.subDir)
        
    def _calculateFigureSize(self) -> tuple:
        """
        Calculate appropriate figure size based on grid dimensions
        
        The larger dimension is always set to 8 inches, and the smaller dimension
        is calculated to maintain the correct aspect ratio.
        
        Returns:
            Tuple of (width, height) in inches
        """
        if self.nx is None or self.ny is None:
            raise RuntimeError("Grid dimensions not set")
            
        # Calculate aspect ratio
        aspectRatio = self.nx / self.ny
        
        # Set larger dimension to 8 inches, calculate smaller dimension
        maxDimension = 8.0
        
        if aspectRatio >= 1.0:
            # Width >= Height: width = 8 inches
            width = maxDimension
            height = width / aspectRatio
        else:
            # Height > Width: height = 8 inches
            height = maxDimension
            width = height * aspectRatio
        
        return (width, height)
        
    def _calculateFontSizes(self) -> dict:
        """
        Calculate appropriate font sizes based on figure dimensions
        
        Since the larger dimension is always 8 inches, font sizes are consistent
        and optimized for this standard size.
        
        Returns:
            Dictionary containing font sizes for different text elements
        """
        # With the larger dimension fixed at 8 inches, use consistent font sizes
        # optimized for this size (baseline: 8 inches = standard readable fonts)
        
        fontSizes = {
            'title': 12,        # Main plot title
            'label': 10,        # Axis labels and colorbar labels
            'ticks': 9,         # Tick labels
            'colorbar': 9       # Colorbar tick labels
        }
        
        return fontSizes
        
    def write(self, data: ti.ScalarField, timeStep: int, fieldName: str = "field") -> None:
        """
        Write 2D scalar field as image
        
        Args:
            data: Taichi field containing scalar data (shape: nx, ny)
            timeStep: Current simulation time step
            fieldName: Name of the field being written
        """
        # Convert Taichi field to numpy array
        dataArray = data.to_numpy()
        
        # Create filename
        filename = f"{self.prefix}_{fieldName}_{timeStep:06d}.{self.imageFormat}"
        filepath = os.path.join(self.outputPath, filename)
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(dataArray.T, origin='lower', cmap=self.colormap, aspect='equal')
        cbar = plt.colorbar(im, ax=ax, shrink=1.0, aspect=20)
        cbar.set_label(fieldName, fontsize=self.fontSizes['label'])
        cbar.ax.tick_params(labelsize=self.fontSizes['colorbar'])
        ax.set_title(f"{fieldName} at timestep {timeStep}", fontsize=self.fontSizes['title'])
        ax.axis('off')
        
        # Save image
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def writeVelocity(self, velocityField: ti.MatrixField, timeStep: int) -> None:
        """
        Write velocity magnitude as image
        
        Args:
            velocityField: Taichi field containing velocity data (shape: nx, ny, 2)
            timeStep: Current simulation time step
        """
        # Convert velocity field to numpy and calculate magnitude
        velArray = velocityField.to_numpy()
        velMagnitude = np.sqrt(velArray[:, :, 0]**2 + velArray[:, :, 1]**2)
        
        # Create filename
        filename = f"{self.prefix}_velocity_{timeStep:06d}.{self.imageFormat}"
        filepath = os.path.join(self.outputPath, filename)
        
        # Create figure and plot
        plt.figure(figsize=self.figsize)
        plt.imshow(velMagnitude.T, origin='lower', cmap=self.colormap, aspect='equal')
        cbar = plt.colorbar(label="Velocity Magnitude")
        cbar.set_label("Velocity Magnitude", fontsize=self.fontSizes['label'])
        cbar.ax.tick_params(labelsize=self.fontSizes['colorbar'])
        plt.title(f"Velocity Magnitude at timestep {timeStep}", fontsize=self.fontSizes['title'])
        plt.axis('off')
        
        # Save image
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def writeStreamlines(self, velocityField: ti.MatrixField, timeStep: int, density: int = 2) -> None:
        """
        Write velocity field as streamlines
        
        Args:
            velocityField: Taichi field containing velocity data (shape: nx, ny, 2)
            timeStep: Current simulation time step
            density: Density of streamlines
        """
        # Convert velocity field to numpy
        velArray = velocityField.to_numpy()
        vx = velArray[:, :, 0]
        vy = velArray[:, :, 1]
        
        # Create coordinate grids
        if self.nx is None or self.ny is None:
            raise RuntimeError("Grid dimensions not set")
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Create filename
        filename = f"{self.prefix}_streamlines_{timeStep:06d}.{self.imageFormat}"
        filepath = os.path.join(self.outputPath, filename)
        
        # Create figure and plot streamlines
        plt.figure(figsize=self.figsize)
        speed = np.sqrt(vx**2 + vy**2)
        plt.streamplot(X.T, Y.T, vx.T, vy.T, color=speed.T, density=density, 
                      cmap=self.colormap, linewidth=1)
        cbar = plt.colorbar(label="Velocity Magnitude")
        cbar.set_label("Velocity Magnitude", fontsize=self.fontSizes['label'])
        cbar.ax.tick_params(labelsize=self.fontSizes['colorbar'])
        plt.title(f"Velocity Streamlines at timestep {timeStep}", fontsize=self.fontSizes['title'])
        
        # Set domain limits and equal aspect ratio
        plt.xlim(0, self.nx-1)
        plt.ylim(0, self.ny-1)
        plt.gca().set_aspect('equal')

        # Remove tick marks and labels
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Save image
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
