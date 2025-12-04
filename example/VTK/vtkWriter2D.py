"""
VTK Writer for 2D data

This module provides functionality to export 2D simulation data in VTK format
for visualization and analysis in Paraview.
"""

import os
import numpy as np
import taichi as ti
import inspect
from typing import List, Optional
from .io2D import IO2D


class VtkWriter2D(IO2D):
    """
    Writer class for exporting 2D data to VTK format for Paraview
    
    This class handles the conversion of Taichi fields to VTK format,
    supporting multiple scalar and vector fields in a single file.
    """
    
    def __init__(self, nx: int, ny: int, outputDir: Optional[str] = None, prefix: str = "taichilb", 
                 subDir: str = "vtk", spacing: Optional[List[float]] = None):
        """
        Initialize VTK writer
        
        Args:
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            outputDir: Base output directory. If None, defaults to 'tmp' directory 
                      in the same directory as the calling script
            prefix: Prefix for output files
            subDir: Subdirectory for VTK files within outputDir
            spacing: Physical spacing [dx, dy] in meters (optional)
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
        self.spacing = spacing if spacing is not None else [1.0, 1.0]
        self.fields = {}  # store multiple fields before writing
        
        # Create output directory
        self.outputPath = self.createOutputDir(self.subDir)
        
    def addScalarField(self, data: ti.ScalarField, fieldName: str) -> None:
        """
        Add a scalar field to be written to VTK file
        
        Args:
            data: Taichi field containing scalar data (shape: nx, ny)
            fieldName: Name of the scalar field
        """
        # Convert to numpy and store
        dataArray = data.to_numpy()
        self.fields[fieldName] = ('scalar', dataArray)
        
    def addVectorField(self, data: ti.MatrixField, fieldName: str) -> None:
        """
        Add a vector field to be written to VTK file
        
        Args:
            data: Taichi field containing vector data (shape: nx, ny, 2)
            fieldName: Name of the vector field
        """
        # Convert to numpy and store
        dataArray = data.to_numpy()
        self.fields[fieldName] = ('vector', dataArray)
        
    def write(self, timeStep: int) -> None:
        """
        Write all added fields to a VTK file
        
        Args:
            timeStep: Current simulation time step
        """
        if not self.fields:
            raise RuntimeError("No fields added. Call addScalarField() or addVectorField() first.")
            
        if self.nx is None or self.ny is None:
            raise RuntimeError("Grid dimensions not set")
            
        # Create filename
        filename = f"{self.prefix}_{timeStep:06d}.vtk"
        filepath = os.path.join(self.outputPath, filename)
        
        # Write VTK file
        with open(filepath, 'w') as f:
            # VTK header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"TaichiLB output at timestep {timeStep}\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            
            # Grid dimensions and spacing
            f.write(f"DIMENSIONS {self.nx} {self.ny} 1\n")
            f.write(f"SPACING {self.spacing[0]} {self.spacing[1]} 1.0\n")
            f.write("ORIGIN 0.0 0.0 0.0\n")
            
            # Point data
            totalPoints = self.nx * self.ny
            f.write(f"POINT_DATA {totalPoints}\n")
            
            # Write each field
            for fieldName, (fieldType, data) in self.fields.items():
                if fieldType == 'scalar':
                    self._writeScalarData(f, fieldName, data)
                elif fieldType == 'vector':
                    self._writeVectorData(f, fieldName, data)
                    
    def _writeScalarData(self, file, fieldName: str, data: np.ndarray) -> None:
        """Write scalar field data to VTK file"""
        if self.nx is None or self.ny is None:
            raise RuntimeError("Grid dimensions not set")
            
        file.write(f"SCALARS {fieldName} float 1\n")
        file.write("LOOKUP_TABLE default\n")
        
        # Write data in VTK order (z varies fastest, then y, then x)
        for j in range(self.ny):
            for i in range(self.nx):
                file.write(f"{data[i, j]}\n")
                
    def _writeVectorData(self, file, fieldName: str, data: np.ndarray) -> None:
        """Write vector field data to VTK file"""
        if self.nx is None or self.ny is None:
            raise RuntimeError("Grid dimensions not set")
            
        file.write(f"VECTORS {fieldName} float\n")
        
        # Write data in VTK order (z varies fastest, then y, then x)
        for j in range(self.ny):
            for i in range(self.nx):
                # VTK expects 3D vectors, so add zero z-component
                file.write(f"{data[i, j, 0]} {data[i, j, 1]} 0.0\n")
                
    def clearFields(self) -> None:
        """Clear all stored fields"""
        self.fields.clear()