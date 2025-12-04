"""
Base IO class for TaichiLB

This module defines the base class for input/output operations shared between
2D and 3D implementations. Following object-oriented principles, this eliminates
code duplication and provides common functionality.
"""

import os
from abc import ABC
from typing import Optional


class IOBase(ABC):
    """
    Base class for input/output operations
    
    This abstract base class defines common functionality for both 2D and 3D
    data I/O operations including setup of output directories and file naming conventions.
    """
    
    def __init__(self, outputDir: str, prefix: str = "taichilb"):
        """
        Initialize IO base class
        
        Args:
            outputDir: Base output directory for all data files
            prefix: Prefix for output files
        """
        self.outputDir = outputDir
        self.prefix = prefix
        
    def createOutputDir(self, subDir: str = "") -> str:
        """
        Create output directory structure
        
        Args:
            subDir: Subdirectory within the main output directory
            
        Returns:
            Full path to the created directory
        """
        if subDir:
            fullPath = os.path.join(self.outputDir, subDir)
        else:
            fullPath = self.outputDir
            
        os.makedirs(fullPath, exist_ok=True)
        return fullPath
        
    def getFileName(self, name: str, extension: str, timeStep: Optional[int] = None) -> str:
        """
        Generate standardized filename with optional timestep
        
        Args:
            name: Base name for the file
            extension: File extension (without dot)
            timeStep: Optional timestep number for time series data
            
        Returns:
            Formatted filename
        """
        if timeStep is not None:
            return f"{self.prefix}_{name}_{timeStep:06d}.{extension}"
        else:
            return f"{self.prefix}_{name}.{extension}"
            
    def getFullPath(self, fileName: str, subDir: str = "") -> str:
        """
        Get full path for a file in the output directory
        
        Args:
            fileName: Name of the file
            subDir: Optional subdirectory
            
        Returns:
            Full path to the file
        """
        if subDir:
            return os.path.join(self.outputDir, subDir, fileName)
        else:
            return os.path.join(self.outputDir, fileName)
            
    def ensureDirectoryExists(self, filePath: str) -> None:
        """
        Ensure that the directory for a given file path exists
        
        Args:
            filePath: Full path to a file
        """
        directory = os.path.dirname(filePath)
        if directory:
            os.makedirs(directory, exist_ok=True)
