"""SUTURE's NumPy-only interface for segmented execution boundaries."""
from .boundary import BoundaryResult, repair_boundary

__all__ = ['BoundaryResult', 'repair_boundary']
