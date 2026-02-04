"""
Utils package for Tokamak Manim visualizations.

Provides parametric geometry functions for creating volumetric 3D coils,
magnetic field lines, and other tokamak components.
"""

from utils.parametric import (
    # Fixed-reference frame computation
    compute_fixed_frame,
    
    # Main tube/volume creation
    tube_along_path,
    
    # Cross-section profiles
    circular_cross_section,
    rectangular_cross_section,
    
    # Coil path geometries
    d_shaped_coil_path,
    pf_coil_path,
    solenoid_path,
    
    # Torus and plasma surfaces
    torus_surface,
    
    # Legacy (kept for compatibility)
    compute_rmf,
)

__all__ = [
    "compute_fixed_frame",
    "compute_rmf",
    "tube_along_path",
    "circular_cross_section",
    "rectangular_cross_section",
    "d_shaped_coil_path",
    "pf_coil_path",
    "solenoid_path",
    "torus_surface",
]
