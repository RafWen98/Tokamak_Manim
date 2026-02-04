"""
Parametric geometry utilities for 3D tokamak visualizations.

This module provides:
- Fixed-reference frame computation for twist-free tube extrusion
- tube_along_path() for creating volumetric surfaces from parametric curves
- Cross-section profile functions (circular, rectangular, custom)
- Coil path geometries (D-shaped TF coils, PF coils, solenoid)
"""

from manim import Surface
import numpy as np
from numpy import pi, sin, cos
from typing import Callable, Literal, Tuple, Optional, Union

TAU = 2 * pi


# =============================================================================
# Fixed-Reference Frame Computation (no twist around tangent)
# =============================================================================

def compute_fixed_frame(
    path_func: Callable[[float], np.ndarray],
    t_values: np.ndarray,
    reference: Union[str, np.ndarray] = "radial"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a fixed-reference frame along a parametric curve.
    
    The cross-section orientation is locked relative to a reference direction,
    preventing any twist around the tangent axis. This is faster than RMF
    and gives predictable, consistent orientation for coils.
    
    Args:
        path_func: Function mapping t -> [x, y, z] point on curve
        t_values: Array of parameter values to sample
        reference: Reference direction for orienting the frame
            - "radial": Normal points away from Z-axis (good for TF coils)
            - "vertical": Normal points toward +Z (good for PF coils)  
            - "horizontal": Normal points in XY plane perpendicular to path
            - np.ndarray: Custom reference direction vector
    
    Returns:
        points: (N, 3) array of curve points
        tangents: (N, 3) array of unit tangent vectors
        normals: (N, 3) array of unit normal vectors
        binormals: (N, 3) array of unit binormal vectors
    """
    n = len(t_values)
    
    # Sample points
    points = np.array([path_func(t) for t in t_values])
    
    # Compute tangents from sampled points (robust at segment boundaries)
    # This avoids issues when path_func has discontinuous derivatives
    tangents = np.zeros((n, 3))
    
    for i in range(n):
        if i == 0:
            # Forward difference at start
            tangents[i] = points[1] - points[0]
        elif i == n - 1:
            # Backward difference at end
            tangents[i] = points[n - 1] - points[n - 2]
        else:
            # Central difference for interior points
            tangents[i] = points[i + 1] - points[i - 1]
    
    # Normalize tangents (with fallback for degenerate cases)
    for i in range(n):
        norm = np.linalg.norm(tangents[i])
        if norm > 1e-10:
            tangents[i] = tangents[i] / norm
        else:
            # Degenerate tangent - use neighbor's tangent
            if i > 0:
                tangents[i] = tangents[i - 1]
            elif i < n - 1:
                # Will be fixed in next iteration
                tangents[i] = np.array([1.0, 0.0, 0.0])
    
    # Compute normals and binormals using fixed reference
    normals = np.zeros((n, 3))
    binormals = np.zeros((n, 3))
    
    # First pass: compute initial normals/binormals
    for i in range(n):
        point = points[i]
        tangent = tangents[i]
        
        # Determine reference direction at this point
        if isinstance(reference, str):
            if reference == "radial":
                # Radial direction: points away from Z-axis
                ref = np.array([point[0], point[1], 0.0])
                ref_norm = np.linalg.norm(ref)
                if ref_norm > 1e-10:
                    ref = ref / ref_norm
                else:
                    ref = np.array([1.0, 0.0, 0.0])
            elif reference == "vertical":
                ref = np.array([0.0, 0.0, 1.0])
            elif reference == "horizontal":
                # Perpendicular to both tangent and Z
                ref = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
                ref_norm = np.linalg.norm(ref)
                if ref_norm > 1e-10:
                    ref = ref / ref_norm
                else:
                    ref = np.array([1.0, 0.0, 0.0])
            else:
                ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.asarray(reference, dtype=float)
            ref = ref / np.linalg.norm(ref)
        
        # Project reference onto plane perpendicular to tangent
        # normal = ref - (ref · tangent) * tangent
        normal = ref - np.dot(ref, tangent) * tangent
        normal_len = np.linalg.norm(normal)
        
        if normal_len < 1e-6:
            # Reference is parallel to tangent, use fallback
            if abs(tangent[0]) < 0.9:
                fallback = np.array([1.0, 0.0, 0.0])
            else:
                fallback = np.array([0.0, 1.0, 0.0])
            normal = np.cross(tangent, fallback)
            normal_len = np.linalg.norm(normal)
        
        normal = normal / normal_len
        binormal = np.cross(tangent, normal)
        
        normals[i] = normal
        binormals[i] = binormal
    
    # Second pass: fix sign flips by ensuring continuity
    # This prevents the cross-section from flipping when tangent passes through horizontal
    for i in range(1, n):
        # Check if normal flipped relative to previous
        if np.dot(normals[i], normals[i-1]) < 0:
            normals[i] = -normals[i]
            binormals[i] = -binormals[i]
    
    return points, tangents, normals, binormals


# =============================================================================
# Cross-Section Profile Functions
# =============================================================================

def circular_cross_section(radius: float) -> Callable[[float], Tuple[float, float]]:
    """
    Create a circular cross-section profile function.
    
    Args:
        radius: Radius of the circular cross-section
    
    Returns:
        Function mapping v in [0, TAU] to (normal_offset, binormal_offset)
    """
    def profile(v: float) -> Tuple[float, float]:
        return (radius * cos(v), radius * sin(v))
    return profile


def rectangular_cross_section(
    width: float,
    height: float
) -> Callable[[float], Tuple[float, float]]:
    """
    Create a rectangular cross-section profile function.
    
    Args:
        width: Width of rectangle (along normal direction)
        height: Height of rectangle (along binormal direction)
    
    Returns:
        Function mapping v in [0, TAU] to (normal_offset, binormal_offset)
    """
    half_w = width / 2
    half_h = height / 2
    
    # Perimeter of rectangle for uniform parameterization
    perimeter = 2 * (width + height)
    
    def profile(v: float) -> Tuple[float, float]:
        # Map v from [0, TAU] to perimeter distance
        dist = (v / TAU) * perimeter
        
        # Walk around rectangle: right edge -> top -> left -> bottom
        if dist < height:
            # Right edge (going up)
            return (half_w, -half_h + dist)
        elif dist < height + width:
            # Top edge (going left)
            return (half_w - (dist - height), half_h)
        elif dist < 2 * height + width:
            # Left edge (going down)
            return (-half_w, half_h - (dist - height - width))
        else:
            # Bottom edge (going right)
            return (-half_w + (dist - 2 * height - width), -half_h)
    
    return profile


# =============================================================================
# Main Tube Surface Creation
# =============================================================================

def tube_along_path(
    path_func: Callable[[float], np.ndarray],
    t_range: Tuple[float, float],
    cross_section: Union[
        Literal["circular", "rectangular"],
        Callable[[float], Tuple[float, float]]
    ] = "circular",
    radius: float = 0.1,
    width: float = 0.1,
    height: float = 0.1,
    reference: Union[str, np.ndarray] = "radial",
    closed: bool = True,
    path_resolution: int = 32,
    cross_resolution: int = 8,
    **surface_kwargs
) -> Surface:
    """
    Create a tube surface by sweeping a cross-section along a parametric path.
    
    Uses a fixed-reference frame for consistent, twist-free cross-section
    orientation along the entire path.
    
    Args:
        path_func: Function mapping t -> [x, y, z] point on curve
        t_range: (t_min, t_max) parameter range for the path
        cross_section: Profile type or custom function
            - "circular": Uses `radius` parameter
            - "rectangular": Uses `width` and `height` parameters  
            - callable: Custom function(v) -> (normal_offset, binormal_offset)
        radius: Radius for circular cross-section
        width: Width for rectangular cross-section (radial thickness)
        height: Height for rectangular cross-section (vertical thickness)
        reference: Reference direction for cross-section orientation
            - "radial": Normal points away from Z-axis (default, good for TF coils)
            - "vertical": Normal points toward +Z (good for PF coils)
            - np.ndarray: Custom reference direction
        closed: Whether the path is a closed loop
        path_resolution: Number of samples along the path (default 32)
        cross_resolution: Number of samples around the cross-section (default 8)
        **surface_kwargs: Additional arguments passed to Manim Surface
            (fill_color, fill_opacity, stroke_color, stroke_width, etc.)
    
    Returns:
        Manim Surface object representing the tube
    """
    t_min, t_max = t_range
    
    # Sample parameter values
    t_values = np.linspace(t_min, t_max, path_resolution, endpoint=not closed)
    
    # Compute fixed-reference frame (no twist!)
    points, tangents, normals, binormals = compute_fixed_frame(
        path_func, t_values, reference=reference
    )
    
    # Setup cross-section profile function
    if cross_section == "circular":
        profile_func = circular_cross_section(radius)
    elif cross_section == "rectangular":
        profile_func = rectangular_cross_section(width, height)
    elif callable(cross_section):
        profile_func = cross_section
    else:
        raise ValueError(f"Unknown cross_section type: {cross_section}")
    
    # Create interpolation function for frames
    def get_frame_at_u(u: float):
        """Get interpolated frame at parameter u in [0, 1]."""
        # Clamp u to valid range
        u = max(0.0, min(u, 0.9999))
        
        # Map u to index
        idx_float = u * (path_resolution - 1) if not closed else u * path_resolution
        idx = int(idx_float)
        frac = idx_float - idx
        
        # Handle boundary
        if closed:
            idx = idx % path_resolution
            idx_next = (idx + 1) % path_resolution
        else:
            idx = min(idx, path_resolution - 1)
            idx_next = min(idx + 1, path_resolution - 1)
        
        # Linear interpolation
        point = (1 - frac) * points[idx] + frac * points[idx_next]
        normal = (1 - frac) * normals[idx] + frac * normals[idx_next]
        binormal = (1 - frac) * binormals[idx] + frac * binormals[idx_next]
        
        # Re-normalize after interpolation
        normal_len = np.linalg.norm(normal)
        binormal_len = np.linalg.norm(binormal)
        
        if normal_len > 1e-10:
            normal = normal / normal_len
        else:
            normal = normals[idx]
            
        if binormal_len > 1e-10:
            binormal = binormal / binormal_len
        else:
            binormal = binormals[idx]
        
        return point, normal, binormal
    
    # Surface parametric function
    def surface_func(u: float, v: float) -> np.ndarray:
        """
        Parametric surface function.
        u in [0, 1]: position along path
        v in [0, TAU]: position around cross-section
        """
        point, normal, binormal = get_frame_at_u(u)
        offset_n, offset_b = profile_func(v)
        
        return point + offset_n * normal + offset_b * binormal
    
    # Default surface styling
    defaults = {
        "resolution": (path_resolution, cross_resolution),
        "fill_opacity": 1.0,
        "stroke_width": 0,  # No stroke for faster rendering
        "checkerboard_colors": False,
    }
    defaults.update(surface_kwargs)
    
    # Create and return surface
    # For closed paths, we need u to go all the way to 1.0 so the surface
    # wraps around and connects back to the start
    u_max = 1.0
    
    return Surface(
        surface_func,
        u_range=[0, u_max],
        v_range=[0, TAU],
        **defaults
    )


# =============================================================================
# Coil Path Geometries
# =============================================================================

def d_shaped_coil_path(
    t: float,
    R: float,
    height: float,
    angle: float
) -> np.ndarray:
    """
    Parametric D-shaped coil path for TF coils.
    
    The coil consists of 4 segments:
    1. Straight vertical inner edge (going up)
    2. Small radius arc at top (connecting to outer bow)
    3. Large radius arc on outside (the main "D" bow)
    4. Small radius arc at bottom (returning to start)
    
    Args:
        t: Parameter in [0, TAU]
        R: Major radius of the tokamak
        height: Vertical extent of the coil
        angle: Toroidal angle (rotation around Z-axis)
    
    Returns:
        [x, y, z] position on the coil path
    """
    # Geometry parameters
    R_inner = R * 0.7       # Distance from axis to inner straight edge
    l = height * 1.8        # Length of straight inner edge
    r_corner = height * 0.4 # Small corner radius
    R_arc = R * 1.2         # Large outer arc radius
    
    # Key points
    half_l = l / 2 - r_corner
    
    t_norm = t / TAU  # Normalize to 0-1
    
    # Segment proportions - based on actual arc lengths for smooth parameterization
    # The D-shape has 4 segments that naturally close:
    # 1. Inner straight line (length = 2*half_l)
    # 2. Top corner arc (length = pi/2 * r_corner)
    # 3. Outer big arc (length = pi * R_arc)
    # 4. Bottom corner arc (length = pi/2 * r_corner)
    
    len1 = 2 * half_l                    # Straight inner edge
    len2 = (pi / 2) * r_corner           # Top corner arc
    len3 = pi * R_arc                    # Big outer arc
    len4 = (pi / 2) * r_corner           # Bottom corner arc
    total_len = len1 + len2 + len3 + len4
    
    seg1 = len1 / total_len
    seg2 = len2 / total_len
    seg3 = len3 / total_len
    seg4 = len4 / total_len
    # Total = 1.0 by construction
    
    if t_norm < seg1:
        # Segment 1: Inner vertical straight line (going UP)
        frac = t_norm / seg1
        R_pos = R_inner
        Z_pos = -half_l + (2 * half_l) * frac
        
    elif t_norm < seg1 + seg2:
        # Segment 2: Top small corner arc (from inner edge to outer)
        frac = (t_norm - seg1) / seg2
        theta = pi - frac * (pi / 2)
        R_pos = R_inner + r_corner + r_corner * cos(theta)
        Z_pos = half_l + r_corner * sin(theta)
        
    elif t_norm < seg1 + seg2 + seg3:
        # Segment 3: Outer large arc (the big D bow, going down)
        frac = (t_norm - seg1 - seg2) / seg3
        theta = pi / 2 - frac * pi
        arc_center_R = R_inner + r_corner
        R_pos = arc_center_R + R_arc * cos(theta)
        Z_pos = (half_l + r_corner) * sin(theta)
        
    else:
        # Segment 4: Bottom small corner arc (returns to start)
        frac = (t_norm - seg1 - seg2 - seg3) / seg4
        frac = min(frac, 1.0)  # Clamp to avoid overshoot at t=TAU
        theta = -pi / 2 - frac * (pi / 2)
        R_pos = R_inner + r_corner + r_corner * cos(theta)
        Z_pos = -half_l + r_corner * sin(theta)
    
    return np.array([
        R_pos * cos(angle),
        R_pos * sin(angle),
        Z_pos
    ])


def pf_coil_path(
    t: float,
    radius: float,
    z_height: float
) -> np.ndarray:
    """
    Parametric circular path for PF (Poloidal Field) coils.
    
    Args:
        t: Parameter in [0, TAU]
        radius: Radius of the PF coil ring
        z_height: Vertical position of the coil
    
    Returns:
        [x, y, z] position on the coil path
    """
    return np.array([
        radius * cos(t),
        radius * sin(t),
        z_height
    ])


def solenoid_path(
    t: float,
    radius: float,
    height: float,
    n_turns: int
) -> np.ndarray:
    """
    Parametric helical path for central solenoid.
    
    Args:
        t: Parameter in [0, TAU * n_turns]
        radius: Radius of the solenoid
        height: Total height of the solenoid
        n_turns: Number of turns
    
    Returns:
        [x, y, z] position on the solenoid path
    """
    return np.array([
        radius * cos(t),
        radius * sin(t),
        height * (t / (TAU * n_turns) - 0.5)
    ])


# =============================================================================
# Torus and Plasma Surfaces
# =============================================================================

def torus_surface(
    R: float,
    r: float,
    **surface_kwargs
) -> Surface:
    """
    Create a torus surface (for plasma visualization).
    
    Args:
        R: Major radius (center of tube to center of torus)
        r: Minor radius (tube radius)
        **surface_kwargs: Additional arguments for Surface
    
    Returns:
        Manim Surface object
    """
    defaults = {
        "resolution": (32, 16),
        "fill_opacity": 0.2,
        "stroke_width": 0,
    }
    defaults.update(surface_kwargs)
    
    return Surface(
        lambda u, v: np.array([
            (R + r * cos(v)) * cos(u),
            (R + r * cos(v)) * sin(u),
            r * sin(v)
        ]),
        u_range=[0, TAU],
        v_range=[0, TAU],
        **defaults
    )


# =============================================================================
# Legacy function alias for backwards compatibility
# =============================================================================

def compute_rmf(points, closed=False):
    """Legacy RMF - redirects to fixed frame computation."""
    # This is kept for any code that still references it
    n = len(points)
    t_values = np.linspace(0, 1, n)
    def path_func(t):
        idx = int(t * (n - 1))
        idx = min(idx, n - 1)
        return points[idx]
    _, tangents, normals, binormals = compute_fixed_frame(path_func, t_values, "vertical")
    return tangents, normals, binormals
