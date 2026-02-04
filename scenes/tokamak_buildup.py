"""
Tokamak Build-Up Scene.

A simple introductory scene showing the tokamak being constructed
coil-by-coil with volumetric 3D coils.
"""

from manim import *
import numpy as np
from numpy import pi, sin, cos

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    tube_along_path,
    d_shaped_coil_path,
    torus_surface,
)

TAU = 2 * pi

# VSCode-style dark anthracite background
ANTHRACITE = "#1e1e1e"


class TokamakBuildup(ThreeDScene):
    """
    Scene showing the tokamak being built up coil-by-coil.
    
    Build order:
    1. TF coils (Toroidal Field) - appear one by one
    2. PF coils (Poloidal Field) - upper and lower pairs
    3. Central Solenoid - helical winding
    4. Plasma torus - fades in at the end
    """
    
    def construct(self):
        # Set dark background
        self.camera.background_color = ANTHRACITE
        
        # Tokamak parameters
        R = 3.0   # Major radius
        r = 1.0   # Minor radius
        n_tf_coils = 12  # Number of TF coils
        
        # Camera setup - nice 3D viewing angle
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
        self.camera.frame_center = ORIGIN
        
        # Title
        title = Text("Building a Tokamak", font_size=42, color=WHITE)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait(0.5)
        
        # === 1. TF Coils ===
        tf_label = Text("Toroidal Field Coils", font_size=28, color=BLUE_B)
        tf_label.next_to(title, DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(tf_label)
        self.play(Write(tf_label))
        
        tf_coils = self._create_tf_coils(R, r, n_coils=n_tf_coils)
        
        # Animate TF coils appearing one by one with slight overlap
        self.play(
            LaggedStart(
                *[FadeIn(coil, scale=0.8) for coil in tf_coils],
                lag_ratio=0.3
            ),
            run_time=5
        )
        self.wait(0.5)
        
        # === 2. PF Coils ===
        self.play(FadeOut(tf_label))
        pf_label = Text("Poloidal Field Coils", font_size=28, color=GREEN_B)
        pf_label.next_to(title, DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(pf_label)
        self.play(Write(pf_label))
        
        pf_coils = self._create_pf_coils(R, r)
        
        self.play(
            LaggedStart(
                *[FadeIn(coil, scale=0.9) for coil in pf_coils],
                lag_ratio=0.4
            ),
            run_time=2
        )
        self.wait(0.5)
        
        # === 3. Central Solenoid ===
        self.play(FadeOut(pf_label))
        cs_label = Text("Central Solenoid", font_size=28, color=RED_B)
        cs_label.next_to(title, DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(cs_label)
        self.play(Write(cs_label))
        
        solenoid = self._create_central_solenoid(height=2.5 * r)
        
        self.play(Create(solenoid), run_time=2)
        self.wait(0.5)
        
        # === 4. Plasma ===
        self.play(FadeOut(cs_label))
        plasma_label = Text("Hot Plasma (150 Million °C)", font_size=28, color=YELLOW)
        plasma_label.next_to(title, DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(plasma_label)
        self.play(Write(plasma_label))
        
        plasma = self._create_plasma(R, r * 0.75)
        
        self.play(FadeIn(plasma, scale=0.7), run_time=2)
        self.wait(0.5)
        
        # === 5. Rotate to showcase ===
        self.play(FadeOut(plasma_label))
        
        complete_label = Text("Complete Tokamak", font_size=28, color=WHITE)
        complete_label.next_to(title, DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(complete_label)
        self.play(Write(complete_label))
        
        # Slow rotation to show the full structure
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(8)
        self.stop_ambient_camera_rotation()
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(complete_label),
            FadeOut(VGroup(*tf_coils, *pf_coils, solenoid, plasma)),
            run_time=2
        )
        self.wait(0.5)
    
    def _create_tf_coils(
        self,
        R: float,
        r: float,
        n_coils: int = 12
    ) -> list:
        """
        Create volumetric D-shaped TF coils using tube extrusion.
        
        Args:
            R: Major radius
            r: Minor radius  
            n_coils: Number of TF coils around the torus
        
        Returns:
            List of Surface objects (one per coil)
        """
        coils = []
        height = r * 2.0
        
        # Cross-section dimensions for TF coil conductors
        coil_width = 0.15   # Radial thickness
        coil_height = 0.25  # Vertical thickness
        
        for i in range(n_coils):
            angle = i * TAU / n_coils
            
            # Create path function for this specific coil angle
            def make_path(a):
                return lambda t: d_shaped_coil_path(t, R, height, a)
            
            path_func = make_path(angle)
            
            # Create volumetric coil using tube extrusion
            coil = tube_along_path(
                path_func=path_func,
                t_range=(0, TAU),
                cross_section="rectangular",
                width=coil_width,
                height=coil_height,
                reference="radial",  # Keep cross-section oriented radially (no twist)
                closed=True,
                path_resolution=32,
                cross_resolution=8,
                fill_color=BLUE_D,
                fill_opacity=1.0,
            )
            coils.append(coil)
        
        return coils
    
    def _create_pf_coils(self, R: float, r: float) -> list:
        """
        Create volumetric PF coils (horizontal rings above/below midplane).
        
        Args:
            R: Major radius
            r: Minor radius
        
        Returns:
            List of Surface objects
        """
        coils = []
        
        # PF coil specifications: (radius, z_height)
        pf_specs = [
            (R * 1.4, r * 1.6),   # Upper outer
            (R * 1.4, -r * 1.6),  # Lower outer
            (R * 0.9, r * 1.8),   # Upper inner
            (R * 0.9, -r * 1.8),  # Lower inner
        ]
        
        # Cross-section for PF coils (slightly different from TF)
        coil_width = 0.12
        coil_height = 0.18
        
        for radius, z_height in pf_specs:
            # Circular path for PF coil
            def make_pf_path(rad, z):
                return lambda t: np.array([
                    rad * cos(t),
                    rad * sin(t),
                    z
                ])
            
            path_func = make_pf_path(radius, z_height)
            
            coil = tube_along_path(
                path_func=path_func,
                t_range=(0, TAU),
                cross_section="rectangular",
                width=coil_width,
                height=coil_height,
                reference="vertical",  # Keep cross-section oriented vertically
                closed=True,
                path_resolution=24,
                cross_resolution=8,
                fill_color=GREEN_D,
                fill_opacity=1.0,
            )
            coils.append(coil)
        
        return coils
    
    def _create_central_solenoid(
        self,
        height: float,
        radius: float = 0.4,
        n_turns: int = 20
    ) -> Surface:
        """
        Create volumetric central solenoid (helical winding).
        
        Args:
            height: Total height of the solenoid
            radius: Radius of the solenoid cylinder
            n_turns: Number of helical turns
        
        Returns:
            Surface object
        """
        # Helical path for solenoid
        def solenoid_path(t):
            return np.array([
                radius * cos(t),
                radius * sin(t),
                height * (t / (TAU * n_turns) - 0.5)
            ])
        
        # Thinner wire for solenoid
        wire_radius = 0.04
        
        solenoid = tube_along_path(
            path_func=solenoid_path,
            t_range=(0, TAU * n_turns),
            cross_section="circular",
            radius=wire_radius,
            reference="radial",  # Keep cross-section oriented radially
            closed=False,
            path_resolution=n_turns * 12,  # Reduced samples per turn
            cross_resolution=6,
            fill_color=RED_D,
            fill_opacity=1.0,
        )
        
        return solenoid
    
    def _create_plasma(self, R: float, r: float) -> Surface:
        """
        Create semi-transparent plasma torus.
        
        Args:
            R: Major radius
            r: Minor radius (slightly smaller than coil minor radius)
        
        Returns:
            Surface object
        """
        return torus_surface(
            R=R,
            r=r,
            fill_color=YELLOW,
            fill_opacity=0.25,
            stroke_color=ORANGE,
            stroke_width=0.5,
            resolution=(32, 16),
        )


class TokamakShowcase(ThreeDScene):
    """
    Simple showcase scene - just displays the complete tokamak rotating.
    Good for testing the volumetric coils render correctly.
    """
    
    def construct(self):
        self.camera.background_color = ANTHRACITE
        
        R = 3.0
        r = 1.0
        
        self.set_camera_orientation(phi=65 * DEGREES, theta=45 * DEGREES)
        
        # Build all components
        tf_coils = self._create_tf_coils(R, r, n_coils=12)
        pf_coils = self._create_pf_coils(R, r)
        solenoid = self._create_central_solenoid(height=2.5 * r)
        plasma = self._create_plasma(R, r * 0.75)
        
        # Add everything
        for coil in tf_coils:
            self.add(coil)
        for coil in pf_coils:
            self.add(coil)
        self.add(solenoid)
        self.add(plasma)
        
        # Rotate
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(10)
        self.stop_ambient_camera_rotation()
    
    # Reuse the same creation methods
    _create_tf_coils = TokamakBuildup._create_tf_coils
    _create_pf_coils = TokamakBuildup._create_pf_coils
    _create_central_solenoid = TokamakBuildup._create_central_solenoid
    _create_plasma = TokamakBuildup._create_plasma
