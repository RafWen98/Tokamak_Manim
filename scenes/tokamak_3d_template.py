"""
3D Tokamak Visualization Scenes.

This module contains Manim scenes for visualizing:
- Tokamak coil structure (TF, PF, Central Solenoid)
- Plasma torus geometry
- Magnetic field lines (toroidal, poloidal, helical)
- Magnetic bottle confinement concept
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


class TokamakConstruction(ThreeDScene):
    """
    Scene showing the construction of a tokamak with all coil systems.
    
    Builds up:
    1. Empty torus (vacuum vessel outline)
    2. Toroidal Field (TF) coils - D-shaped, create toroidal field
    3. Poloidal Field (PF) coils - horizontal rings for shaping
    4. Central Solenoid - drives plasma current
    5. Semi-transparent plasma
    """
    
    def construct(self):
        # Tokamak parameters
        R = 3.0   # Major radius
        r = 1.0   # Minor radius
        
        # Camera setup - start with nice 3D view
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
        self.camera.frame_center = ORIGIN
        
        # Title
        title = Text("Tokamak: Magnetic Confinement", font_size=36)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # === 1. Start building the structure ===
        self.wait(0.5)
        
        # === 2. Add TF Coils ===
        tf_label = Text("Toroidal Field Coils", font_size=24, color=BLUE)
        tf_label.next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(tf_label)
        
        tf_coils = self._create_tf_coils(R, r, n_coils=12)
        
        self.play(Write(tf_label))
        self.play(
            LaggedStart(*[Create(coil) for coil in tf_coils], lag_ratio=0.1),
            run_time=3
        )
        self.wait(0.5)
        
        # === 3. Add PF Coils ===
        self.play(FadeOut(tf_label))
        pf_label = Text("Poloidal Field Coils", font_size=24, color=GREEN)
        pf_label.next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(pf_label)
        
        pf_coils = self._create_pf_coils(R, r)
        
        self.play(Write(pf_label))
        self.play(
            LaggedStart(*[Create(coil) for coil in pf_coils], lag_ratio=0.2),
            run_time=2
        )
        self.wait(0.5)
        
        # === 4. Add Central Solenoid ===
        self.play(FadeOut(pf_label))
        cs_label = Text("Central Solenoid", font_size=24, color=RED)
        cs_label.next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(cs_label)
        
        solenoid = self._create_central_solenoid(height=2.5 * r)
        
        self.play(Write(cs_label))
        self.play(Create(solenoid), run_time=2)
        self.wait(0.5)
        
        # === 5. Add Plasma ===
        self.play(FadeOut(cs_label))
        plasma_label = Text("Hot Plasma", font_size=24, color=YELLOW)
        plasma_label.next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(plasma_label)
        
        plasma = self._create_plasma_torus(R, r * 0.8)
        
        self.play(Write(plasma_label))
        self.play(FadeIn(plasma, scale=0.8), run_time=2)
        
        # === 6. Rotate to show structure ===
        self.play(FadeOut(plasma_label))
        
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(6)
        self.stop_ambient_camera_rotation()
        
        self.play(FadeOut(title))
        self.wait(1)
    
    def _create_torus_wireframe(self, R: float, r: float, 
                                 color=WHITE, opacity=0.5) -> VGroup:
        """Create wireframe representation of torus."""
        wireframe = VGroup()
        
        # Toroidal circles (around the long way)
        n_toroidal = 24
        for i in range(n_toroidal):
            phi = i * TAU / n_toroidal
            circle = ParametricFunction(
                lambda t, phi=phi: np.array([
                    (R + r * cos(t)) * cos(phi),
                    (R + r * cos(t)) * sin(phi),
                    r * sin(t)
                ]),
                t_range=[0, TAU, 0.1],
                color=color,
            ).set_stroke(width=1, opacity=opacity)
            wireframe.add(circle)
        
        # Poloidal circles (around the short way)
        n_poloidal = 8
        for i in range(n_poloidal):
            theta = i * TAU / n_poloidal
            circle = ParametricFunction(
                lambda t, theta=theta: np.array([
                    (R + r * cos(theta)) * cos(t),
                    (R + r * cos(theta)) * sin(t),
                    r * sin(theta)
                ]),
                t_range=[0, TAU, 0.1],
                color=color,
            ).set_stroke(width=1, opacity=opacity)
            wireframe.add(circle)
        
        return wireframe
    
    def _create_tf_coils(self, R: float, r: float, n_coils: int = 16) -> VGroup:
        """Create D-shaped TF coils."""
        coils = VGroup()
        height = r * 1.8
        
        for i in range(n_coils):
            angle = i * TAU / n_coils
            
            # D-shaped coil using custom parametric function
            coil = ParametricFunction(
                lambda t, a=angle: self._d_shaped_coil(t, R, height, a),
                t_range=[0, TAU, 0.02],
                color=BLUE,
            ).set_stroke(width=4)
            coils.add(coil)
        
        return coils
    
    def _d_shaped_coil(self, t: float, R: float, height: float, 
                       angle: float) -> np.ndarray:
        """
        Parametric D-shaped coil with 4 segments:
        1. Straight vertical inner edge (length l)
        2. Small radius arc at top (radius r_corner)
        3. Large radius arc on outside (radius R_arc)
        4. Small radius arc at bottom (radius r_corner)
        """
        # Geometry parameters
        R_inner = R * 0.5      # Distance from axis to inner straight edge
        l = height * 1.8          # Length of straight inner edge
        r_corner = height * 0.4  # Small corner radius
        R_arc = R * 1.0         # Large outer arc radius
        
        # Key points
        half_l = l / 2 - r_corner  # Half of straight section (excluding corners)
        
        t_norm = t / TAU  # Normalize to 0-1
        
        # Segment proportions (approximate arc lengths)
        seg1 = 0.20  # Inner straight line
        seg2 = 0.10  # Top small arc
        seg3 = 0.40  # Outer large arc  
        seg4 = 0.10  # Bottom small arc
        # seg1 again to close (0.20)
        
        if t_norm < seg1:
            # Segment 1: Inner vertical straight line (going UP)
            frac = t_norm / seg1
            R_pos = R_inner
            Z_pos = -half_l + (2 * half_l) * frac
            
        elif t_norm < seg1 + seg2:
            # Segment 2: Top small corner arc (connects inner line to outer arc)
            frac = (t_norm - seg1) / seg2
            theta = pi - frac * (pi / 2)  # From 180° to 90°
            # Center of small arc is at (R_inner + r_corner, half_l)
            R_pos = R_inner + r_corner + r_corner * cos(theta)
            Z_pos = half_l + r_corner * sin(theta)
            
        elif t_norm < seg1 + seg2 + seg3:
            # Segment 3: Outer large arc (the big bow)
            frac = (t_norm - seg1 - seg2) / seg3
            theta = pi/2 - frac * pi  # From 90° to -90°
            # Center of large arc - positioned so it connects smoothly
            arc_center_R = R_inner + r_corner
            R_pos = arc_center_R + R_arc * cos(theta)
            Z_pos = R_arc * sin(theta) * (half_l + r_corner) / R_arc
            # Scale Z to match the height
            Z_pos = (half_l + r_corner) * sin(theta)
            
        elif t_norm < seg1 + seg2 + seg3 + seg4:
            # Segment 4: Bottom small corner arc (connects outer arc to inner line)
            frac = (t_norm - seg1 - seg2 - seg3) / seg4
            theta = -pi/2 - frac * (pi / 2)  # From -90° to -180°
            # Center of small arc is at (R_inner + r_corner, -half_l)
            R_pos = R_inner + r_corner + r_corner * cos(theta)
            Z_pos = -half_l + r_corner * sin(theta)
            
        else:
            # Back to start (shouldn't normally reach here due to periodicity)
            R_pos = R_inner
            Z_pos = -half_l
        
        return np.array([
            R_pos * cos(angle),
            R_pos * sin(angle),
            Z_pos
        ])
    
    def _create_pf_coils(self, R: float, r: float) -> VGroup:
        """Create horizontal PF coils above and below midplane."""
        coils = VGroup()
        
        # Coil positions (z heights and radii)
        coil_specs = [
            (R * 1.4, r * 1.6),   # Upper outer
            (R * 1.4, -r * 1.6),  # Lower outer
        ]
        
        for radius, z in coil_specs:
            coil = ParametricFunction(
                lambda t, rad=radius, z_h=z: np.array([
                    rad * cos(t),
                    rad * sin(t),
                    z_h
                ]),
                t_range=[0, TAU, 0.05],
                color=GREEN,
            ).set_stroke(width=5)
            coils.add(coil)
        
        return coils
    
    def _create_central_solenoid(self, height: float, 
                                  radius: float = 0.4,
                                  n_turns: int = 25) -> VMobject:
        """Create helical central solenoid."""
        solenoid = ParametricFunction(
            lambda t: np.array([
                radius * cos(t),
                radius * sin(t),
                height * (t / (TAU * n_turns) - 0.5)
            ]),
            t_range=[0, TAU * n_turns, 0.05],
            color=RED,
        ).set_stroke(width=2)
        
        return solenoid
    
    def _create_plasma_torus(self, R: float, r: float) -> Surface:
        """Create semi-transparent plasma torus."""
        plasma = Surface(
            lambda u, v: np.array([
                (R + r * cos(v)) * cos(u),
                (R + r * cos(v)) * sin(u),
                r * sin(v)
            ]),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(32, 16),
            fill_color=YELLOW,
            fill_opacity=0.2,
            stroke_color=GRAY,
            stroke_width=0.5,
        )
        return plasma


class MagneticFieldLines(ThreeDScene):
    """
    Scene showing how magnetic field lines are created and combine
    to form the helical structure that confines plasma.
    """
    
    def construct(self):
        R = 3.0
        r = 1.0
        
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
        
        # Create basic structure (simplified)
        plasma = self._create_plasma_torus(R, r * 0.8)
        self.play(FadeIn(plasma))
        
        # === 1. Show toroidal field ===
        title = Text("Toroidal Magnetic Field (from TF coils)", font_size=28)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        toroidal_arrows = self._create_toroidal_field_arrows(R, r)
        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in toroidal_arrows], lag_ratio=0.05),
            run_time=2
        )
        self.wait(2)
        
        # === 2. Show poloidal field ===
        self.play(FadeOut(title))
        title2 = Text("Poloidal Magnetic Field (from plasma current)", font_size=28)
        title2.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title2)
        self.play(Write(title2))
        
        poloidal_arrows = self._create_poloidal_field_arrows(R, r)
        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in poloidal_arrows], lag_ratio=0.1),
            run_time=2
        )
        self.wait(2)
        
        # === 3. Show combined helical field ===
        self.play(
            FadeOut(title2),
            FadeOut(toroidal_arrows),
            FadeOut(poloidal_arrows),
        )
        
        title3 = Text("Combined Field: Helical Field Lines", font_size=28)
        title3.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title3)
        self.play(Write(title3))
        
        # Create helical field lines on different flux surfaces
        field_lines = self._create_helical_field_lines(R, r)
        
        self.play(
            LaggedStart(*[Create(line) for line in field_lines], lag_ratio=0.1),
            run_time=4
        )
        
        # Rotate to show helical structure
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(6)
        self.stop_ambient_camera_rotation()
        
        # === 4. Explain safety factor ===
        self.play(FadeOut(title3))
        q_text = MathTex(r"q = \frac{\text{toroidal turns}}{\text{poloidal turn}}", 
                         font_size=32)
        q_text.to_edge(UP)
        self.add_fixed_in_frame_mobjects(q_text)
        self.play(Write(q_text))
        
        self.wait(3)
        
        self.play(FadeOut(q_text), FadeOut(field_lines), FadeOut(plasma))
    
    def _create_plasma_torus(self, R: float, r: float) -> Surface:
        """Create semi-transparent plasma torus."""
        return Surface(
            lambda u, v: np.array([
                (R + r * cos(v)) * cos(u),
                (R + r * cos(v)) * sin(u),
                r * sin(v)
            ]),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(32, 16),
            fill_color=YELLOW,
            fill_opacity=0.15,
            stroke_width=0,
        )
    
    def _create_toroidal_field_arrows(self, R: float, r: float) -> VGroup:
        """Create arrows showing toroidal field direction."""
        arrows = VGroup()
        
        n_phi = 16
        for i in range(n_phi):
            phi = i * TAU / n_phi
            
            # Position at midplane
            pos = np.array([
                (R + r * 0.5) * cos(phi),
                (R + r * 0.5) * sin(phi),
                0
            ])
            
            # Toroidal direction
            direction = np.array([
                -sin(phi),
                cos(phi),
                0
            ]) * 0.5
            
            arrow = Arrow3D(
                start=pos - direction * 0.5,
                end=pos + direction * 0.5,
                color=BLUE,
                thickness=0.02,
            )
            arrows.add(arrow)
        
        return arrows
    
    def _create_poloidal_field_arrows(self, R: float, r: float) -> VGroup:
        """Create arrows showing poloidal field direction."""
        arrows = VGroup()
        
        phi = 0  # Show in one slice
        n_theta = 12
        
        for i in range(n_theta):
            theta = i * TAU / n_theta
            r_local = r * 0.6
            
            pos = np.array([
                (R + r_local * cos(theta)) * cos(phi),
                (R + r_local * cos(theta)) * sin(phi),
                r_local * sin(theta)
            ])
            
            # Poloidal direction (tangent to poloidal circle)
            direction = np.array([
                -sin(theta) * cos(phi),
                -sin(theta) * sin(phi),
                cos(theta)
            ]) * 0.3
            
            arrow = Arrow3D(
                start=pos - direction * 0.5,
                end=pos + direction * 0.5,
                color=GREEN,
                thickness=0.02,
            )
            arrows.add(arrow)
        
        return arrows
    
    def _create_helical_field_lines(self, R: float, r: float) -> VGroup:
        """Create helical field lines on different flux surfaces."""
        lines = VGroup()
        
        # Different flux surfaces with different colors
        surfaces = [
            (0.2, 2.5, RED),      # Inner, low q
            (0.4, 3.0, ORANGE),
            (0.6, 3.5, YELLOW),
            (0.8, 4.0, GREEN),    # Outer, high q
        ]
        
        for r_frac, q, color in surfaces:
            r_surface = r * r_frac * 0.9
            
            # Multiple field lines per surface (different starting phases)
            for phase in [0, TAU/3, 2*TAU/3]:
                line = ParametricFunction(
                    lambda t, rs=r_surface, qq=q, ph=phase: np.array([
                        (R + rs * cos(t + ph)) * cos(qq * t),
                        (R + rs * cos(t + ph)) * sin(qq * t),
                        rs * sin(t + ph)
                    ]),
                    t_range=[0, TAU * 2, 0.02],
                    color=color,
                ).set_stroke(width=2, opacity=0.8)
                lines.add(line)
        
        return lines


class MagneticBottle(ThreeDScene):
    """
    Scene explaining the magnetic bottle/mirror concept that traps particles.
    Shows why particles bounce back from regions of stronger field.
    """
    
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=20 * DEGREES)
        
        # Title
        title = Text("Magnetic Bottle: Particle Trapping", font_size=32)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # === 1. Show converging field lines ===
        field_lines = self._create_bottle_field_lines()
        self.play(Create(field_lines), run_time=2)
        
        # Label the field strength
        weak_label = Text("Weak B", font_size=20, color=GREEN)
        strong_label1 = Text("Strong B", font_size=20, color=RED)
        strong_label2 = Text("Strong B", font_size=20, color=RED)
        
        weak_label.move_to([2.5, 0, 0])
        strong_label1.move_to([2.5, 0, 2.2])
        strong_label2.move_to([2.5, 0, -2.2])
        
        self.add_fixed_in_frame_mobjects(weak_label, strong_label1, strong_label2)
        self.play(
            Write(weak_label),
            Write(strong_label1),
            Write(strong_label2),
        )
        
        self.wait(1)
        
        # === 2. Animate trapped particle ===
        self.play(
            FadeOut(weak_label),
            FadeOut(strong_label1),
            FadeOut(strong_label2),
        )
        
        # Explanation text
        explanation = VGroup(
            Text("Particle gyrates around field line", font_size=20),
            Text("Bounces back at strong field regions", font_size=20),
            Text("→ Confined in the magnetic bottle!", font_size=20, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT)
        explanation.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(explanation)
        
        # Create and animate particle
        particle = Sphere(radius=0.08, color=YELLOW)
        particle.set_fill(YELLOW, opacity=1)
        
        # Particle trajectory
        trajectory = ParametricFunction(
            lambda t: self._particle_in_bottle(t),
            t_range=[0, TAU * 2, 0.01],
            color=YELLOW,
        ).set_stroke(width=1, opacity=0.3)
        
        self.add(particle)
        
        self.play(Write(explanation[0]))
        
        # Animate particle bouncing
        t_tracker = ValueTracker(0)
        particle.add_updater(
            lambda m: m.move_to(self._particle_in_bottle(t_tracker.get_value()))
        )
        
        self.play(
            t_tracker.animate.set_value(TAU),
            Write(explanation[1]),
            Create(trajectory),
            run_time=4,
            rate_func=linear,
        )
        
        self.play(
            t_tracker.animate.set_value(TAU * 2),
            Write(explanation[2]),
            run_time=4,
            rate_func=linear,
        )
        
        particle.clear_updaters()
        
        self.wait(2)
        
        # === 3. Connect to tokamak ===
        self.play(
            FadeOut(explanation),
            FadeOut(title),
        )
        
        tokamak_text = VGroup(
            Text("In a tokamak:", font_size=24),
            Text("• The twisted field lines form closed loops", font_size=20),
            Text("• Particles spiral around the torus", font_size=20),
            Text("• No open ends → better confinement!", font_size=20, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT)
        tokamak_text.to_corner(UL)
        self.add_fixed_in_frame_mobjects(tokamak_text)
        
        self.play(
            LaggedStart(*[Write(line) for line in tokamak_text], lag_ratio=0.5),
            run_time=4
        )
        
        self.wait(3)
    
    def _create_bottle_field_lines(self) -> VGroup:
        """Create magnetic bottle field lines that converge at ends."""
        lines = VGroup()
        
        n_lines = 12
        for i in range(n_lines):
            theta = i * TAU / n_lines
            
            line = ParametricFunction(
                lambda t, th=theta: self._bottle_field_line(t, th),
                t_range=[0, 1, 0.01],
                color=BLUE,
            ).set_stroke(width=2)
            lines.add(line)
        
        return lines
    
    def _bottle_field_line(self, t: float, theta: float) -> np.ndarray:
        """Field line in magnetic bottle geometry."""
        z_max = 2.0
        r_center = 1.0
        mirror_ratio = 4.0
        
        z = z_max * (2 * t - 1)
        
        # Field lines converge (smaller radius) where field is stronger
        r = r_center / np.sqrt(1 + (mirror_ratio - 1) * (z / z_max) ** 2)
        
        return np.array([
            r * cos(theta),
            r * sin(theta),
            z
        ])
    
    def _particle_in_bottle(self, t: float) -> np.ndarray:
        """Particle trajectory bouncing in magnetic bottle."""
        # Bouncing frequency along z
        bounce_freq = 2.0
        z = 1.6 * sin(bounce_freq * t)
        
        # Gyration frequency and radius
        gyro_freq = 15.0
        z_max = 2.0
        
        # Gyration radius decreases where field is stronger
        r_gyro = 0.3 / np.sqrt(1 + 2 * (z / z_max) ** 2)
        
        return np.array([
            r_gyro * cos(gyro_freq * t),
            r_gyro * sin(gyro_freq * t),
            z
        ])


class TokamakTo2DTransition(ThreeDScene):
    """
    Scene that transitions from 3D tokamak view to 2D poloidal cross-section.
    This bridges the 3D visualization with the 2D flux surface analysis.
    """
    
    def construct(self):
        R = 3.0
        r = 1.0
        
        # Start with 3D view
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
        
        # Create tokamak structure
        plasma = self._create_plasma_torus(R, r * 0.8)
        tf_coils = self._create_simplified_tf_coils(R, r)
        
        self.play(FadeIn(plasma), Create(tf_coils))
        
        # Rotate a bit to show 3D
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        
        # === Add slice plane ===
        title = Text("Poloidal Cross-Section", font_size=32)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create a vertical slice plane
        slice_plane = Surface(
            lambda u, v: np.array([
                R + 1.5 * r * u,
                0.01,  # Slightly off y=0 for visibility
                1.5 * r * v
            ]),
            u_range=[-1, 1],
            v_range=[-1.2, 1.2],
            fill_color=WHITE,
            fill_opacity=0.3,
            stroke_color=WHITE,
            stroke_width=1,
        )
        
        self.play(Create(slice_plane))
        self.wait(1)
        
        # === Transition camera to 2D view ===
        # Move camera to look along y-axis (seeing R-Z plane)
        self.play(
            self.camera.frame.animate.set_euler_angles(
                phi=90 * DEGREES,
                theta=0 * DEGREES,
            ),
            run_time=3
        )
        
        self.wait(1)
        
        # Highlight the cross-section
        cross_section = self._create_cross_section_outline(R, r)
        self.play(Create(cross_section))
        
        # Add labels
        r_label = MathTex("R", font_size=28)
        z_label = MathTex("Z", font_size=28)
        r_label.move_to([R + 1.3 * r, 0, -1.3 * r])
        z_label.move_to([R - 1.3 * r, 0, 1.3 * r])
        self.add_fixed_in_frame_mobjects(r_label, z_label)
        self.play(Write(r_label), Write(z_label))
        
        # Explain
        explanation = Text(
            "We study plasma equilibrium in this 2D slice",
            font_size=24
        )
        explanation.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(explanation)
        self.play(Write(explanation))
        
        self.wait(3)
        
        # Fade out for transition to next scene
        self.play(
            FadeOut(plasma),
            FadeOut(tf_coils),
            FadeOut(slice_plane),
            FadeOut(title),
            FadeOut(explanation),
            FadeOut(r_label),
            FadeOut(z_label),
        )
    
    def _create_plasma_torus(self, R: float, r: float) -> Surface:
        """Create semi-transparent plasma torus."""
        return Surface(
            lambda u, v: np.array([
                (R + r * cos(v)) * cos(u),
                (R + r * cos(v)) * sin(u),
                r * sin(v)
            ]),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(32, 16),
            fill_color=YELLOW,
            fill_opacity=0.25,
            stroke_width=0,
        )
    
    def _create_simplified_tf_coils(self, R: float, r: float) -> VGroup:
        """Create D-shaped TF coil representation."""
        coils = VGroup()
        n_coils = 8
        height = r * 1.8
        
        for i in range(n_coils):
            angle = i * TAU / n_coils
            coil = ParametricFunction(
                lambda t, a=angle, h=height: self._d_shaped_coil(t, R, h, a),
                t_range=[0, TAU, 0.02],
                color=BLUE,
            ).set_stroke(width=3)
            coils.add(coil)
        
        return coils
    
    def _d_shaped_coil(self, t: float, R: float, height: float, 
                       angle: float) -> np.ndarray:
        """
        Parametric D-shaped coil with 4 segments:
        1. Straight vertical inner edge
        2. Small radius arc at top
        3. Large radius arc on outside
        4. Small radius arc at bottom
        """
        R_inner = R * 0.35
        l = height * 2
        r_corner = height * 0.3
        R_arc = R * 1.2
        
        half_l = l / 2 - r_corner
        t_norm = t / TAU
        
        seg1, seg2, seg3, seg4 = 0.20, 0.10, 0.40, 0.10
        
        if t_norm < seg1:
            frac = t_norm / seg1
            R_pos = R_inner
            Z_pos = -half_l + (2 * half_l) * frac
        elif t_norm < seg1 + seg2:
            frac = (t_norm - seg1) / seg2
            theta = pi - frac * (pi / 2)
            R_pos = R_inner + r_corner + r_corner * cos(theta)
            Z_pos = half_l + r_corner * sin(theta)
        elif t_norm < seg1 + seg2 + seg3:
            frac = (t_norm - seg1 - seg2) / seg3
            theta = pi/2 - frac * pi
            arc_center_R = R_inner + r_corner
            R_pos = arc_center_R + R_arc * cos(theta)
            Z_pos = (half_l + r_corner) * sin(theta)
        elif t_norm < seg1 + seg2 + seg3 + seg4:
            frac = (t_norm - seg1 - seg2 - seg3) / seg4
            theta = -pi/2 - frac * (pi / 2)
            R_pos = R_inner + r_corner + r_corner * cos(theta)
            Z_pos = -half_l + r_corner * sin(theta)
        else:
            R_pos = R_inner
            Z_pos = -half_l
        
        return np.array([
            R_pos * cos(angle),
            R_pos * sin(angle),
            Z_pos
        ])
    
    def _create_cross_section_outline(self, R: float, r: float) -> VMobject:
        """Create D-shaped cross-section outline."""
        kappa = 1.7  # Elongation
        delta = 0.33  # Triangularity
        
        outline = ParametricFunction(
            lambda t: np.array([
                R + r * cos(t + delta * sin(t)),
                0,
                kappa * r * sin(t)
            ]),
            t_range=[0, TAU, 0.02],
            color=ORANGE,
        ).set_stroke(width=4)
        
        return outline
