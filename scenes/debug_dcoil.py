"""
Debug scene for analyzing D-shaped coil spline continuity.

Shows:
1. 2D view of D-coil path
2. Moving point with tangent/normal vectors
3. Graph tracking direction angle and curvature
4. 3D volume creation with cross-section animation
"""

from manim import *
import numpy as np
from numpy import pi, sin, cos

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import tube_along_path, d_shaped_coil_path
from utils.parametric import compute_fixed_frame, rectangular_cross_section

TAU = 2 * pi


class DCoilSplineAnalysis(Scene):
    """
    2D analysis of the D-shaped coil spline to identify discontinuities.
    """
    
    def construct(self):
        # Parameters
        R = 2.0
        height = 1.2
        angle = 0  # View in XZ plane (angle=0 means coil is in XZ plane)
        
        # Create the spline path (project to 2D: use R_pos as x, Z_pos as y)
        n_points = 200
        t_values = np.linspace(0, TAU, n_points)
        
        # Get points and compute tangents
        points_3d = [d_shaped_coil_path(t, R, height, angle) for t in t_values]
        
        # Project to 2D (X, Z) -> we'll use (x, z) as our 2D coordinates
        points_2d = np.array([[p[0], p[2]] for p in points_3d])
        
        # Scale and shift for display
        scale = 1.2
        shift = np.array([-0.5, 1.0])
        display_points = points_2d * scale + shift
        
        # Create the path curve
        path_curve = VMobject()
        path_curve.set_points_smoothly([np.array([p[0], p[1], 0]) for p in display_points])
        path_curve.set_color(BLUE)
        path_curve.set_stroke(width=3)
        
        # Compute tangent angles and curvature
        tangent_angles = []
        curvatures = []
        
        for i in range(n_points):
            # Tangent from finite difference
            if i == 0:
                tangent = points_2d[1] - points_2d[0]
            elif i == n_points - 1:
                tangent = points_2d[-1] - points_2d[-2]
            else:
                tangent = points_2d[i + 1] - points_2d[i - 1]
            
            norm = np.linalg.norm(tangent)
            if norm > 1e-10:
                tangent = tangent / norm
            
            # Angle of tangent
            angle_rad = np.arctan2(tangent[1], tangent[0])
            tangent_angles.append(angle_rad)
            
            # Curvature (rate of change of angle)
            if i > 0:
                d_angle = tangent_angles[i] - tangent_angles[i - 1]
                # Handle wrap-around
                if d_angle > pi:
                    d_angle -= TAU
                elif d_angle < -pi:
                    d_angle += TAU
                curvatures.append(d_angle * n_points / TAU)  # Scale by sampling rate
            else:
                curvatures.append(0)
        
        # Create axes for the graphs
        # Tangent angle graph
        angle_axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-4, 4, 1],
            x_length=10,
            y_length=2,
            tips=False,
            axis_config={"include_numbers": True, "font_size": 16},
        ).shift(DOWN * 2.5)
        
        angle_label = Text("Tangent Angle (rad)", font_size=16).next_to(angle_axes, UP, buff=0.1)
        
        # Plot tangent angle
        angle_graph = angle_axes.plot_line_graph(
            x_values=[i / (n_points - 1) for i in range(n_points)],
            y_values=tangent_angles,
            line_color=GREEN,
            add_vertex_dots=False,
            stroke_width=2,
        )
        
        # Segment boundaries (now based on arc-length proportions)
        # Computed from: len1=1.8, len2=1.13, len3=11.31, len4=1.13 => total=15.37
        # seg1 ≈ 0.117, seg2 ≈ 0.073, seg3 ≈ 0.736, seg4 ≈ 0.073
        seg_boundaries = [0.117, 0.190, 0.927]  # After seg1, after seg2, after seg3
        boundary_lines = VGroup()
        for seg in seg_boundaries:
            line = angle_axes.get_vertical_line(
                angle_axes.c2p(seg, 4),
                line_config={"color": RED, "stroke_width": 1, "stroke_opacity": 0.5}
            )
            boundary_lines.add(line)
        
        # Moving point on the path
        moving_dot = Dot(color=YELLOW, radius=0.08)
        moving_dot.move_to(np.array([display_points[0][0], display_points[0][1], 0]))
        
        # Tangent arrow
        tangent_arrow = Arrow(
            start=ORIGIN, end=RIGHT * 0.5,
            color=GREEN, buff=0,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.3,
        )
        
        # Normal arrow
        normal_arrow = Arrow(
            start=ORIGIN, end=UP * 0.3,
            color=RED, buff=0,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.3,
        )
        
        # Tracker dot on graph
        graph_dot = Dot(color=YELLOW, radius=0.06)
        
        # Title
        title = Text("D-Coil Spline Analysis", font_size=28).to_edge(UP)
        
        # Parameter display
        param_text = Variable(0, r"t/\tau", num_decimal_places=3).scale(0.7)
        param_text.next_to(title, DOWN)
        
        # Build scene
        self.add(title)
        self.play(Create(path_curve), run_time=2)
        self.play(
            Create(angle_axes),
            Write(angle_label),
            Create(angle_graph),
            Create(boundary_lines),
        )
        self.add(param_text)
        
        # Add moving elements
        self.add(moving_dot, tangent_arrow, normal_arrow, graph_dot)
        
        # Animation: move point around the path
        def update_point(mob, alpha):
            idx = int(alpha * (n_points - 1))
            idx = min(idx, n_points - 1)
            
            # Update dot position
            pos = np.array([display_points[idx][0], display_points[idx][1], 0])
            mob.move_to(pos)
            
            # Update tangent arrow
            if idx < n_points - 1:
                tangent = display_points[idx + 1] - display_points[idx]
            else:
                tangent = display_points[idx] - display_points[idx - 1]
            
            norm = np.linalg.norm(tangent)
            if norm > 1e-10:
                tangent = tangent / norm
            
            tangent_arrow.put_start_and_end_on(
                pos,
                pos + np.array([tangent[0], tangent[1], 0]) * 0.5
            )
            
            # Update normal arrow (perpendicular to tangent)
            normal = np.array([-tangent[1], tangent[0]])
            normal_arrow.put_start_and_end_on(
                pos,
                pos + np.array([normal[0], normal[1], 0]) * 0.3
            )
            
            # Update graph dot
            graph_dot.move_to(angle_axes.c2p(alpha, tangent_angles[idx]))
            
            # Update parameter display
            param_text.tracker.set_value(alpha)
        
        # Animate
        self.play(
            UpdateFromAlphaFunc(moving_dot, update_point),
            run_time=10,
            rate_func=linear,
        )
        
        self.wait(1)
        
        # Highlight problem areas
        problem_text = Text("Red lines = segment boundaries", font_size=20, color=RED)
        problem_text.next_to(angle_axes, DOWN)
        self.play(Write(problem_text))
        
        # Show where jumps occur
        jump_text = Text("Sharp jumps in angle = discontinuous tangent!", font_size=20, color=YELLOW)
        jump_text.next_to(problem_text, DOWN)
        self.play(Write(jump_text))
        
        self.wait(2)


class DCoilSplineAnalysis3D(ThreeDScene):
    """
    3D visualization showing the coil path and where volume creation fails.
    """
    
    def construct(self):
        self.camera.background_color = "#1e1e1e"
        
        R = 3.0
        height = 1.8
        angle = 0
        
        self.set_camera_orientation(phi=90 * DEGREES, theta=-90 * DEGREES)
        
        # Create path as a 3D curve
        n_points = 100
        t_values = np.linspace(0, TAU, n_points)
        points = [d_shaped_coil_path(t, R, height, angle) for t in t_values]
        
        path_curve = VMobject()
        path_curve.set_points_smoothly([np.array(p) for p in points])
        path_curve.set_color(BLUE)
        path_curve.set_stroke(width=4)
        
        self.play(Create(path_curve), run_time=2)
        
        # Add markers at segment boundaries (now 4 segments)
        # seg1 ≈ 0.117, seg2 ≈ 0.073, seg3 ≈ 0.736, seg4 ≈ 0.073
        seg_boundaries = [0.117, 0.190, 0.927]
        markers = VGroup()
        labels = VGroup()
        
        seg_names = ["Seg1→2", "Seg2→3", "Seg3→4"]
        for i, seg in enumerate(seg_boundaries):
            t = seg * TAU
            point = d_shaped_coil_path(t, R, height, angle)
            marker = Sphere(radius=0.1, color=RED).move_to(point)
            markers.add(marker)
            
            label = Text(seg_names[i], font_size=16, color=RED)
            label.move_to(point + np.array([0.3, 0, 0.3]))
            labels.add(label)
        
        self.play(Create(markers))
        self.add_fixed_in_frame_mobjects(*labels)
        self.play(*[Write(l) for l in labels])
        
        self.wait(1)
        
        # Now try to create the volume
        title = Text("Creating 3D volume...", font_size=24)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create the tube
        path_func = lambda t: d_shaped_coil_path(t, R, height, angle)
        
        coil_volume = tube_along_path(
            path_func=path_func,
            t_range=(0, TAU),
            cross_section="rectangular",
            width=0.2,
            height=0.3,
            reference="radial",
            closed=True,
            path_resolution=48,
            cross_resolution=8,
            fill_color=BLUE_D,
            fill_opacity=0.8,
        )
        
        self.play(
            FadeOut(path_curve),
            FadeIn(coil_volume),
            run_time=2
        )
        
        """         # Rotate to show 3D
        self.move_camera(
            phi=70 * DEGREES,
            theta=30 * DEGREES,
            run_time=3
        )
        
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(5)
        self.stop_ambient_camera_rotation() """


class DCoilContinuityTest(Scene):
    """
    Detailed test showing exactly where the spline has discontinuities.
    """
    
    def construct(self):
        R = 2.0
        height = 1.2
        angle = 0
        
        n_points = 500  # High resolution
        t_values = np.linspace(0, TAU, n_points)
        
        # Get points
        points_3d = [d_shaped_coil_path(t, R, height, angle) for t in t_values]
        points_2d = np.array([[p[0], p[2]] for p in points_3d])
        
        # Compute tangent vectors
        tangents = []
        for i in range(n_points):
            if i == 0:
                t = points_2d[1] - points_2d[0]
            elif i == n_points - 1:
                t = points_2d[-1] - points_2d[-2]
            else:
                t = points_2d[i + 1] - points_2d[i - 1]
            
            norm = np.linalg.norm(t)
            if norm > 1e-10:
                t = t / norm
            tangents.append(t)
        
        # Compute tangent angle
        angles = [np.arctan2(t[1], t[0]) for t in tangents]
        
        # Compute angular velocity (d_angle / dt)
        angular_vel = [0]
        for i in range(1, n_points):
            d_angle = angles[i] - angles[i - 1]
            if d_angle > pi:
                d_angle -= TAU
            elif d_angle < -pi:
                d_angle += TAU
            angular_vel.append(d_angle * n_points)  # Scale
        
        # Compute angular acceleration (curvature rate of change)
        angular_accel = [0, 0]
        for i in range(2, n_points):
            angular_accel.append((angular_vel[i] - angular_vel[i - 1]) * n_points)
        
        # Create graphs
        title = Text("D-Coil Spline Continuity Analysis", font_size=24).to_edge(UP)
        
        # Tangent angle graph
        axes1 = Axes(
            x_range=[0, 1, 0.2],
            y_range=[-4, 4, 2],
            x_length=10,
            y_length=1.5,
            tips=False,
        ).shift(UP * 1.5)
        label1 = Text("Tangent Angle", font_size=14).next_to(axes1, LEFT)
        
        graph1 = axes1.plot_line_graph(
            x_values=[i / (n_points - 1) for i in range(n_points)],
            y_values=angles,
            line_color=GREEN,
            add_vertex_dots=False,
            stroke_width=1.5,
        )
        
        # Angular velocity graph
        axes2 = Axes(
            x_range=[0, 1, 0.2],
            y_range=[-50, 50, 25],
            x_length=10,
            y_length=1.5,
            tips=False,
        ).shift(DOWN * 0.5)
        label2 = Text("Angular Velocity", font_size=14).next_to(axes2, LEFT)
        
        # Clip extreme values for visualization
        angular_vel_clipped = [max(-50, min(50, v)) for v in angular_vel]
        graph2 = axes2.plot_line_graph(
            x_values=[i / (n_points - 1) for i in range(n_points)],
            y_values=angular_vel_clipped,
            line_color=YELLOW,
            add_vertex_dots=False,
            stroke_width=1.5,
        )
        
        # Angular acceleration graph
        axes3 = Axes(
            x_range=[0, 1, 0.2],
            y_range=[-500, 500, 250],
            x_length=10,
            y_length=1.5,
            tips=False,
        ).shift(DOWN * 2.5)
        label3 = Text("Angular Accel (Curvature')", font_size=14).next_to(axes3, LEFT)
        
        # Clip extreme values
        angular_accel_clipped = [max(-500, min(500, v)) for v in angular_accel]
        graph3 = axes3.plot_line_graph(
            x_values=[i / (n_points - 1) for i in range(n_points)],
            y_values=angular_accel_clipped,
            line_color=RED,
            add_vertex_dots=False,
            stroke_width=1.5,
        )
        
        # Segment boundaries (now arc-length proportioned - 4 segments)
        seg_boundaries = [0.117, 0.190, 0.927]
        
        def add_boundary_lines(axes, y_max):
            lines = VGroup()
            for seg in seg_boundaries:
                line = DashedLine(
                    start=axes.c2p(seg, -y_max),
                    end=axes.c2p(seg, y_max),
                    color=WHITE,
                    stroke_width=1,
                    stroke_opacity=0.5,
                )
                lines.add(line)
            return lines
        
        bounds1 = add_boundary_lines(axes1, 4)
        bounds2 = add_boundary_lines(axes2, 50)
        bounds3 = add_boundary_lines(axes3, 500)
        
        # Build scene
        self.add(title)
        self.play(
            Create(axes1), Write(label1), Create(graph1), Create(bounds1),
            run_time=2
        )
        self.play(
            Create(axes2), Write(label2), Create(graph2), Create(bounds2),
            run_time=2
        )
        self.play(
            Create(axes3), Write(label3), Create(graph3), Create(bounds3),
            run_time=2
        )
        
        # Explanation
        explanation = VGroup(
            Text("White lines = segment boundaries", font_size=16),
            Text("Spikes in red graph = discontinuous curvature", font_size=16, color=RED),
            Text("This causes the cross-section to collapse!", font_size=16, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN)
        
        self.play(Write(explanation), run_time=2)
        self.wait(3)


class DCoilCrossSectionAnimation(ThreeDScene):
    """
    3D animation showing a cross-section moving along the D-coil path,
    with the volume being created as it sweeps.
    """
    
    def construct(self):
        self.camera.background_color = "#1e1e1e"
        
        # Coil parameters
        R = 3.0
        height = 1.8
        angle = 0
        
        # Cross-section parameters
        cs_width = 0.25
        cs_height = 0.35
        
        # Set up camera - front view initially
        self.set_camera_orientation(phi=80 * DEGREES, theta=20 * DEGREES, zoom=0.8)
        
        # Create path function
        path_func = lambda t: d_shaped_coil_path(t, R, height, angle)
        
        # Sample the path for the curve visualization
        n_path_points = 100
        t_path = np.linspace(0, TAU, n_path_points)
        path_points = [path_func(t) for t in t_path]
        
        # Create the path curve
        path_curve = VMobject()
        path_curve.set_points_smoothly([np.array(p) for p in path_points])
        path_curve.set_color(BLUE)
        path_curve.set_stroke(width=3)
        
        # Title
        title = Text("D-Coil Cross-Section Animation", font_size=24)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        
        # Show the path first
        self.play(Create(path_curve), run_time=2)
        self.wait(0.5)
        
        # Precompute frame data for animation
        n_frames = 150  # Number of animation frames
        t_values = np.linspace(0, TAU * 0.99, n_frames)  # Slightly less than TAU to avoid exact closure
        
        # Compute fixed frame along the path
        points, tangents, normals, binormals = compute_fixed_frame(
            path_func, t_values, reference="radial"
        )
        
        # Create cross-section polygon (rectangle)
        def create_cross_section_at(idx, color=YELLOW):
            """Create a rectangular cross-section at frame index idx."""
            pos = points[idx]
            normal = normals[idx]
            binormal = binormals[idx]
            
            # Rectangle corners in local frame
            hw, hh = cs_width / 2, cs_height / 2
            corners_local = [
                (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
            ]
            
            # Transform to 3D
            corners_3d = []
            for u, v in corners_local:
                corner = pos + u * normal + v * binormal
                corners_3d.append(corner)
            
            # Create polygon
            polygon = Polygon(
                *[np.array(c) for c in corners_3d],
                color=color,
                fill_color=color,
                fill_opacity=0.6,
                stroke_width=2,
            )
            return polygon
        
        # Create initial cross-section
        cross_section = create_cross_section_at(0)
        
        # Create tangent/normal/binormal arrows
        arrow_scale = 0.5
        tangent_arrow = Arrow3D(
            start=points[0],
            end=points[0] + tangents[0] * arrow_scale,
            color=GREEN,
        )
        normal_arrow = Arrow3D(
            start=points[0],
            end=points[0] + normals[0] * arrow_scale,
            color=RED,
        )
        binormal_arrow = Arrow3D(
            start=points[0],
            end=points[0] + binormals[0] * arrow_scale,
            color=BLUE,
        )
        
        # Add legend
        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=GREEN), Text("Tangent", font_size=14)).arrange(RIGHT, buff=0.1),
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=RED), Text("Normal", font_size=14)).arrange(RIGHT, buff=0.1),
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=BLUE), Text("Binormal", font_size=14)).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UL).shift(DOWN * 0.8)
        self.add_fixed_in_frame_mobjects(legend)
        
        # Add position tracker
        pos_tracker = Variable(0, r"t/\tau", num_decimal_places=3).scale(0.6)
        pos_tracker.to_corner(UR).shift(DOWN * 0.8)
        self.add_fixed_in_frame_mobjects(pos_tracker)
        
        self.play(
            FadeIn(cross_section),
            FadeIn(tangent_arrow),
            FadeIn(normal_arrow),
            FadeIn(binormal_arrow),
            Write(legend),
            Write(pos_tracker),
        )
        
        self.wait(0.5)
        
        # Create the growing tube surface
        # We'll build it up by adding segments
        tube_segments = VGroup()
        
        # Animation: Move cross-section along path and build tube
        segment_interval = 3  # Add a tube segment every N frames
        
        def update_cross_section(mob, alpha):
            """Update the cross-section position and orientation."""
            idx = int(alpha * (n_frames - 1))
            idx = min(idx, n_frames - 1)
            
            pos = points[idx]
            normal = normals[idx]
            binormal = binormals[idx]
            tangent = tangents[idx]
            
            # Update cross-section polygon
            hw, hh = cs_width / 2, cs_height / 2
            corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            new_corners = [pos + u * normal + v * binormal for u, v in corners_local]
            
            # Recreate the polygon with new vertices
            mob.become(Polygon(
                *[np.array(c) for c in new_corners],
                color=YELLOW,
                fill_color=YELLOW,
                fill_opacity=0.6,
                stroke_width=2,
            ))
            
            # Update arrows
            tangent_arrow.become(Arrow3D(
                start=pos,
                end=pos + tangent * arrow_scale,
                color=GREEN,
            ))
            normal_arrow.become(Arrow3D(
                start=pos,
                end=pos + normal * arrow_scale,
                color=RED,
            ))
            binormal_arrow.become(Arrow3D(
                start=pos,
                end=pos + binormal * arrow_scale,
                color=BLUE,
            ))
            
            # Update position tracker
            pos_tracker.tracker.set_value(alpha)
        
        # Create tube surface incrementally
        def create_tube_segment(t_start, t_end, resolution=8):
            """Create a tube segment between two t values."""
            t_seg = np.linspace(t_start, t_end, resolution)
            pts, tans, norms, binorms = compute_fixed_frame(path_func, t_seg, reference="radial")
            
            # Create surface function for this segment
            def surface_func(u, v):
                # u: along path (0 to 1 within segment)
                # v: around cross-section (0 to TAU)
                idx = int(u * (resolution - 1))
                idx = min(idx, resolution - 1)
                
                pos = pts[idx]
                normal = norms[idx]
                binormal = binorms[idx]
                
                # Rectangular cross-section
                hw, hh = cs_width / 2, cs_height / 2
                # Map v to rectangle perimeter
                v_norm = v / TAU
                if v_norm < 0.25:
                    local_u = -hw + (v_norm / 0.25) * (2 * hw)
                    local_v = -hh
                elif v_norm < 0.5:
                    local_u = hw
                    local_v = -hh + ((v_norm - 0.25) / 0.25) * (2 * hh)
                elif v_norm < 0.75:
                    local_u = hw - ((v_norm - 0.5) / 0.25) * (2 * hw)
                    local_v = hh
                else:
                    local_u = -hw
                    local_v = hh - ((v_norm - 0.75) / 0.25) * (2 * hh)
                
                return pos + local_u * normal + local_v * binormal
            
            return Surface(
                surface_func,
                u_range=[0, 1],
                v_range=[0, TAU],
                resolution=(resolution, 16),
                fill_color=BLUE_D,
                fill_opacity=0.7,
                stroke_width=0,
            )
        
        # Main animation
        self.play(
            UpdateFromAlphaFunc(cross_section, update_cross_section),
            run_time=12,
            rate_func=linear,
        )
        
        self.wait(0.5)
        
        # Now show the complete tube
        info_text = Text("Creating complete volume...", font_size=20)
        info_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(info_text)
        self.play(Write(info_text))
        
        # Create full tube
        full_tube = tube_along_path(
            path_func=path_func,
            t_range=(0, TAU),
            cross_section="rectangular",
            width=cs_width,
            height=cs_height,
            reference="radial",
            closed=True,
            path_resolution=64,
            cross_resolution=16,
            fill_color=BLUE_D,
            fill_opacity=0.8,
        )
        
        self.play(
            FadeOut(cross_section),
            FadeOut(tangent_arrow),
            FadeOut(normal_arrow),
            FadeOut(binormal_arrow),
            FadeOut(path_curve),
            FadeIn(full_tube),
            run_time=2
        )
        
        # Rotate to show 3D
        self.play(FadeOut(info_text))
        
        final_text = Text("Complete D-coil volume", font_size=20)
        final_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(final_text)
        self.play(Write(final_text))
        
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(6)
        self.stop_ambient_camera_rotation()


if __name__ == "__main__":
    # For quick testing
    print("Run with: manim -pql scenes/debug_dcoil.py DCoilSplineAnalysis")
    print("Or: manim -pql scenes/debug_dcoil.py DCoilContinuityTest")
    print("Or: manim -pql scenes/debug_dcoil.py DCoilCrossSectionAnimation")
