"""
STL generation module for creating 3D models of the direction sign.
"""

from typing import List, Tuple
import numpy as np
from stl import mesh
import math
import trimesh


class DirectionSignGenerator:
    """Generates 3D models for direction signs."""
    
    def __init__(self, 
                 post_height: float = 200.0,
                 post_radius: float = 15.0,
                 base_radius: float = 40.0,
                 base_height: float = 10.0,
                 sign_width: float = 100.0,
                 sign_height: float = 20.0,
                 sign_thickness: float = 3.0,
                 flat_depth: float = 3.0,
                 flat_height: float = 28.0,
                 arrow_length: float = 15.0,
                 arrow_width: float = 8.0,
                 sign_spacing: float = 36.0):
        """
        Initialize the sign generator with dimensions (all in mm).
        
        Args:
            post_height: Height of the main post
            post_radius: Radius of the cylindrical post
            base_radius: Radius of the base platform
            base_height: Height of the base
            sign_width: Width of each directional sign
            sign_height: Height of each directional sign
            sign_thickness: Thickness of sign plates
            flat_depth: How deep to cut the flat surface into the post
            flat_height: Height of each flat surface on the post
            arrow_length: Length of the north arrow
            arrow_width: Width of the north arrow at base
            sign_spacing: Vertical spacing between signs (center to center)
        """
        self.post_height = post_height
        self.post_radius = post_radius
        self.base_radius = base_radius
        self.base_height = base_height
        self.sign_width = sign_width
        self.sign_height = sign_height
        self.sign_thickness = sign_thickness
        self.flat_depth = flat_depth
        self.flat_height = flat_height
        self.arrow_length = arrow_length
        self.arrow_width = arrow_width
        self.sign_spacing = sign_spacing
    
    def generate_post(self, bearings: List[float], output_path: str):
        """
        Generate two post sections with flat indents at specified bearings.
        Creates a two-piece design to fit within 210mm print height.
        Uses boolean subtraction to create flat mounting surfaces.
        
        Args:
            bearings: List of bearings (in degrees) where signs will attach
            output_path: Path to save the STL file
        """
        print(f"Generating 2-piece post with {len(bearings)} flat indents...")
        
        all_meshes = []
        
        # Configuration
        segments = 64
        post_separation = 150.0  # Distance between the two posts when laid side by side
        max_post_height = 200.0  # Maximum height for each post section
        
        # Calculate sign distribution and heights
        bottom_start_height = 50.0  # Start first sign 50mm from bottom
        sign_increment = self.flat_height + 8.0  # flat_height (28mm) + spacing (8mm)
        top_padding = 8.0  # Spacing above the last sign
        
        # Determine how many signs fit on bottom post
        num_signs_bottom = 0
        current_height = bottom_start_height
        while current_height + self.flat_height <= max_post_height and num_signs_bottom < len(bearings):
            num_signs_bottom += 1
            current_height += sign_increment
        
        # Calculate actual bottom post height (last sign center + half flat height + padding)
        bottom_post_height = bottom_start_height + (num_signs_bottom - 1) * sign_increment + self.flat_height / 2 + top_padding
        
        # Remaining signs go on top post
        num_signs_top = len(bearings) - num_signs_bottom
        
        # Calculate top post height
        if num_signs_top > 0:
            # First sign BOTTOM should be at 8mm above Z=0
            # Since we position by center, center = bottom + flat_height/2
            top_start_height = 8.0 + self.flat_height / 2  # 8mm (spacing) + 14mm (half flat) = 22mm center
            # Post height goes from 0 to (last sign center + half flat height + padding)
            top_post_height = top_start_height + (num_signs_top - 1) * sign_increment + self.flat_height / 2 + top_padding
        else:
            top_post_height = 0
            top_start_height = 0
        
        print(f"  Bottom post: {num_signs_bottom} signs, height: {bottom_post_height:.1f}mm")
        print(f"  Top post: {num_signs_top} signs, height: {top_post_height:.1f}mm")
        print(f"  Sign increment: {sign_increment:.1f}mm (flat_height {self.flat_height} + spacing 8mm)")
        
        # ===== CREATE BOTTOM POST =====
        print("  Creating bottom post cylinder...")
        bottom_post_mesh = trimesh.creation.cylinder(
            radius=self.post_radius,
            height=bottom_post_height,
            sections=64
        )
        # Trimesh creates cylinder centered at origin, translate to sit on Z=0
        bottom_post_mesh.apply_translation([0, 0, bottom_post_height / 2])
        
        # Subtract boxes for bottom post signs
        print("  Subtracting boxes from bottom post...")
        for i in range(num_signs_bottom):
            bearing = bearings[i]
            sign_height = bottom_start_height + (i * sign_increment)
            
            print(f"    Box {i+1}: bearing {bearing:.1f}°, height {sign_height:.1f}mm")
            
            box_mesh = self._create_box_mesh_at_bearing(bearing, sign_height, 0, 0)
            try:
                new_mesh = bottom_post_mesh.difference(box_mesh)
                if new_mesh is not None and len(new_mesh.faces) > 0:
                    bottom_post_mesh = new_mesh
                else:
                    print(f"      Warning: Boolean operation returned empty mesh")
            except Exception as e:
                print(f"      Warning: Boolean operation failed: {e}")
        
        all_meshes.append(bottom_post_mesh)
        
        # ===== CREATE TOP POST (offset to the side) =====
        if num_signs_top > 0:
            print("  Creating top post cylinder...")
            x_offset = post_separation
            
            top_post_mesh = trimesh.creation.cylinder(
                radius=self.post_radius,
                height=top_post_height,
                sections=64
            )
            # Translate to sit on Z=0 and offset to the side
            top_post_mesh.apply_translation([x_offset, 0, top_post_height / 2])
            
            # Subtract boxes for top post signs
            print("  Subtracting boxes from top post...")
            for i in range(num_signs_top):
                bearing = bearings[num_signs_bottom + i]
                sign_bottom = 8.0 + (i * sign_increment)
                sign_center = sign_bottom + self.flat_height / 2
                
                print(f"    Box {i+1}: bearing {bearing:.1f}°, center height {sign_center:.1f}mm")
                
                box_mesh = self._create_box_mesh_at_bearing(bearing, sign_center, x_offset, 0)
                try:
                    new_mesh = top_post_mesh.difference(box_mesh)
                    if new_mesh is not None and len(new_mesh.faces) > 0:
                        top_post_mesh = new_mesh
                    else:
                        print(f"      Warning: Boolean operation returned empty mesh")
                except Exception as e:
                    print(f"      Warning: Boolean operation failed: {e}")
            
            all_meshes.append(top_post_mesh)
        
        # Combine all meshes
        print("  Combining meshes...")
        combined_mesh = trimesh.util.concatenate(all_meshes)
        
        # Export to STL
        combined_mesh.export(output_path)
        print(f"  Saved: {output_path}")
    
    def _create_box_at_bearing(self, bearing: float, sign_height: float, 
                               post_x_offset: float, post_y_offset: float,
                               vertex_offset: int) -> Tuple[List, List]:
        """
        Create a box at a specific bearing and height.
        
        Args:
            bearing: Bearing angle in degrees
            sign_height: Height on the post
            post_x_offset: X offset of the post center
            post_y_offset: Y offset of the post center
            vertex_offset: Starting index for vertices
            
        Returns:
            Tuple of (vertices, faces)
        """
        # Box dimensions
        box_width = self.sign_width
        box_height = self.flat_height
        # Box depth - this is how thick the box is (in the radial direction)
        # Should be just enough to cut into the post and stick out a bit
        box_depth = self.flat_depth * 3  # e.g., 3mm cut depth × 3 = 9mm total box depth
        
        # Create box vertices (8 corners)
        # X = width (perpendicular to bearing)
        # Y = depth (radial direction - toward/away from center)
        # Z = height (vertical)
        half_w = box_width / 2
        half_h = box_height / 2
        half_d = box_depth / 2
        
        # Orient box so Y-axis points radially outward (the "depth" direction)
        # X-axis is tangent to the circle (the "width" direction)
        # The inner face is at Y = -half_d, outer face at Y = +half_d
        box_vertices_local = [
            [-half_w, -half_d, -half_h],  # 0: back bottom left (inner, left)
            [half_w, -half_d, -half_h],   # 1: back bottom right (inner, right)
            [half_w, half_d, -half_h],    # 2: front bottom right (outer, right)
            [-half_w, half_d, -half_h],   # 3: front bottom left (outer, left)
            [-half_w, -half_d, half_h],   # 4: back top left (inner, left)
            [half_w, -half_d, half_h],    # 5: back top right (inner, right)
            [half_w, half_d, half_h],     # 6: front top right (outer, right)
            [-half_w, half_d, half_h],    # 7: front top left (outer, left)
        ]
        
        # Transform box to correct position
        bearing_rad = math.radians(bearing)
        
        # We want the inner face (at local Y = -half_d) to be at distance (radius - flat_depth) from center
        # So the box center should be at: (radius - flat_depth) + half_d
        distance_from_center = self.post_radius - self.flat_depth + half_d
        
        print(f"    Box positioning: radius={self.post_radius}, flat_depth={self.flat_depth}, "
              f"box_depth={box_depth}, half_d={half_d}")
        print(f"    Inner face at: {self.post_radius - self.flat_depth}, "
              f"Box center at: {distance_from_center}, "
              f"Outer face at: {distance_from_center + half_d}")
        
        # Position box initially along +Y axis (bearing = 0°)
        # Box center is at (0, distance_from_center, sign_height)
        x_base = 0
        y_base = distance_from_center
        z_base = sign_height
        
        # Now rotate the entire positioned box around the Z-axis by the bearing angle
        box_vertices = []
        for vx, vy, vz in box_vertices_local:
            # The box vertex in local coordinates
            # Add to base position (this gives us the vertex position at bearing=0)
            x_at_zero = x_base + vx
            y_at_zero = y_base + vy
            z_at_zero = z_base + vz
            
            # Now rotate this point around the Z-axis (post center) by bearing angle
            x_final = x_at_zero * math.cos(bearing_rad) - y_at_zero * math.sin(bearing_rad)
            y_final = x_at_zero * math.sin(bearing_rad) + y_at_zero * math.cos(bearing_rad)
            z_final = z_at_zero
            
            # Apply post offset
            x_final += post_x_offset
            y_final += post_y_offset
            
            box_vertices.append([x_final, y_final, z_final])
        
        # Box faces (12 triangles for 6 faces)
        box_faces = [
            # Bottom face (vertices 0,1,2,3)
            [0, 2, 1], [0, 3, 2],
            # Top face (vertices 4,5,6,7)
            [4, 5, 6], [4, 6, 7],
            # Front face (vertices 2,3,6,7)
            [2, 3, 7], [2, 7, 6],
            # Back face (vertices 0,1,5,4)
            [0, 1, 5], [0, 5, 4],
            # Left face (vertices 0,3,7,4)
            [0, 3, 7], [0, 7, 4],
            # Right face (vertices 1,2,6,5)
            [1, 2, 6], [1, 6, 5],
        ]
        
        # Add vertex offset to face indices
        box_faces = [[f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset] for f in box_faces]
        
        return box_vertices, box_faces
    
    def _create_box_mesh_at_bearing(self, bearing: float, sign_height: float,
                                    post_x_offset: float, post_y_offset: float) -> trimesh.Trimesh:
        """
        Create a trimesh box at a specific bearing and height for boolean operations.
        
        Args:
            bearing: Bearing angle in degrees
            sign_height: Height on the post (center of the flat)
            post_x_offset: X offset of the post center
            post_y_offset: Y offset of the post center
            
        Returns:
            trimesh.Trimesh: Box mesh positioned at the bearing
        """
        # Box dimensions
        box_depth = self.flat_depth * 3  # 9mm - extends through post
        box_width = self.post_radius * 2  # Wide enough to cover post diameter
        box_height = self.flat_height
        
        # Create box centered at origin
        box = trimesh.creation.box(extents=[box_width, box_depth, box_height])
        
        # Position the box at bearing 0 (North, +Y axis)
        # Box should be tangent to post surface (not cutting through center)
        distance_from_center = self.post_radius - self.flat_depth + box_depth / 2
        
        # Position box along +Y axis at bearing 0
        box.apply_translation([0, distance_from_center, sign_height])
        
        # Rotate around Z-axis by bearing angle
        angle_rad = math.radians(bearing)
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
        box.apply_transform(rotation_matrix)
        
        # Apply post offset
        box.apply_translation([post_x_offset, post_y_offset, 0])
        
        return box
    
    def generate_base(self, output_path: str):
        """
        Generate the base with a north arrow indicator.
        
        Args:
            output_path: Path to save the STL file
        """
        print(f"Generating base with north arrow...")
        
        vertices = []
        faces = []
        
        # Create circular base disk
        segments = 64
        
        # Bottom center
        bottom_center_idx = 0
        vertices.append([0, 0, 0])
        
        # Top center
        top_center_idx = 1
        vertices.append([0, 0, self.base_height])
        
        # Generate circular perimeter vertices
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = self.base_radius * math.cos(angle)
            y = self.base_radius * math.sin(angle)
            
            vertices.append([x, y, 0])  # Bottom
            vertices.append([x, y, self.base_height])  # Top
        
        # Create bottom face (triangles from center to perimeter)
        for i in range(segments):
            next_i = (i + 1) % segments
            bottom_outer = 2 + i * 2
            bottom_outer_next = 2 + next_i * 2
            faces.append([bottom_center_idx, bottom_outer_next, bottom_outer])
        
        # Create top face
        for i in range(segments):
            next_i = (i + 1) % segments
            top_outer = 2 + i * 2 + 1
            top_outer_next = 2 + next_i * 2 + 1
            faces.append([top_center_idx, top_outer, top_outer_next])
        
        # Create side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            bottom_outer = 2 + i * 2
            top_outer = 2 + i * 2 + 1
            bottom_outer_next = 2 + next_i * 2
            top_outer_next = 2 + next_i * 2 + 1
            
            faces.append([bottom_outer, top_outer, bottom_outer_next])
            faces.append([top_outer, top_outer_next, bottom_outer_next])
        
        # Add north arrow on top (pointing in +Y direction, which is 0° bearing)
        arrow_height = 2.0  # Height of arrow above base
        arrow_base_y = self.base_radius * 0.6
        arrow_tip_y = self.base_radius * 0.85
        
        # Arrow shaft (rectangle)
        shaft_width = self.arrow_width * 0.3
        shaft_vertices_start = len(vertices)
        
        # Arrow shaft vertices
        vertices.extend([
            [-shaft_width/2, arrow_base_y - self.arrow_length*0.3, self.base_height],
            [shaft_width/2, arrow_base_y - self.arrow_length*0.3, self.base_height],
            [-shaft_width/2, arrow_base_y, self.base_height],
            [shaft_width/2, arrow_base_y, self.base_height],
            [-shaft_width/2, arrow_base_y - self.arrow_length*0.3, self.base_height + arrow_height],
            [shaft_width/2, arrow_base_y - self.arrow_length*0.3, self.base_height + arrow_height],
            [-shaft_width/2, arrow_base_y, self.base_height + arrow_height],
            [shaft_width/2, arrow_base_y, self.base_height + arrow_height],
        ])
        
        # Arrow head (triangle)
        head_vertices_start = len(vertices)
        vertices.extend([
            [-self.arrow_width/2, arrow_base_y, self.base_height],
            [self.arrow_width/2, arrow_base_y, self.base_height],
            [0, arrow_tip_y, self.base_height],
            [-self.arrow_width/2, arrow_base_y, self.base_height + arrow_height],
            [self.arrow_width/2, arrow_base_y, self.base_height + arrow_height],
            [0, arrow_tip_y, self.base_height + arrow_height],
        ])
        
        # Create arrow shaft faces
        s = shaft_vertices_start
        faces.extend([
            # Bottom
            [s, s+2, s+1], [s+1, s+2, s+3],
            # Top
            [s+4, s+5, s+6], [s+5, s+7, s+6],
            # Sides
            [s, s+1, s+4], [s+1, s+5, s+4],
            [s+2, s+6, s+3], [s+3, s+6, s+7],
            [s, s+4, s+2], [s+2, s+4, s+6],
            [s+1, s+3, s+5], [s+3, s+7, s+5],
        ])
        
        # Create arrow head faces
        h = head_vertices_start
        faces.extend([
            # Bottom
            [h, h+2, h+1],
            # Top
            [h+3, h+4, h+5],
            # Sides
            [h, h+1, h+3], [h+1, h+4, h+3],
            [h+1, h+2, h+4], [h+2, h+5, h+4],
            [h+2, h, h+5], [h, h+3, h+5],
        ])
        
        # Convert to numpy arrays
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        # Create the mesh object
        base_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                base_mesh.vectors[i][j] = vertices[face[j]]
        
        # Save to file
        base_mesh.save(output_path)
        print(f"  Saved: {output_path}")
    
    def generate_sign(self, text: str, distance: str, output_path: str):
        """
        Generate a directional sign plate with text.
        
        Args:
            text: Location name to display
            distance: Distance text to display
            output_path: Path to save the STL file
        """
        # TODO: Implement sign plate generation with embossed/debossed text
        print(f"Generating sign for '{text}' ({distance})...")
        print(f"  Output: {output_path}")
    
    def generate_arrow(self, output_path: str):
        """
        Generate an arrow pointer for the sign.
        
        Args:
            output_path: Path to save the STL file
        """
        # TODO: Implement arrow generation
        print(f"Generating arrow pointer...")
        print(f"  Output: {output_path}")
