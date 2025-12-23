"""
STL generation module for creating 3D models of the direction sign.
"""

from typing import List, Tuple
import numpy as np
from stl import mesh
import math
import trimesh

# Try to import optional text rendering libraries
try:
    import freetype
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    FREETYPE_AVAILABLE = True
except ImportError:
    FREETYPE_AVAILABLE = False


class DirectionSignGenerator:
    """Generates 3D models for direction signs."""
    
    def __init__(self, 
                 post_height: float = 200.0,
                 post_radius: float = 8.0,
                 base_radius: float = 40.0,
                 base_height: float = 10.0,
                 sign_width: float = 100.0,
                 sign_height: float = 20.0,
                 sign_thickness: float = 3.0,
                 flat_depth: float = 3.0,
                 flat_height: float = 28.0,
                 arrow_length: float = 15.0,
                 arrow_width: float = 8.0,
                 sign_spacing: float = 36.0,
                 sign_clearance: float = 0.5,
                 max_sign_length: float = 200.0,
                 min_font_size: float = 10.0,
                 max_font_size: float = 20.0,
                 text_height: float = 1.0,
                 base_text_font_size: float = 4.5,
                 base_text_height: float = 0.5,
                 base_text_gap: float = 2.0,
                 base_text_radius_factor: float = 0.7,
                 base_text_rotation_deg: float = 90.0,
                 index_pin_radius: float = 1.0,
                 index_pin_length: float = 2.0,
                 index_pin_clearance: float = 0.2,
                 index_pin_inset: float = 5.0):
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
            sign_clearance: Vertical clearance between sign and flat edges
            max_sign_length: Maximum length of a sign plate
            min_font_size: Minimum font size for text
            max_font_size: Maximum font size for text
            text_height: Height of embossed text (mm)
            base_text_font_size: Font size for base coordinates text (mm)
            base_text_height: Emboss height for base coordinates (mm)
            base_text_gap: Gap between latitude and longitude text (mm)
            base_text_radius_factor: Base radius factor for south-side text placement
            base_text_rotation_deg: Z rotation for base text orientation
            index_pin_radius: Radius of indexing pins on post flats (mm)
            index_pin_length: Length of indexing pins from flat surface (mm)
            index_pin_clearance: Radial clearance for sign pin hole (mm)
            index_pin_inset: Inset from square end for sign pin hole (mm)
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
        self.sign_clearance = sign_clearance
        self.max_sign_length = max_sign_length
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.text_height = text_height
        self.base_text_font_size = base_text_font_size
        self.base_text_height = base_text_height
        self.base_text_gap = base_text_gap
        self.base_text_radius_factor = base_text_radius_factor
        self.base_text_rotation_deg = base_text_rotation_deg
        self.index_pin_radius = index_pin_radius
        self.index_pin_length = index_pin_length
        self.index_pin_clearance = index_pin_clearance
        self.index_pin_inset = index_pin_inset

    def _rotate_mesh_z(self, target_mesh: trimesh.Trimesh, degrees: float,
                       center: Tuple[float, float, float]) -> None:
        """Rotate a mesh around the Z axis in-place."""
        rotation_matrix = trimesh.transformations.rotation_matrix(
            math.radians(degrees), [0, 0, 1], center
        )
        target_mesh.apply_transform(rotation_matrix)

    def _create_index_pin_at_bearing(self, bearing: float, sign_height: float,
                                     post_x_offset: float, post_y_offset: float) -> trimesh.Trimesh:
        """Create an indexing pin on the flat spot at a specific bearing."""
        pin = trimesh.creation.cylinder(
            radius=self.index_pin_radius,
            height=self.index_pin_length,
            sections=24
        )
        # Rotate cylinder axis from +Z to +Y for bearing 0.
        pin.apply_transform(trimesh.transformations.rotation_matrix(
            math.radians(90), [1, 0, 0]
        ))
        # Place pin so it protrudes from the flat surface.
        radial_center = self.post_radius - self.flat_depth + (self.index_pin_length / 2)
        pin.apply_translation([0, radial_center, sign_height])
        # Rotate around post center by bearing (match box subtraction orientation).
        rotation_matrix = trimesh.transformations.rotation_matrix(
            math.radians(-bearing), [0, 0, 1], [0, 0, 0]
        )
        pin.apply_transform(rotation_matrix)
        pin.apply_translation([post_x_offset, post_y_offset, 0])
        return pin

    def _create_index_hole_for_sign(self, sign_length: float, sign_height: float,
                                    point_left: bool) -> trimesh.Trimesh:
        """Create a matching indexing hole on the sign backside."""
        hole_radius = self.index_pin_radius + self.index_pin_clearance
        hole_depth = min(self.sign_thickness, self.index_pin_length + self.index_pin_clearance)
        hole = trimesh.creation.cylinder(
            radius=hole_radius,
            height=hole_depth,
            sections=24
        )
        x_pos = sign_length / 2
        y_pos = sign_height / 2
        hole.apply_translation([x_pos, y_pos, hole_depth / 2])
        return hole

    def _split_distance_text(self, distance_text: str) -> Tuple[str, str]:
        """Split distance into value and units for two-line display."""
        if not distance_text:
            return "", ""
        parts = distance_text.strip().split()
        if len(parts) >= 2:
            return " ".join(parts[:-1]), parts[-1]
        return distance_text.strip(), ""
    
    def generate_post(self, bearings: List[float], output_path: str, home_lat: float = None, home_lon: float = None):
        """
        Generate two post sections with flat indents at specified bearings.
        Creates a two-piece design to fit within 210mm print height.
        Uses boolean subtraction to create flat mounting surfaces.
        Signs are ordered from top to bottom matching the input bearings list.
        The first (taller) post includes the base with north arrow.
        
        Args:
            bearings: List of bearings (in degrees) where signs will attach, ordered top to bottom
            output_path: Path to save the STL file
            home_lat: Home latitude to emboss on base (optional)
            home_lon: Home longitude to emboss on base (optional)
        """
        print(f"Generating 2-piece post with {len(bearings)} flat indents...")
        
        all_meshes = []
        
        # Configuration
        segments = 64
        post_separation = 150.0  # Distance between the two posts when laid side by side
        max_post_height = 200.0  # Maximum height for each post section
        sign_vertical_spacing = 8.0
        base_sign_offset = 40.0
        top_padding = 8.0
        
        # Calculate sign distribution and heights
        # Account for base height in the first post
        base_height_offset = self.base_height
        bottom_start_height = base_height_offset + base_sign_offset
        sign_increment = self.flat_height + sign_vertical_spacing
        
        # Determine how many signs fit on bottom post (taller post)
        num_signs_bottom = 0
        current_height = bottom_start_height
        while current_height + self.flat_height <= max_post_height and num_signs_bottom < len(bearings):
            num_signs_bottom += 1
            current_height += sign_increment
        
        # Calculate actual bottom post height (last sign center + half flat height + padding)
        bottom_post_height = bottom_start_height + (num_signs_bottom - 1) * sign_increment + self.flat_height / 2 + top_padding
        
        # Remaining signs go on top post (shorter post)
        num_signs_top = len(bearings) - num_signs_bottom
        
        # Calculate top post height
        if num_signs_top > 0:
            # First sign BOTTOM should be at 8mm above Z=0
            # Since we position by center, center = bottom + flat_height/2
            top_start_height = sign_vertical_spacing + self.flat_height / 2
            # Post height goes from 0 to (last sign center + half flat height + padding)
            top_post_height = top_start_height + (num_signs_top - 1) * sign_increment + self.flat_height / 2 + top_padding
        else:
            top_post_height = 0
            top_start_height = 0
        
        print(f"  First post (taller, with base): {num_signs_bottom} signs, height: {bottom_post_height:.1f}mm (including {base_height_offset:.1f}mm base)")
        print(f"  Second post (shorter): {num_signs_top} signs, height: {top_post_height:.1f}mm")
        print(f"  Sign increment: {sign_increment:.1f}mm (flat_height {self.flat_height} + spacing 8mm)")
        
        # ===== CREATE FIRST POST (TALLER) WITH BASE =====
        print("  Creating first post with base...")
        
        # Create base
        base_mesh = trimesh.creation.cylinder(
            radius=self.base_radius,
            height=self.base_height,
            sections=64
        )
        base_mesh.apply_translation([0, 0, self.base_height / 2])
        
        # Create north arrow on top of base
        arrow_mesh = self._create_north_arrow()
        
        # Create coordinates text on base if provided
        coords_meshes = []
        if home_lat is not None and home_lon is not None:
            coords_meshes = self._create_coordinates_text(home_lat, home_lon)
        
        # Create post cylinder
        bottom_post_mesh = trimesh.creation.cylinder(
            radius=self.post_radius,
            height=bottom_post_height,
            sections=64
        )
        # Trimesh creates cylinder centered at origin, translate to sit on top of base
        bottom_post_mesh.apply_translation([0, 0, bottom_post_height / 2])
        
        # Subtract boxes for first post signs
        # First post gets the LAST num_signs_bottom locations, in reverse order (bottom to top)
        print("  Subtracting boxes from first post...")
        bottom_pin_meshes = []
        for i in range(num_signs_bottom):
            # Get bearing from end of list, working backwards
            bearing_index = len(bearings) - 1 - i
            bearing = bearings[bearing_index]
            sign_height = bottom_start_height + (i * sign_increment)
            
            print(f"    Box {i+1}: bearing {bearing:.1f}° (location #{bearing_index+1}), height {sign_height:.1f}mm")
            
            box_mesh = self._create_box_mesh_at_bearing(bearing, sign_height, 0, 0)
            try:
                new_mesh = bottom_post_mesh.difference(box_mesh)
                if new_mesh is not None and len(new_mesh.faces) > 0:
                    bottom_post_mesh = new_mesh
                else:
                    print(f"      Warning: Boolean operation returned empty mesh")
            except Exception as e:
                print(f"      Warning: Boolean operation failed: {e}")
            
            bottom_pin_meshes.append(
                self._create_index_pin_at_bearing(bearing, sign_height, 0, 0)
            )
        
        # Combine base, arrow, coordinates text, and post
        print("  Combining base, arrow, coordinates, and post...")
        
        # Add alignment peg on top of first post
        print("  Adding alignment peg to first post...")
        peg_mesh = self._create_alignment_peg(bottom_post_height)
        
        meshes_to_combine = [base_mesh, arrow_mesh, bottom_post_mesh, peg_mesh]
        if bottom_pin_meshes:
            meshes_to_combine.extend(bottom_pin_meshes)
        if coords_meshes:
            meshes_to_combine.extend(coords_meshes)
        
        combined_first_post = trimesh.util.concatenate(meshes_to_combine)
        all_meshes.append(combined_first_post)
        
        # ===== CREATE SECOND POST (SHORTER, offset to the side) =====
        if num_signs_top > 0:
            print("  Creating second post cylinder...")
            x_offset = post_separation
            
            top_post_mesh = trimesh.creation.cylinder(
                radius=self.post_radius,
                height=top_post_height,
                sections=64
            )
            # Translate to sit on Z=0 and offset to the side
            top_post_mesh.apply_translation([x_offset, 0, top_post_height / 2])
            
            # Subtract boxes for second post signs
            # Second post gets the FIRST num_signs_top locations, in reverse order (bottom to top)
            print("  Subtracting boxes from second post...")
            top_pin_meshes = []
            for i in range(num_signs_top):
                # Get bearing from beginning of list, working backwards
                bearing_index = num_signs_top - 1 - i
                bearing = bearings[bearing_index]
                sign_bottom = sign_vertical_spacing + (i * sign_increment)
                sign_center = sign_bottom + self.flat_height / 2
                
                print(f"    Box {i+1}: bearing {bearing:.1f}° (location #{bearing_index+1}), center height {sign_center:.1f}mm, post offset X={x_offset:.1f}")
                
                box_mesh = self._create_box_mesh_at_bearing(bearing, sign_center, x_offset, 0)
                try:
                    new_mesh = top_post_mesh.difference(box_mesh)
                    if new_mesh is not None and len(new_mesh.faces) > 0:
                        top_post_mesh = new_mesh
                    else:
                        print(f"      Warning: Boolean operation returned empty mesh")
                except Exception as e:
                    print(f"      Warning: Boolean operation failed: {e}")
                
                top_pin_meshes.append(
                    self._create_index_pin_at_bearing(bearing, sign_center, x_offset, 0)
                )
            
            # Add alignment socket to bottom of second post
            print("  Adding alignment socket to second post...")
            socket_mesh = self._create_alignment_socket(x_offset, 0)
            try:
                new_mesh = top_post_mesh.difference(socket_mesh)
                if new_mesh is not None and len(new_mesh.faces) > 0:
                    top_post_mesh = new_mesh
                else:
                    print(f"      Warning: Socket boolean operation returned empty mesh")
            except Exception as e:
                print(f"      Warning: Socket boolean operation failed: {e}")
            
            if top_pin_meshes:
                top_post_mesh = trimesh.util.concatenate([top_post_mesh, *top_pin_meshes])
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
    
    def _create_north_arrow(self) -> trimesh.Trimesh:
        """
        Create a north indicator mesh ("N") to sit on top of the base.
        The letter is oriented toward +Y to indicate north (bearing 0°).
        
        Returns:
            trimesh.Trimesh: North indicator mesh
        """
        letter_thickness = 2.0  # Height above base
        letter_height = min(self.arrow_length * 0.6, self.base_radius * 0.25)
        letter_width = letter_height * 0.6
        stroke = max(letter_width * 0.22, 1.0)
        
        # Build a blocky "N" in the XY plane, then extrude in Z.
        z_center = self.base_height + letter_thickness / 2
        y_center = letter_height / 2
        left_bar = trimesh.creation.box(extents=[stroke, letter_height, letter_thickness])
        left_bar.apply_translation([-letter_width / 2 + stroke / 2, y_center, z_center])
        
        right_bar = trimesh.creation.box(extents=[stroke, letter_height, letter_thickness])
        right_bar.apply_translation([letter_width / 2 - stroke / 2, y_center, z_center])
        
        diag_length = math.hypot(letter_height - stroke, letter_width - stroke)
        diag_bar = trimesh.creation.box(extents=[stroke, diag_length, letter_thickness])
        diag_angle = math.atan2(letter_width - stroke, letter_height - stroke)
        diag_bar.apply_transform(trimesh.transformations.rotation_matrix(diag_angle, [0, 0, 1]))
        diag_bar.apply_translation([0, y_center, z_center])
        
        letter_mesh = trimesh.util.concatenate([left_bar, right_bar, diag_bar])
        
        # Rotate another 90° so the letter orientation matches the coordinate text.
        self._rotate_mesh_z(letter_mesh, 90, (0, 0, z_center))
        
        # Position on the base top, nudged toward west (-X); Y offset is kept
        # at the midline to preserve current visual placement.
        base_y =  0
        max_y = self.base_radius * 0.85
        if base_y + letter_height > max_y:
            base_y = max_y - letter_height
        if base_y < 0:
            base_y = 0
        base_x = -self.base_radius * 0.5
        min_x = -self.base_radius * 0.85
        if base_x - (letter_width / 2) < min_x:
            base_x = min_x + (letter_width / 2)
        letter_mesh.apply_translation([base_x, base_y, 0])
        
        return letter_mesh
    
    def _create_coordinates_text(self, latitude: float, longitude: float) -> List[trimesh.Trimesh]:
        """
        Create embossed coordinate text on the base.
        Truncates to 4 decimal places for readability.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            List[trimesh.Trimesh]: List of text meshes for coordinates
        """
        if not FREETYPE_AVAILABLE:
            return []
        
        try:
            print(f"  Adding coordinates text to base (south side)...")
            # Format coordinates to 4 decimal places
            lat_text = f"{latitude:.4f}"
            lon_text = f"{longitude:.4f}"
            # Font size for coordinates (small, readable)
            font_size = self.base_text_font_size
            text_height = self.base_text_height
            base_z = self.base_height + text_height
            # Both texts on south side (-Y), centered horizontally
            y_pos = -self.base_radius * self.base_text_radius_factor
            gap = self.base_text_gap
            lat_width = len(lat_text) * font_size * 0.6
            lon_width = len(lon_text) * font_size * 0.6
            total_width = lat_width + gap + lon_width
            start_x = -total_width / 2
            lat_x = start_x
            lon_x = start_x + lat_width + gap
            lat_mesh = self._create_text_mesh_vector(lat_text, font_size, (lat_x, y_pos, base_z))
            lon_mesh = self._create_text_mesh_vector(lon_text, font_size, (lon_x, y_pos, base_z))

            # Rotate both texts another 90° to align with the intended facing direction.
            self._rotate_mesh_z(lat_mesh, self.base_text_rotation_deg, (0, 0, base_z))
            self._rotate_mesh_z(lon_mesh, self.base_text_rotation_deg, (0, 0, base_z))

            print(f"  Coordinates embossed: {lat_text}, {lon_text}")
            return [lat_mesh, lon_mesh]
        except Exception as e:
            print(f"  Warning: Could not create coordinates text: {e}")
            return []
    
    def _create_alignment_peg(self, post_height: float) -> trimesh.Trimesh:
        """
        Create an alignment peg for the top of the first post.
        Includes a keying feature to ensure correct rotational alignment.
        
        Args:
            post_height: Height of the post (peg sits on top)
            
        Returns:
            trimesh.Trimesh: Alignment peg mesh
        """
        # Peg dimensions - scale with post radius, ensure key stays within post diameter
        peg_radius = self.post_radius * 0.5  # 50% of post radius (4mm for 8mm post)
        peg_height = 8.0  # 8mm tall
        key_width = self.post_radius * 0.3  # 30% of post radius (2.4mm for 8mm post)
        key_depth = self.post_radius * 0.15  # 15% of post radius (1.2mm for 8mm post)
        
        # Create main cylindrical peg
        peg = trimesh.creation.cylinder(
            radius=peg_radius,
            height=peg_height,
            sections=32
        )
        peg.apply_translation([0, 0, post_height + peg_height / 2])
        
        # Create alignment key (rectangular protrusion at 0° / +X reference)
        key_box = trimesh.creation.box(
            extents=[key_width, key_depth * 2, peg_height]
        )
        # Position key at +X side of peg (0° reference for alignment keying)
        key_box.apply_translation([peg_radius + key_depth, 0, post_height + peg_height / 2])
        
        # Combine peg and key
        return trimesh.util.concatenate([peg, key_box])
    
    def _create_alignment_socket(self, post_x_offset: float, post_y_offset: float) -> trimesh.Trimesh:
        """
        Create an alignment socket for the bottom of the second post.
        Includes a keying slot to match the peg's alignment key.
        
        Args:
            post_x_offset: X offset of the post center
            post_y_offset: Y offset of the post center
            
        Returns:
            trimesh.Trimesh: Alignment socket mesh (for boolean subtraction)
        """
        # Socket dimensions (slightly larger than peg for clearance) - scale with post radius
        peg_radius = self.post_radius * 0.5
        socket_radius = peg_radius + 0.3   # 0.3mm radial clearance
        socket_depth = 8.5    # 0.5mm deeper than peg
        key_width = self.post_radius * 0.3 + 0.3       # 0.3mm clearance
        key_depth = self.post_radius * 0.15 + 0.3       # 0.3mm clearance
        
        # Create main cylindrical socket
        socket = trimesh.creation.cylinder(
            radius=socket_radius,
            height=socket_depth,
            sections=32
        )
        socket.apply_translation([post_x_offset, post_y_offset, socket_depth / 2])
        
        # Create alignment key slot (rectangular cutout at 0° / +X reference)
        key_slot = trimesh.creation.box(
            extents=[key_width, key_depth * 2, socket_depth]
        )
        # Position slot at +X side of socket (0° reference for alignment keying)
        key_slot.apply_translation([
            post_x_offset + peg_radius + key_depth,
            post_y_offset,
            socket_depth / 2
        ])
        
        # Combine socket and key slot
        return trimesh.util.concatenate([socket, key_slot])
    
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
        
        # Position the box at bearing 0 (north for bearings, +Y axis)
        # Box should be tangent to post surface (not cutting through center)
        distance_from_center = self.post_radius - self.flat_depth + box_depth / 2
        
        # First apply post offset, then position box relative to that post center
        box.apply_translation([post_x_offset, post_y_offset, 0])
        
        # Position box along +Y axis at bearing 0 (relative to post center)
        box.apply_translation([0, distance_from_center, sign_height])
        
        # Rotate around post center (at post_x_offset, post_y_offset) by bearing angle
        # IMPORTANT: Negate bearing because cylinder is viewed from bottom looking up
        angle_rad = math.radians(-bearing)
        # Create rotation matrix around the post center, not origin
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle_rad, [0, 0, 1], [post_x_offset, post_y_offset, 0]
        )
        box.apply_transform(rotation_matrix)
        
        return box
    
    def _create_text_mesh_vector(self, text: str, font_size: float, position: Tuple[float, float, float]) -> trimesh.Trimesh:
        """
        Create high-quality vector-based 3D text mesh using FreeType.
        
        Args:
            text: Text to render (will be converted to uppercase)
            font_size: Font size in mm
            position: (x, y, z) position for the text
            
        Returns:
            trimesh.Trimesh: 3D text mesh
        """
        if not FREETYPE_AVAILABLE:
            raise ImportError("freetype-py and shapely required for vector text")
        
        # Convert to uppercase to avoid descenders
        text = text.upper()
        
        # Font paths to try (prefer bold variants)
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # Try to load bold face from collection
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",  # Arial Bold on Windows
            "C:\\Windows\\Fonts\\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        face = None
        for font_path in font_paths:
            try:
                face = freetype.Face(font_path)
                # For TTC files (font collections), try to select a bold face
                if font_path.endswith('.ttc'):
                    # Try to find a bold face in the collection
                    for face_index in range(face.num_faces):
                        try:
                            test_face = freetype.Face(font_path, face_index)
                            face_name = test_face.family_name.decode('utf-8').lower() if hasattr(test_face.family_name, 'decode') else str(test_face.family_name).lower()
                            style_name = test_face.style_name.decode('utf-8').lower() if hasattr(test_face.style_name, 'decode') else str(test_face.style_name).lower()
                            if 'bold' in face_name or 'bold' in style_name:
                                face = test_face
                                break
                        except:
                            continue
                break
            except:
                continue
        
        if face is None:
            raise RuntimeError("Could not load any system font")
        
        # Set font size (FreeType uses 1/64th of a point)
        # Minimum font size to avoid division by zero errors
        if font_size < 3.0:
            raise ValueError(f"Font size {font_size} too small (minimum 3.0mm)")
        
        face.set_char_size(int(font_size * 64))
        
        # Collect all character polygons
        all_polygons = []
        pen_x = 0
        
        for char in text:
            face.load_char(char, freetype.FT_LOAD_NO_BITMAP)
            glyph = face.glyph
            outline = glyph.outline
            
            if len(outline.points) > 0:
                # Convert outline points
                points = [(pt[0] / 64.0 + pen_x, pt[1] / 64.0) for pt in outline.points]
                
                # Process contours - need to handle holes properly
                start = 0
                char_polygons = []
                for end in outline.contours:
                    contour_points = points[start:end+1]
                    # Need at least 3 points for a valid polygon
                    if len(contour_points) >= 3:
                        try:
                            # Close the contour if needed
                            if contour_points[0] != contour_points[-1]:
                                contour_points.append(contour_points[0])
                            
                            poly = Polygon(contour_points)
                            if poly.is_valid and poly.area > 1e-9:  # Filter out degenerate polygons
                                char_polygons.append(poly)
                        except Exception as e:
                            pass
                    start = end + 1
                
                # Separate exterior and holes based on area/orientation
                if char_polygons:
                    # The largest polygon is typically the exterior
                    char_polygons.sort(key=lambda p: abs(p.area), reverse=True)
                    if char_polygons:
                        exterior = char_polygons[0]
                        
                        # For simple characters (like comma, period), just use the exterior
                        # For complex characters, identify and subtract holes
                        if len(char_polygons) > 1:
                            holes = [p for p in char_polygons[1:] if p.within(exterior)]
                            
                            # Create polygon with holes
                            if holes:
                                try:
                                    # Subtract holes from exterior
                                    result = exterior
                                    for hole in holes:
                                        result = result.difference(hole)
                                    all_polygons.append(result)
                                except:
                                    all_polygons.append(exterior)
                            else:
                                all_polygons.append(exterior)
                        else:
                            # Single contour - just add it
                            all_polygons.append(exterior)
            
            # Advance pen position
            pen_x += glyph.advance.x / 64.0
        
        if not all_polygons:
            raise ValueError(f"No valid geometry generated for text: {text}")
        
        # Convert to trimesh
        meshes = []
        for poly in all_polygons:
            # Handle both Polygon and MultiPolygon
            if isinstance(poly, MultiPolygon):
                poly_list = list(poly.geoms)
            else:
                poly_list = [poly]
            
            for p in poly_list:
                if p.is_valid and not p.is_empty and p.area > 1e-6:  # Skip tiny polygons
                    try:
                        # Extrude the 2D polygon to 3D
                        text_mesh = trimesh.creation.extrude_polygon(p, height=self.text_height)
                        if text_mesh is not None and len(text_mesh.vertices) > 0:
                            meshes.append(text_mesh)
                    except Exception as e:
                        print(f"      Warning: Could not extrude polygon (area={p.area:.2f}): {e}")
                        pass
        
        if not meshes:
            raise ValueError(f"Failed to create 3D mesh for text: {text}")
        
        # Combine all character meshes
        result = trimesh.util.concatenate(meshes)
        
        # Position the text
        result.apply_translation([position[0], position[1], position[2]])
        
        return result
    
    def generate_sign(self, text: str, distance: str, output_path: str, bearing: float = 0.0):
        """
        Generate a directional sign plate with text.
        Creates a pointed sign (arrow end) with text embossed on it.
        Sign height is flat_height - clearance to fit within the flat indent.
        If bearing > 180°, sign points left to keep all signs on the same side of post.
        
        Args:
            text: Location name to display
            distance: Distance text to display
            output_path: Path to save the STL file
            bearing: Bearing angle in degrees (0-360), used to determine sign orientation
        """
        # Determine if sign should point left (bearing > 180°)
        point_left = bearing > 180.0
        direction_note = " (pointing left)" if point_left else " (pointing right)"
        print(f"Generating sign for '{text}'{direction_note}...")
        
        # Calculate sign dimensions
        sign_height = self.flat_height - (2 * self.sign_clearance)
        
        # Create the basic sign shape parameters
        point_length = sign_height * 0.5  # Point extends half the sign height
        
        # Use maximum font size for main text (adjusted later to fit distance text)
        font_size = min(self.max_font_size, sign_height * 0.8)
        distance_value, distance_units = self._split_distance_text(distance)
        distance_font_size = min(self.max_font_size * 0.5, sign_height * 0.38)
        distance_font_size = min(distance_font_size, font_size * 0.65)
        min_distance_font_size = 5.0
        distance_font_size = max(min_distance_font_size, distance_font_size)
        units_font_size = max(min_distance_font_size, distance_font_size * 0.85)
        
        # Calculate text width (layout estimate)
        # Uppercase text is wider; add margin to reduce overlap risk.
        name_width_factor = 0.65
        distance_width_factor = 0.6
        distance_width_margin = 2.0
        main_text_len = len(text.upper())
        main_text_width = main_text_len * font_size * name_width_factor
        def compute_distance_width() -> float:
            value_width = len(distance_value) * distance_font_size * distance_width_factor
            units_width = len(distance_units) * units_font_size * distance_width_factor
            return max(value_width, units_width) + distance_width_margin

        distance_width = compute_distance_width()
        
        # Minimum readable size
        min_main_font = 12.0  # Main text must be at least 12mm
        
        # Layout paddings and spacing
        attach_padding = 10.0
        tip_padding = 3.0
        base_text_gap = 16.0
        text_gap = max(10.0, base_text_gap - max(0, main_text_len - 6) * 1.2)
        
        # Try to fit text at max sign length
        sign_length = self.max_sign_length
        body_length = sign_length - point_length

        def compute_required_body_length() -> float:
            return attach_padding + main_text_width + text_gap + distance_width + tip_padding

        required_body_length = compute_required_body_length()
        print(
            f"  Layout widths: name={main_text_width:.1f}mm, "
            f"distance={distance_width:.1f}mm, "
            f"required_body={required_body_length:.1f}mm, "
            f"body_length={body_length:.1f}mm"
        )
        if required_body_length < body_length:
            optimal_length = required_body_length + point_length
            if optimal_length < sign_length:
                sign_length = max(60.0, optimal_length)
                body_length = sign_length - point_length
                print(f"  Note: Reduced sign length to {sign_length:.1f}mm to fit text")
        elif required_body_length > body_length:
            # Reduce font size(s) until both texts fit within max length
            while required_body_length > body_length and (font_size > min_main_font or distance_font_size > min_distance_font_size):
                if font_size > min_main_font:
                    font_size = max(min_main_font, font_size - 0.5)
                else:
                    distance_font_size = max(min_distance_font_size, distance_font_size - 0.5)
                    units_font_size = max(min_distance_font_size, distance_font_size * 0.85)
                distance_font_size = min(distance_font_size, font_size * 0.65)
                units_font_size = max(min_distance_font_size, distance_font_size * 0.85)
                main_text_width = main_text_len * font_size * name_width_factor
                distance_width = compute_distance_width()
                required_body_length = compute_required_body_length()
            print(
                f"  Layout after sizing: name={main_text_width:.1f}mm, "
                f"distance={distance_width:.1f}mm, "
                f"required_body={required_body_length:.1f}mm, "
                f"body_length={body_length:.1f}mm, "
                f"font={font_size:.1f}mm, dist_font={distance_font_size:.1f}mm"
            )
            if required_body_length > body_length:
                print(f"  Warning: Text may overlap; name text at minimum size")
        
        # Final calculations
        distance_font_size = min(distance_font_size, font_size * 0.65)
        units_font_size = max(min_distance_font_size, distance_font_size * 0.85)
        distance_width = compute_distance_width()
        main_text_width = main_text_len * font_size * name_width_factor
        
        # Clamp font size
        font_size = max(self.min_font_size, min(font_size, self.max_font_size))
        
        # Minimum practical length
        if sign_length < 60:
            sign_length = 60
        
        # Final body_length calculation
        body_length = sign_length - point_length
        
        # After font adjustments, shrink again if there is extra space.
        required_body_length = compute_required_body_length()
        if required_body_length < body_length:
            sign_length = max(60.0, required_body_length + point_length)
            body_length = sign_length - point_length
            print(f"  Note: Reduced sign length to {sign_length:.1f}mm to fit text")
        
        print(f"  Sign dimensions: {sign_length:.1f}mm long × {sign_height:.1f}mm tall × {self.sign_thickness:.1f}mm thick")
        print(f"  Font size: {font_size:.1f}mm")
        print(f"  Distance font: {distance_font_size:.1f}mm")
        
        # Create the basic sign shape (pointed on one end, square on the other)
        # The pointed end will aim toward the location
        
        vertices = []
        
        # Build right-pointing geometry (square end at X=0, pointed end at X=sign_length)
        vertices.append([0, 0, 0])
        vertices.append([0, sign_height, 0])
        vertices.append([0, sign_height, self.sign_thickness])
        vertices.append([0, 0, self.sign_thickness])
        
        body_length = sign_length - point_length
        vertices.append([body_length, 0, 0])
        vertices.append([body_length, sign_height, 0])
        vertices.append([body_length, sign_height, self.sign_thickness])
        vertices.append([body_length, 0, self.sign_thickness])
        
        tip_y = sign_height / 2
        vertices.append([sign_length, tip_y, 0])
        vertices.append([sign_length, tip_y, self.sign_thickness])
        
        # Define faces for right-pointing geometry
        faces = []
        faces.extend([[0, 2, 1], [0, 3, 2]])
        faces.extend([[0, 1, 5], [0, 5, 4]])
        faces.extend([[3, 6, 2], [3, 7, 6]])
        faces.extend([[0, 4, 7], [0, 7, 3]])
        faces.extend([[1, 2, 6], [1, 6, 5]])
        faces.append([4, 5, 8])
        faces.append([7, 9, 6])
        faces.extend([[4, 8, 9], [4, 9, 7]])
        faces.extend([[5, 6, 9], [5, 9, 8]])
        
        if point_left:
            vertices = [[sign_length - v[0], v[1], v[2]] for v in vertices]
            faces = [[f[2], f[1], f[0]] for f in faces]
        
        # Create base sign mesh using trimesh for easier text operations
        vertices_array = np.array(vertices)
        faces_array = np.array(faces)
        
        sign_base = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
        
        # Add indexing hole on the backside (for post pin alignment).
        try:
            hole_mesh = self._create_index_hole_for_sign(sign_length, sign_height, point_left)
            new_mesh = sign_base.difference(hole_mesh)
            if new_mesh is not None and len(new_mesh.faces) > 0:
                sign_base = new_mesh
            else:
                print(f"  Warning: Index hole boolean returned empty mesh")
        except Exception as e:
            print(f"  Warning: Index hole boolean operation failed: {e}")
        
        # Add embossed text using vector-based rendering
        if not FREETYPE_AVAILABLE:
            print(f"  Warning: freetype-py not installed - text embossing unavailable")
            print(f"  Install with: pip install freetype-py shapely")
            sign_mesh = sign_base
        else:
            try:
                print(f"  Creating high-quality vector text...")
                
                # Create main text mesh
                # Position: near square end (attachment point), vertically centered
                if point_left:
                    text_x = sign_length - attach_padding - main_text_width
                else:
                    text_x = attach_padding
                text_y = (sign_height / 2) - (font_size / 2.8)  # Adjusted for baseline offset
                text_z = self.sign_thickness
                
                text_mesh = self._create_text_mesh_vector(text, font_size, (text_x, text_y, text_z))
                
                # Create distance text meshes near the arrow end
                distance_meshes = []
                if distance_value:
                    if point_left:
                        distance_x = point_length + tip_padding
                    else:
                        distance_x = body_length - tip_padding - distance_width
                    row_offset = distance_font_size * 0.8
                    top_center = (sign_height / 2) + (row_offset / 2)
                    bottom_center = (sign_height / 2) - (row_offset / 2)
                    dist_y = top_center - (distance_font_size / 2.8)
                    units_y = bottom_center - (units_font_size / 2.8) - (distance_font_size * 0.2)
                    distance_meshes.append(
                        self._create_text_mesh_vector(distance_value, distance_font_size, (distance_x, dist_y, text_z))
                    )
                    if distance_units:
                        value_width = len(distance_value) * distance_font_size * distance_width_factor
                        units_width = len(distance_units) * units_font_size * distance_width_factor
                        units_x = distance_x + (value_width - units_width) / 2
                        distance_meshes.append(
                            self._create_text_mesh_vector(distance_units, units_font_size, (units_x, units_y, text_z))
                        )
                
                # Combine base and text
                meshes = [sign_base, text_mesh]
                if distance_meshes:
                    meshes.extend(distance_meshes)
                sign_mesh = trimesh.util.concatenate(meshes)
                print(f"  Text embossed: '{text}'")
                
            except Exception as e:
                import traceback
                print(f"  Warning: Could not create vector text: {e}")
                print(f"  Details: {traceback.format_exc()}")
                print(f"  Saving blank sign")
                sign_mesh = sign_base
        
        # Export
        sign_mesh.export(output_path)
        print(f"  Saved: {output_path}")
    
    def generate_arrow(self, output_path: str):
        """
        Generate an arrow pointer for the sign.
        
        Args:
            output_path: Path to save the STL file
        """
        # TODO: Implement arrow generation
        print(f"Generating arrow pointer...")
        print(f"  Output: {output_path}")
