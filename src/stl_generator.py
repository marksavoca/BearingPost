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
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

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
                 sign_spacing: float = 36.0,
                 sign_clearance: float = 2.0,
                 max_sign_length: float = 200.0,
                 min_font_size: float = 10.0,
                 max_font_size: float = 20.0,
                 text_height: float = 1.0):
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
        Creates a pointed sign (arrow end) with text embossed on it.
        Sign height is flat_height - clearance to fit within the flat indent.
        
        Args:
            text: Location name to display
            distance: Distance text to display
            output_path: Path to save the STL file
        """
        print(f"Generating sign for '{text}' ({distance})...")
        
        # Calculate sign dimensions
        sign_height = self.flat_height - (2 * self.sign_clearance)
        
        # Create the basic sign shape parameters
        point_length = sign_height * 0.5  # Point extends half the sign height
        
        # Start with maximum font size for main text
        font_size = min(self.max_font_size, sign_height * 0.8)
        
        # Calculate text widths
        # Uppercase text is wider, use better estimate: ~0.6 * font_size per character
        main_text_len = len(text.upper())
        
        # Distance will be stacked: number over "MI"
        # Extract just the number from distance (e.g., "214 mi" -> "214")
        dist_number = distance.split()[0] if distance else ""
        dist_text_len = len(dist_number)  # Just the number
        
        # Distance font is 50% of main
        dist_font_size = font_size * 0.5
        
        # Minimum readable sizes
        min_main_font = 12.0  # Main text must be at least 12mm
        min_dist_font = 6.0   # Distance must be at least 6mm
        min_mi_font = 7.0     # MI text minimum (FreeType seems to fail below ~6.5mm)
        
        # Padding: main text 5% from left, distance 3% from right (closer to point)
        left_padding_pct = 0.05
        right_padding_pct = 0.03
        min_gap = 5.0  # Minimum gap between texts
        
        # Try different strategies to fit text
        # Strategy 1: Try at max sign length with max font size
        sign_length = self.max_sign_length
        body_length = sign_length - point_length
        
        main_text_width = main_text_len * font_size * 0.6
        # Distance is now just the number (narrower than "214 mi")
        dist_text_width = max(dist_text_len * dist_font_size * 0.6, 2 * dist_font_size * 0.6) if distance else 0  # At least as wide as "MI"
        
        left_padding = body_length * left_padding_pct
        right_padding = body_length * right_padding_pct
        available_width = body_length - left_padding - right_padding
        needed_width = main_text_width + min_gap + dist_text_width
        
        if needed_width <= available_width:
            # Fits perfectly at max font size!
            # If both texts fit comfortably, reduce sign length to actual needed width
            # Add padding back to get actual body length needed
            optimal_body_length = needed_width + left_padding + right_padding
            # Add some extra margin (10mm) for aesthetics
            optimal_body_length += 10.0
            
            # Only reduce if significantly shorter (save at least 20mm)
            if optimal_body_length < body_length - 20.0:
                body_length = optimal_body_length
                sign_length = body_length + point_length
                print(f"  Note: Reduced sign length to {sign_length:.1f}mm to fit text")
        else:
            # Strategy 2: Reduce font size proportionally to fit within max length
            scale_factor = available_width / needed_width
            font_size = font_size * scale_factor
            dist_font_size = font_size * 0.5
            
            # Check if we're below minimum readable size
            # MI text will be 80% of distance font, so ensure it won't drop below 3mm
            mi_font_check = dist_font_size * 0.8
            if font_size < min_main_font or (distance and (dist_font_size < min_dist_font or mi_font_check < min_mi_font)):
                # Strategy 3: Drop distance text and maximize main text
                font_size = min(self.max_font_size, sign_height * 0.8)
                main_text_width = main_text_len * font_size * 0.6
                
                if main_text_width + left_padding + right_padding <= body_length:
                    # Main text fits at max size without distance
                    distance = None  # Drop distance
                    print(f"  Note: Distance text omitted to maintain readable main text size")
                else:
                    # Even main text needs to be smaller
                    scale_factor = (body_length - left_padding - right_padding) / main_text_width
                    font_size = font_size * scale_factor
                    font_size = max(min_main_font, font_size)
                    distance = None  # Drop distance
                    print(f"  Note: Distance text omitted and main text reduced to fit")
                
                dist_font_size = 0
        
        # Final calculations
        main_text_width = main_text_len * font_size * 0.6
        dist_text_width = max(dist_text_len * dist_font_size * 0.6, 2 * dist_font_size * 0.6) if distance else 0
        
        # Clamp font size
        font_size = max(self.min_font_size, min(font_size, self.max_font_size))
        if distance:
            dist_font_size = font_size * 0.5
        
        # Minimum practical length
        if sign_length < 60:
            sign_length = 60
        
        # Final body_length calculation
        body_length = sign_length - point_length
        
        print(f"  Sign dimensions: {sign_length:.1f}mm long × {sign_height:.1f}mm tall × {self.sign_thickness:.1f}mm thick")
        print(f"  Font size: {font_size:.1f}mm" + (f" (distance: {dist_font_size:.1f}mm)" if distance else ""))
        
        # Create the basic sign shape (pointed on one end, square on the other)
        # The pointed end will aim toward the location
        
        vertices = []
        
        # Square end (attaches to post) at X=0
        # Bottom left
        vertices.append([0, 0, 0])
        # Bottom right
        vertices.append([0, sign_height, 0])
        # Top right
        vertices.append([0, sign_height, self.sign_thickness])
        # Top left
        vertices.append([0, 0, self.sign_thickness])
        
        # Body rectangle extends to near the end
        body_length = sign_length - point_length
        # Bottom left
        vertices.append([body_length, 0, 0])
        # Bottom right
        vertices.append([body_length, sign_height, 0])
        # Top right
        vertices.append([body_length, sign_height, self.sign_thickness])
        # Top left
        vertices.append([body_length, 0, self.sign_thickness])
        
        # Pointed end - tip at center height
        tip_y = sign_height / 2  # Point at center
        # Tip vertex (shared by all point faces)
        vertices.append([sign_length, tip_y, 0])  # Bottom tip (index 8)
        vertices.append([sign_length, tip_y, self.sign_thickness])  # Top tip (index 9)
        
        # Define faces
        faces = []
        
        # Square end face (vertices 0,1,2,3)
        faces.extend([[0, 2, 1], [0, 3, 2]])
        
        # Body rectangular faces
        # Bottom face of rectangle
        faces.extend([[0, 1, 5], [0, 5, 4]])
        # Top face of rectangle
        faces.extend([[3, 6, 2], [3, 7, 6]])
        # Left side of rectangle
        faces.extend([[0, 4, 7], [0, 7, 3]])
        # Right side of rectangle
        faces.extend([[1, 2, 6], [1, 6, 5]])
        
        # Triangular point - 4 faces connecting rectangle end to tip
        # Bottom face: left-bottom (4), right-bottom (5), tip-bottom (8)
        faces.append([4, 5, 8])
        # Top face: left-top (7), tip-top (9), right-top (6)
        faces.append([7, 9, 6])
        # Left face: left-bottom (4), tip-bottom (8), tip-top (9), left-top (7)
        faces.extend([[4, 8, 9], [4, 9, 7]])
        # Right face: right-bottom (5), right-top (6), tip-top (9), tip-bottom (8)
        faces.extend([[5, 6, 9], [5, 9, 8]])
        
        # Create base sign mesh using trimesh for easier text operations
        vertices_array = np.array(vertices)
        faces_array = np.array(faces)
        
        sign_base = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
        
        # Add embossed text using vector-based rendering
        if not FREETYPE_AVAILABLE:
            print(f"  Warning: freetype-py not installed - text embossing unavailable")
            print(f"  Install with: pip install freetype-py shapely")
            sign_mesh = sign_base
        else:
            try:
                print(f"  Creating high-quality vector text...")
                
                # Create main text mesh
                # Position: left-justified (near square end), vertically centered
                # Add extra offset to account for font baseline and any descenders
                text_x = body_length * 0.05  # 5% from left edge (square end)
                text_y = (sign_height / 2) - (font_size / 2.8)  # Adjusted for baseline offset
                text_z = self.sign_thickness
                
                text_mesh = self._create_text_mesh_vector(text, font_size, (text_x, text_y, text_z))
                
                # Create distance text if provided (stacked: number over "MI")
                if distance:
                    dist_font_size = font_size * 0.5
                    
                    # Check if distance font size is too small
                    # Also check that MI font size (80% of distance) won't be too small
                    mi_font_size = dist_font_size * 0.8
                    if dist_font_size < 3.0 or mi_font_size < 3.0:
                        print(f"  Warning: Distance text too small (number: {dist_font_size:.1f}mm, MI: {mi_font_size:.1f}mm), skipping distance")
                        sign_mesh = trimesh.util.concatenate([sign_base, text_mesh])
                        print(f"  Text embossed: '{text}' (distance omitted - too small)")
                    else:
                        try:
                            # Extract just the number from distance
                            dist_number = distance.split()[0]
                            
                            print(f"  Debug: Rendering distance number '{dist_number}' at {dist_font_size:.1f}mm")
                            
                            # Create number text (top line)
                            dist_num_width = len(dist_number) * dist_font_size * 0.6
                            dist_x = body_length * 0.97 - max(dist_num_width, 2 * dist_font_size * 0.6)
                            
                            # Position number in upper part of sign (adjusted for baseline)
                            dist_num_y = (sign_height / 2) + (dist_font_size * 0.2)  # Upper half, baseline adjusted
                            dist_num_z = self.sign_thickness
                            
                            distance_num_mesh = self._create_text_mesh_vector(dist_number, dist_font_size, (dist_x, dist_num_y, dist_num_z))
                            
                            print(f"  Debug: Rendering 'MI' at {mi_font_size:.1f}mm")
                            
                            # Create "MI" text (bottom line) - smaller
                            mi_width = 2 * mi_font_size * 0.6
                            mi_x = dist_x + (max(dist_num_width, mi_width) - mi_width) / 2  # Center "MI" under number
                            mi_y = (sign_height / 2) - (mi_font_size * 1.1)  # Lower half, baseline adjusted
                            mi_z = self.sign_thickness
                            
                            distance_mi_mesh = self._create_text_mesh_vector("MI", mi_font_size, (mi_x, mi_y, mi_z))
                            
                            # Combine all meshes
                            sign_mesh = trimesh.util.concatenate([sign_base, text_mesh, distance_num_mesh, distance_mi_mesh])
                            print(f"  Text embossed: '{text}' + '{distance}' (stacked)")
                        except Exception as dist_error:
                            print(f"  Warning: Distance text failed ({dist_error}), using main text only")
                            sign_mesh = trimesh.util.concatenate([sign_base, text_mesh])
                            print(f"  Text embossed: '{text}' (distance failed)")
                else:
                    sign_mesh = trimesh.util.concatenate([sign_base, text_mesh])
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
