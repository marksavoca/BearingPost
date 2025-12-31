"""
STL generation module for creating 3D models of the direction sign.
"""

from typing import List, Tuple
import numpy as np
from stl import mesh
import math
import os
from datetime import datetime
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
                 post_height: float = 180.0,
                 post_radius: float = 10.0,
                 base_radius: float = 50.0,
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
                 text_ramp_height: float = 0.4,
                 text_ramp_scale: float = 1.15,
                 base_text_font_size: float = 4.5,
                 base_text_height: float = 0.5,
                 base_text_gap: float = 2.0,
                 base_text_radius_factor: float = 0.7,
                 base_text_rotation_deg: float = 90.0,
                 base_segments: int = 128,
                 base_chamfer: float = 2.0,
                 index_pin_radius: float = 1.0,
                 index_pin_length: float = 2.0,
                 index_pin_clearance: float = 0.2,
                 index_pin_inset: float = 5.0,
                 id_pin_radius: float = 0.7,
                 id_pin_length: float = 1.5,
                 id_pin_spacing: float = 3.0,
                 id_pin_clearance: float = 0.2,
                 id_pin_inset: float = 2.0,
                 sign_vertical_spacing: float = 8.0,
                 magnet_diameter: float = 6.0,
                 magnet_thickness: float = 2.0,
                 magnet_clearance: float = 0.2,
                 peg_clearance: float = 0.1,
                 join_pin_radius: float = 1.2,
                 join_pin_length: float = 4.0,
                 join_pin_clearance: float = 0.2,
                 debug: bool = False):
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
            text_ramp_height: Height of the text ramp blend (mm)
            text_ramp_scale: XY scale at the text base for ramp blending
            base_text_font_size: Font size for base coordinates text (mm)
            base_text_height: Emboss height for base coordinates (mm)
            base_text_gap: Gap between latitude and longitude text (mm)
            base_text_radius_factor: Base radius factor for south-side text placement
            base_text_rotation_deg: Z rotation for base text orientation
            base_segments: Number of segments for base cylinders (smoothness)
            base_chamfer: Size of the top chamfer on the base (mm)
            index_pin_radius: Radius of indexing pins on post flats (mm)
            index_pin_length: Length of indexing pins from flat surface (mm)
            index_pin_clearance: Radial clearance for sign pin hole (mm)
            index_pin_inset: Inset from square end for sign pin hole (mm)
            id_pin_radius: Radius of ID pins on post flats (mm)
            id_pin_length: Length of ID pins from flat surface (mm)
            id_pin_spacing: Spacing between ID pins on the flat (mm)
            id_pin_clearance: Radial clearance for ID pin holes (mm)
            id_pin_inset: Inset from square end for ID pin holes (mm)
            sign_vertical_spacing: Vertical spacing between signs (mm)
            magnet_diameter: Diameter of alignment magnets (mm)
            magnet_thickness: Thickness of alignment magnets (mm)
            magnet_clearance: Radial clearance for magnet pockets (mm)
            peg_clearance: Radial clearance for peg/socket and key slot (mm)
            join_pin_radius: Radius of post-to-post alignment pins (mm)
            join_pin_length: Length of post-to-post alignment pins (mm)
            join_pin_clearance: Radial clearance for post-to-post pin holes (mm)
            debug: Enable verbose debug output
        """
        self.debug = debug
        self._print = print if self.debug else (lambda *args, **kwargs: None)
        self.boolean_overlap = 0.1
        self._warned_no_boolean_engine = False
        self.post_height = min(post_height, 180.0)
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
        self.text_ramp_height = text_ramp_height
        self.text_ramp_scale = text_ramp_scale
        self.base_text_font_size = base_text_font_size
        self.base_text_height = base_text_height
        self.base_text_gap = base_text_gap
        self.base_text_radius_factor = base_text_radius_factor
        self.base_text_rotation_deg = base_text_rotation_deg
        self.base_segments = base_segments
        self.base_chamfer = base_chamfer
        self.index_pin_radius = index_pin_radius
        self.index_pin_length = index_pin_length
        self.index_pin_clearance = index_pin_clearance
        self.index_pin_inset = index_pin_inset
        self.id_pin_radius = id_pin_radius
        self.id_pin_length = id_pin_length
        self.id_pin_spacing = id_pin_spacing
        self.id_pin_clearance = id_pin_clearance
        self.id_pin_inset = id_pin_inset
        self.sign_vertical_spacing = sign_vertical_spacing
        self.magnet_diameter = magnet_diameter
        self.magnet_thickness = magnet_thickness
        self.magnet_clearance = magnet_clearance
        self.peg_clearance = peg_clearance
        self.join_pin_radius = join_pin_radius
        self.join_pin_length = join_pin_length
        self.join_pin_clearance = join_pin_clearance

    def _get_boolean_engine(self) -> str | None:
        available = getattr(trimesh.boolean, "engines_available", set())
        if "manifold" in available:
            return "manifold"
        raise RuntimeError("manifold boolean engine not available. Install manifold3d.")

    def _prepare_mesh_for_boolean(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Return a cleaned mesh for boolean operations."""
        cleaned = mesh.copy()
        try:
            cleaned.remove_degenerate_faces()
            cleaned.remove_duplicate_faces()
            cleaned.remove_unreferenced_vertices()
            cleaned.merge_vertices()
            cleaned.process(validate=True)
        except Exception:
            pass
        return cleaned

    def _rotate_mesh_z(self, target_mesh: trimesh.Trimesh, degrees: float,
                       center: Tuple[float, float, float]) -> None:
        """Rotate a mesh around the Z axis in-place."""
        rotation_matrix = trimesh.transformations.rotation_matrix(
            math.radians(degrees), [0, 0, 1], center
        )
        target_mesh.apply_transform(rotation_matrix)

    def _create_chamfered_base_mesh(self) -> List[trimesh.Trimesh]:
        """Create base meshes with a top chamfer."""
        chamfer = max(0.0, min(self.base_chamfer, self.base_height))
        if chamfer <= 0.0:
            base = trimesh.creation.cylinder(
                radius=self.base_radius,
                height=self.base_height,
                sections=self.base_segments
            )
            base.apply_translation([0, 0, self.base_height / 2])
            return [base]

        bottom_height = self.base_height - chamfer
        bottom = trimesh.creation.cylinder(
            radius=self.base_radius,
            height=bottom_height,
            sections=self.base_segments
        )
        bottom.apply_translation([0, 0, bottom_height / 2])

        frustum = trimesh.creation.cylinder(
            radius=self.base_radius,
            height=chamfer,
            sections=self.base_segments
        )
        z_min = -chamfer / 2
        z_max = chamfer / 2
        top_radius = max(self.base_radius - chamfer, 0.1)
        top_scale = top_radius / self.base_radius
        vertices = frustum.vertices.copy()
        for i in range(len(vertices)):
            z = vertices[i][2]
            t = (z - z_min) / (z_max - z_min)
            scale = 1.0 + (top_scale - 1.0) * t
            vertices[i][0] *= scale
            vertices[i][1] *= scale
        frustum.vertices = vertices
        frustum.apply_translation([0, 0, bottom_height + chamfer / 2])

        return [bottom, frustum]

    def _center_mesh_xy(self, target_mesh: trimesh.Trimesh) -> None:
        """Center a mesh in the XY plane, preserving Z."""
        bounds = target_mesh.bounds
        center_x = (bounds[0][0] + bounds[1][0]) / 2
        center_y = (bounds[0][1] + bounds[1][1]) / 2
        target_mesh.apply_translation([-center_x, -center_y, 0])

    def _apply_text_ramp(self, text_mesh: trimesh.Trimesh, base_z: float) -> None:
        """Apply a shallow XY ramp at the base of a text mesh."""
        ramp_height = min(self.text_ramp_height, self.text_height)
        if ramp_height <= 0 or self.text_ramp_scale <= 1.0:
            return
        bounds = text_mesh.bounds
        center_x = (bounds[0][0] + bounds[1][0]) / 2
        center_y = (bounds[0][1] + bounds[1][1]) / 2
        vertices = text_mesh.vertices.copy()
        z_min = base_z
        z_max = base_z + ramp_height
        for i in range(len(vertices)):
            z = vertices[i][2]
            if z <= z_min:
                scale = self.text_ramp_scale
            elif z >= z_max:
                scale = 1.0
            else:
                t = (z - z_min) / (z_max - z_min)
                scale = self.text_ramp_scale - (self.text_ramp_scale - 1.0) * t
            vertices[i][0] = center_x + (vertices[i][0] - center_x) * scale
            vertices[i][1] = center_y + (vertices[i][1] - center_y) * scale
        text_mesh.vertices = vertices

    def _union_meshes(self, meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
        """Boolean-union meshes into a single solid; fall back to concat on failure."""
        if not meshes:
            raise ValueError("No meshes to union")
        engine = self._get_boolean_engine()
        if engine is None and not self._warned_no_boolean_engine:
            self._print("  Warning: No boolean engine available; meshes may remain separate shells")
            self._warned_no_boolean_engine = True
        if engine is None:
            return trimesh.util.concatenate(meshes)
        try:
            cleaned_meshes = [self._prepare_mesh_for_boolean(m) for m in meshes]
            unioned = trimesh.boolean.union(
                cleaned_meshes,
                engine=engine,
                use_exact=False,
                check_volume=False,
                debug=self.debug
            )
            if unioned is not None and len(unioned.faces) > 0:
                return unioned
            self._print("  Warning: Boolean union returned empty mesh; retrying exact mode")
            unioned = trimesh.boolean.union(
                meshes,
                engine=engine,
                use_exact=True,
                use_self=True,
                check_volume=False,
                debug=self.debug
            )
            if unioned is not None and len(unioned.faces) > 0:
                return unioned
            self._print("  Warning: Boolean union returned empty mesh; falling back to concat")
        except Exception as e:
            self._print(f"  Warning: Boolean union failed: {e}")
        return trimesh.util.concatenate(meshes)

    def _log_components(self, mesh: trimesh.Trimesh, label: str) -> None:
        """Log connected component count for debugging union results."""
        if not self.debug:
            return
        try:
            parts = mesh.split(only_watertight=False)
            self._print(f"    Components ({label}): {len(parts)}")
        except Exception as e:
            self._print(f"    Warning: Could not compute components for {label}: {e}")

    def _create_base_bottom_text_mesh(self, text_lines: List[str],
                                      engraving_depth: float) -> trimesh.Trimesh:
        """Create mirrored multiline text mesh for bottom engraving."""
        if not text_lines:
            raise ValueError("No text lines provided for engraving")
        font_size = 6.0
        line_gap = font_size * 0.3
        line_height = font_size + line_gap
        total_height = line_height * len(text_lines) - line_gap
        start_y = total_height / 2 - font_size
        meshes = []
        for i, line in enumerate(text_lines):
            y_pos = start_y - i * line_height
            line_mesh = self._create_text_mesh_vector(line, font_size, (0, 0, 0))
            bounds = line_mesh.bounds
            center_x = (bounds[0][0] + bounds[1][0]) / 2
            line_mesh.apply_translation([-center_x, y_pos, 0])
            meshes.append(line_mesh)
        text_mesh = trimesh.util.concatenate(meshes)
        # Mirror so the text reads correctly from the bottom.
        text_mesh.apply_scale([1, -1, 1])
        # Scale the extrude height to the engraving depth (keep bottom at Z=0).
        z_scale = engraving_depth / max(self.text_height, 0.01)
        text_mesh.apply_scale([1, 1, z_scale])
        return text_mesh

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
        # Place pin so it protrudes from the flat surface and overlaps the post.
        radial_center = self.post_radius - self.flat_depth + (self.index_pin_length / 2) - self.boolean_overlap
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

    def _create_id_pins_at_bearing(self, bearing: float, sign_height: float,
                                   post_x_offset: float, post_y_offset: float,
                                   segment_id: int) -> trimesh.Trimesh:
        """Create up to 4 ID pins (binary) on the flat spot for a segment ID (1-15)."""
        if segment_id <= 0 or segment_id > 15:
            raise ValueError(f"segment_id must be 1-15, got {segment_id}")
        pin_offsets = [
            -1.5 * self.id_pin_spacing,
            -0.5 * self.id_pin_spacing,
            0.5 * self.id_pin_spacing,
            1.5 * self.id_pin_spacing,
        ]
        pin_meshes = []
        for bit_index, x_offset in enumerate(pin_offsets):
            if not (segment_id & (1 << bit_index)):
                continue
            pin = trimesh.creation.cylinder(
                radius=self.id_pin_radius,
                height=self.id_pin_length,
                sections=24
            )
            pin.apply_transform(trimesh.transformations.rotation_matrix(
                math.radians(90), [1, 0, 0]
            ))
            radial_center = self.post_radius - self.flat_depth + (self.id_pin_length / 2) - self.boolean_overlap
            pin.apply_translation([0, radial_center, sign_height + x_offset])
            rotation_matrix = trimesh.transformations.rotation_matrix(
                math.radians(-bearing), [0, 0, 1], [0, 0, 0]
            )
            pin.apply_transform(rotation_matrix)
            pin.apply_translation([post_x_offset, post_y_offset, 0])
            pin_meshes.append(pin)
        return trimesh.util.concatenate(pin_meshes)

    def _create_id_holes_for_sign(self, sign_length: float, sign_height: float,
                                  point_left: bool, segment_id: int) -> List[trimesh.Trimesh]:
        """Create matching ID pin holes on the sign backside."""
        if segment_id <= 0 or segment_id > 15:
            raise ValueError(f"segment_id must be 1-15, got {segment_id}")
        hole_radius = self.id_pin_radius + self.id_pin_clearance
        hole_depth = min(self.sign_thickness, self.id_pin_length + self.id_pin_clearance)
        pin_offsets = [
            -1.5 * self.id_pin_spacing,
            -0.5 * self.id_pin_spacing,
            0.5 * self.id_pin_spacing,
            1.5 * self.id_pin_spacing,
        ]
        x_base = sign_length / 2
        holes = []
        for bit_index, x_offset in enumerate(pin_offsets):
            if not (segment_id & (1 << bit_index)):
                continue
            hole = trimesh.creation.cylinder(
                radius=hole_radius,
                height=hole_depth,
                sections=24
            )
            x_pos = x_base
            y_pos = sign_height / 2 + x_offset
            hole.apply_translation([x_pos, y_pos, hole_depth / 2])
            holes.append(hole)
        return holes

    def _split_distance_text(self, distance_text: str) -> Tuple[str, str]:
        """Split distance into value and units for two-line display."""
        if not distance_text:
            return "", ""
        parts = distance_text.strip().split()
        if len(parts) >= 2:
            return " ".join(parts[:-1]), parts[-1]
        return distance_text.strip(), ""
    
    def generate_post(self, bearings: List, output_path: str, home_lat: float = None, home_lon: float = None):
        """
        Generate a two-part post with flats for signs, splitting the signs across the
        two post parts from top to bottom. The first sign is at the very top of the
        upper post; remaining signs continue down the upper post, then restart at
        the top of the lower post. The two post parts include alignment pins for glue-up.
        
        Args:
            bearings: List of bearings (in degrees) where signs will attach, ordered top to bottom
            output_path: Path to save the STL file
            home_lat: Home latitude to emboss on base (optional)
            home_lon: Home longitude to emboss on base (optional)
        """
        self._print(f"Generating two-part post with {len(bearings)} sign slots...")
        
        # Configuration
        segments = 64
        sign_vertical_spacing = self.sign_vertical_spacing
        sign_gap_half = sign_vertical_spacing / 2
        segment_height = self.flat_height + sign_vertical_spacing
        
        output_base = os.path.splitext(output_path)[0]

        entries = list(bearings)

        def max_slots_per_post(post_height: float) -> int:
            usable = post_height - self.sign_vertical_spacing - self.flat_height
            if usable <= 0:
                return 1
            return max(1, int(math.floor(usable / segment_height) + 1))

        upper_capacity = max_slots_per_post(self.post_height)
        split_index = min(len(entries), upper_capacity)
        upper_entries = entries[:split_index]
        lower_entries = entries[split_index:]

        def slot_centers(post_height: float, count: int) -> List[float]:
            if count <= 0:
                return []
            top_center = post_height - sign_gap_half - (self.flat_height / 2)
            return [top_center - i * segment_height for i in range(count)]

        def build_post(entries: List, post_height: float, base_index: int,
                       add_join_pins: bool, cut_join_holes: bool) -> trimesh.Trimesh:
            post_mesh = trimesh.creation.cylinder(
                radius=self.post_radius,
                height=post_height,
                sections=segments
            )
            post_mesh.apply_translation([0, 0, post_height / 2])
            add_meshes = []
            centers = slot_centers(post_height, len(entries))
            for i, (entry, sign_center) in enumerate(zip(entries, centers)):
                if isinstance(entry, dict):
                    bearing = entry.get("bearing")
                    segment_id = entry.get("segment_id")
                    is_spacer = entry.get("spacer", False)
                else:
                    bearing = entry
                    segment_id = base_index + i + 1
                    is_spacer = False
                label = f"{segment_id}" if segment_id is not None else "spacer"
                self._print(f"    Slot {i+1}: bearing {bearing} (ID {label})")
                if is_spacer or bearing is None:
                    continue
                adjusted_bearing = (bearing + 90.0) % 360.0
                box_mesh = self._create_box_mesh_at_bearing(adjusted_bearing, sign_center, 0, 0)
                try:
                    new_mesh = post_mesh.difference(box_mesh)
                    if new_mesh is not None and len(new_mesh.faces) > 0:
                        post_mesh = new_mesh
                    else:
                        self._print("      Warning: Flat boolean returned empty mesh")
                except Exception as e:
                    self._print(f"      Warning: Flat boolean failed: {e}")

                if segment_id is not None and segment_id <= 15:
                    id_pin_mesh = self._create_id_pins_at_bearing(
                        adjusted_bearing, sign_center, 0, 0, segment_id
                    )
                    add_meshes.append(id_pin_mesh)
                else:
                    if segment_id is not None and segment_id > 15:
                        self._print(f"      Note: segment_id {segment_id} exceeds 15; using center pin only")
                    center_pin_mesh = self._create_index_pin_at_bearing(adjusted_bearing, sign_center, 0, 0)
                    add_meshes.append(center_pin_mesh)

            if cut_join_holes:
                join_holes = self._create_post_join_pin_holes()
                try:
                    new_mesh = post_mesh.difference(join_holes)
                    if new_mesh is not None and len(new_mesh.faces) > 0:
                        post_mesh = new_mesh
                    else:
                        self._print("      Warning: Join pin holes returned empty mesh")
                except Exception as e:
                    self._print(f"      Warning: Join pin holes failed: {e}")

            if add_join_pins:
                add_meshes.append(self._create_post_join_pins(post_height))

            if add_meshes:
                post_mesh = self._union_meshes([post_mesh] + add_meshes)
            return post_mesh

        post_height = max(self.post_height, segment_height)
        upper_height = post_height
        lower_height = post_height

        # ===== LOWER POST (BASE + POST) =====
        self._print("  Creating lower post...")
        base_meshes = self._create_chamfered_base_mesh()
        base_mesh = trimesh.util.concatenate(base_meshes)

        # Engrave maker text on the bottom of the base.
        if FREETYPE_AVAILABLE:
            year = datetime.now().year
            maker_lines = ["Mark W Savoca", f"© {year}"]
            try:
                engraving_depth = 0.6
                text_mesh = self._create_base_bottom_text_mesh(maker_lines, engraving_depth)
                new_mesh = base_mesh.difference(text_mesh)
                if new_mesh is not None and len(new_mesh.faces) > 0:
                    base_mesh = new_mesh
            except Exception:
                pass

        arrow_mesh = self._create_north_arrow()
        coords_meshes = []
        if home_lat is not None and home_lon is not None:
            coords_meshes = self._create_coordinates_text(home_lat, home_lon)
        compass_meshes = self._create_compass_decorations()

        lower_post = build_post(lower_entries, lower_height, split_index, add_join_pins=True, cut_join_holes=False)
        lower_post.apply_translation([0, 0, self.base_height])

        lower_meshes = [base_mesh, arrow_mesh, lower_post]
        if compass_meshes:
            lower_meshes.extend(compass_meshes)
        if coords_meshes:
            lower_meshes.extend(coords_meshes)
        lower_segment = self._union_meshes(lower_meshes)
        self._log_components(lower_segment, "lower post")
        lower_path = f"{output_base}_post_lower.stl"
        lower_segment.export(lower_path)
        self._print(f"  Saved: {lower_path}")

        # ===== UPPER POST =====
        self._print("  Creating upper post...")
        upper_post = build_post(upper_entries, upper_height, 0, add_join_pins=False, cut_join_holes=True)
        self._log_components(upper_post, "upper post")
        upper_path = f"{output_base}_post_upper.stl"
        upper_post.export(upper_path)
        self._print(f"  Saved: {upper_path}")
    
    def _create_north_arrow(self) -> trimesh.Trimesh:
        """
        Create a north indicator mesh ("N") to sit on top of the base.
        The letter is oriented toward +Y to indicate north (bearing 0°).
        
        Returns:
            trimesh.Trimesh: North indicator mesh
        """
        if FREETYPE_AVAILABLE:
            font_size = min(self.arrow_length * 0.9, self.base_radius * 0.3)
            z_base = self.base_height - self.boolean_overlap
            letter_mesh = self._create_text_mesh_vector("N", font_size, (0, 0, z_base), apply_ramp=True)
            self._center_mesh_xy(letter_mesh)
            bounds = letter_mesh.bounds
            letter_height = bounds[1][1] - bounds[0][1]
            letter_width = bounds[1][0] - bounds[0][0]
        else:
            letter_thickness = 2.0  # Height above base
            letter_height = min(self.arrow_length * 0.6, self.base_radius * 0.25)
            letter_width = letter_height * 0.6
            stroke = max(letter_width * 0.22, 1.0)
            
            # Build a blocky "N" in the XY plane, then extrude in Z.
            z_center = self.base_height - self.boolean_overlap + letter_thickness / 2
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
        
        # Position on the base top at north (+Y).
        base_y = self.base_radius * 0.7
        max_y = self.base_radius * 0.85
        if base_y + letter_height > max_y:
            base_y = max_y - letter_height
        if base_y < 0:
            base_y = 0
        base_x = 0.0
        letter_mesh.apply_translation([base_x, base_y, 0])
        
        return letter_mesh

    def _create_compass_decorations(self) -> List[trimesh.Trimesh]:
        """Create E/S/W letters and compass ticks/ring."""
        meshes = []
        # Letters
        if FREETYPE_AVAILABLE:
            z_base = self.base_height - self.boolean_overlap
            north_font = min(self.arrow_length * 0.9, self.base_radius * 0.3)
            other_font = north_font * 0.85
            letter_radius = self.base_radius * 0.7
            for letter, bearing_deg, size in [
                ("E", 90, other_font),
                ("S", 180, other_font),
                ("W", 270, other_font),
            ]:
                angle = math.radians(bearing_deg)
                x = math.sin(angle) * letter_radius
                y = math.cos(angle) * letter_radius
                letter_mesh = self._create_text_mesh_vector(letter, size, (0, 0, z_base), apply_ramp=True)
                self._center_mesh_xy(letter_mesh)
                letter_mesh.apply_translation([x, y, 0])
                meshes.append(letter_mesh)
        
        # Ring
        ring_height = 0.6
        outer_radius = self.base_radius * 0.9
        inner_radius = self.base_radius * 0.85
        try:
            outer = trimesh.creation.cylinder(radius=outer_radius, height=ring_height, sections=self.base_segments)
            inner = trimesh.creation.cylinder(radius=inner_radius, height=ring_height, sections=self.base_segments)
            z_center = self.base_height - self.boolean_overlap + ring_height / 2
            outer.apply_translation([0, 0, z_center])
            inner.apply_translation([0, 0, z_center])
            ring = outer.difference(inner)
            if ring is not None and len(ring.faces) > 0:
                meshes.append(ring)
        except Exception:
            pass
        
        # Ticks
        tick_height = 0.6
        tick_width = 0.8
        tick_length_small = 2.0
        tick_length_med = 3.0
        tick_length_large = 4.0
        tick_radius = self.base_radius * 0.92
        for deg in range(0, 360, 10):
            angle = math.radians(-deg)
            if deg % 90 == 0:
                tick_length = tick_length_large
            elif deg % 45 == 0:
                tick_length = tick_length_med
            else:
                tick_length = tick_length_small
            tick = trimesh.creation.box(extents=[tick_width, tick_length, tick_height])
            z_center = self.base_height - self.boolean_overlap + tick_height / 2
            tick.apply_translation([0, tick_radius - tick_length / 2, z_center])
            tick.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            meshes.append(tick)
        
        return meshes
    
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
            self._print(f"  Adding coordinates text to base (south side)...")
            # Format coordinates to 4 decimal places
            lat_text = f"{latitude:.4f}"
            lon_text = f"{longitude:.4f}"
            # Font size for coordinates (small, readable)
            font_size = self.base_text_font_size
            text_height = self.base_text_height
            base_z = self.base_height - self.boolean_overlap
            # Both texts on south side (-Y), centered horizontally
            y_pos = -self.base_radius * self.base_text_radius_factor
            gap = self.base_text_gap
            lat_width = len(lat_text) * font_size * 0.6
            lon_width = len(lon_text) * font_size * 0.6
            total_width = lat_width + gap + lon_width
            start_x = -total_width / 2
            lat_x = start_x
            lon_x = start_x + lat_width + gap
            lat_mesh = self._create_text_mesh_vector(lat_text, font_size, (lat_x, y_pos, base_z), apply_ramp=True)
            lon_mesh = self._create_text_mesh_vector(lon_text, font_size, (lon_x, y_pos, base_z), apply_ramp=True)

            # Rotate both texts another 90° to align with the intended facing direction.
            self._rotate_mesh_z(lat_mesh, self.base_text_rotation_deg, (0, 0, base_z))
            self._rotate_mesh_z(lon_mesh, self.base_text_rotation_deg, (0, 0, base_z))

            self._print(f"  Coordinates embossed: {lat_text}, {lon_text}")
            return [lat_mesh, lon_mesh]
        except Exception as e:
            self._print(f"  Warning: Could not create coordinates text: {e}")
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
        join_max_height = max(0.0, (self.sign_vertical_spacing / 2) - self.sign_clearance)
        peg_height = min(8.0, join_max_height)
        key_width = self.post_radius * 0.3  # 30% of post radius (2.4mm for 8mm post)
        key_depth = self.post_radius * 0.1  # 10% of post radius (shallower key)
        
        # Create main cylindrical peg (overlap slightly with the post for union).
        z_base = post_height - self.boolean_overlap
        peg = trimesh.creation.cylinder(
            radius=peg_radius,
            height=peg_height,
            sections=32
        )
        peg.apply_translation([0, 0, z_base + peg_height / 2])
        
        # Create alignment key (rectangular protrusion at 0° / +X reference)
        key_box = trimesh.creation.box(
            extents=[key_width, key_depth * 2, peg_height]
        )
        # Position key at south side (-Y) for alignment reference
        key_box.apply_translation([0, -(peg_radius + key_depth - self.boolean_overlap), z_base + peg_height / 2])
        
        peg_mesh = self._union_meshes([peg, key_box])
        
        # Add magnet pocket centered on top of peg
        magnet_radius = (self.magnet_diameter / 2) + self.magnet_clearance
        magnet_depth = min(self.magnet_thickness + self.magnet_clearance, peg_height)
        magnet = trimesh.creation.cylinder(
            radius=magnet_radius,
            height=magnet_depth,
            sections=32
        )
        magnet.apply_translation([0, 0, z_base + peg_height - magnet_depth / 2])
        try:
            new_mesh = peg_mesh.difference(magnet)
            if new_mesh is not None and len(new_mesh.faces) > 0:
                peg_mesh = new_mesh
        except Exception:
            pass
        
        return peg_mesh
    
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
        socket_radius = peg_radius + self.peg_clearance
        join_max_height = max(0.0, (self.sign_vertical_spacing / 2) - self.sign_clearance)
        socket_depth = min(8.5, join_max_height)
        key_width = self.post_radius * 0.3 + self.peg_clearance
        key_depth = self.post_radius * 0.1 + self.peg_clearance
        
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
        # Position slot at south side (-Y) to match the peg key orientation
        key_slot.apply_translation([
            post_x_offset,
            post_y_offset - (peg_radius + key_depth),
            socket_depth / 2
        ])
        
        return trimesh.util.concatenate([socket, key_slot])

    def _create_socket_magnet_cutter(self, post_x_offset: float,
                                     post_y_offset: float) -> trimesh.Trimesh:
        """Create a magnet cutter that recesses upward from the socket ceiling."""
        join_max_height = max(0.0, (self.sign_vertical_spacing / 2) - self.sign_clearance)
        socket_depth = min(8.5, join_max_height)
        magnet_radius = (self.magnet_diameter / 2) + self.magnet_clearance
        magnet_depth = min(self.magnet_thickness + self.magnet_clearance, socket_depth)
        magnet = trimesh.creation.cylinder(
            radius=magnet_radius,
            height=magnet_depth,
            sections=32
        )
        magnet.apply_translation([
            post_x_offset,
            post_y_offset,
            socket_depth + (magnet_depth / 2)
        ])
        return magnet

    def _get_post_join_pin_offsets(self) -> List[Tuple[float, float]]:
        """Return two asymmetric XY offsets for post join pins."""
        primary = self.post_radius * 0.45
        secondary = self.post_radius * 0.28
        return [
            (primary, 0.0),
            (-secondary, -primary * 0.6),
        ]

    def _create_post_join_pins(self, post_height: float) -> trimesh.Trimesh:
        """Create alignment pins on the top of a post segment for glue-up."""
        z_base = post_height - self.boolean_overlap
        pins = []
        for x, y in self._get_post_join_pin_offsets():
            pin = trimesh.creation.cylinder(
                radius=self.join_pin_radius,
                height=self.join_pin_length,
                sections=24
            )
            pin.apply_translation([x, y, z_base + self.join_pin_length / 2])
            pins.append(pin)
        return trimesh.util.concatenate(pins)

    def _create_post_join_pin_holes(self) -> trimesh.Trimesh:
        """Create matching holes for post join pins."""
        hole_radius = self.join_pin_radius + self.join_pin_clearance
        hole_depth = self.join_pin_length + self.join_pin_clearance
        holes = []
        for x, y in self._get_post_join_pin_offsets():
            hole = trimesh.creation.cylinder(
                radius=hole_radius,
                height=hole_depth,
                sections=24
            )
            hole.apply_translation([x, y, hole_depth / 2])
            holes.append(hole)
        return trimesh.util.concatenate(holes)
    
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
    
    def _create_text_mesh_vector(self, text: str, font_size: float, position: Tuple[float, float, float],
                                 apply_ramp: bool = False) -> trimesh.Trimesh:
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
                        self._print(f"      Warning: Could not extrude polygon (area={p.area:.2f}): {e}")
                        pass
        
        if not meshes:
            raise ValueError(f"Failed to create 3D mesh for text: {text}")
        
        # Combine all character meshes
        result = trimesh.util.concatenate(meshes)
        
        # Position the text
        result.apply_translation([position[0], position[1], position[2]])
        if apply_ramp:
            self._apply_text_ramp(result, position[2])
        
        return result
    
    def generate_sign(self, text: str, distance: str, output_path: str, bearing: float = 0.0,
                      segment_id: int | None = None, arrowed: bool = True):
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
        point_left = bearing > 180.0 if arrowed else False
        direction_note = " (pointing left)" if point_left else " (pointing right)"
        self._print(f"Generating sign for '{text}'{direction_note}...")
        
        # Calculate sign dimensions
        sign_height = self.flat_height - (2 * self.sign_clearance)
        
        # Create the basic sign shape parameters
        point_length = sign_height * 0.5 if arrowed else 0.0
        
        # Use maximum font size for main text (adjusted later to fit distance text)
        font_size = min(self.max_font_size, sign_height * 0.8)
        distance_value, distance_units = self._split_distance_text(distance)
        has_distance = bool(distance_value)
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

        distance_width = compute_distance_width() if has_distance else 0.0
        
        # Minimum readable size
        min_main_font = 12.0  # Main text must be at least 12mm
        
        # Layout paddings and spacing
        attach_padding = 10.0
        left_attach_padding = attach_padding + 4.0
        attach_pad = left_attach_padding if point_left else attach_padding
        tip_padding = 3.0
        base_text_gap = 16.0
        text_gap = max(10.0, base_text_gap - max(0, main_text_len - 6) * 1.2)
        effective_gap = text_gap if has_distance else 0.0
        
        # Try to fit text at max sign length
        sign_length = self.max_sign_length
        body_length = sign_length - point_length

        def compute_required_body_length() -> float:
            return attach_pad + main_text_width + effective_gap + distance_width + tip_padding

        required_body_length = compute_required_body_length()
        self._print(
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
                self._print(f"  Note: Reduced sign length to {sign_length:.1f}mm to fit text")
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
                distance_width = compute_distance_width() if has_distance else 0.0
                required_body_length = compute_required_body_length()
            self._print(
                f"  Layout after sizing: name={main_text_width:.1f}mm, "
                f"distance={distance_width:.1f}mm, "
                f"required_body={required_body_length:.1f}mm, "
                f"body_length={body_length:.1f}mm, "
                f"font={font_size:.1f}mm, dist_font={distance_font_size:.1f}mm"
            )
            if required_body_length > body_length:
                self._print(f"  Warning: Text may overlap; name text at minimum size")
        
        # Final calculations
        distance_font_size = min(distance_font_size, font_size * 0.65)
        units_font_size = max(min_distance_font_size, distance_font_size * 0.85)
        distance_width = compute_distance_width() if has_distance else 0.0
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
            self._print(f"  Note: Reduced sign length to {sign_length:.1f}mm to fit text")
        
        self._print(f"  Sign dimensions: {sign_length:.1f}mm long × {sign_height:.1f}mm tall × {self.sign_thickness:.1f}mm thick")
        self._print(f"  Font size: {font_size:.1f}mm")
        self._print(f"  Distance font: {distance_font_size:.1f}mm")
        
        # Create the basic sign shape (pointed on one end, square on the other)
        # The pointed end will aim toward the location
        
        vertices = []
        
        if arrowed:
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
        else:
            sign_base = trimesh.creation.box(
                extents=[sign_length, sign_height, self.sign_thickness]
            )
            sign_base.apply_translation([
                sign_length / 2,
                sign_height / 2,
                self.sign_thickness / 2
            ])
            vertices = sign_base.vertices.tolist()
            faces = sign_base.faces.tolist()
        
        # Create base sign mesh using trimesh for easier text operations
        vertices_array = np.array(vertices)
        faces_array = np.array(faces)
        
        sign_base = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
        
        # Add indexing hole on the backside (for post pin alignment).
        try:
            if segment_id is not None and segment_id <= 15:
                hole_meshes = self._create_id_holes_for_sign(
                    sign_length, sign_height, point_left, segment_id
                )
                hole_mesh = trimesh.util.concatenate(hole_meshes)
            else:
                if segment_id is not None and segment_id > 15:
                    self._print(f"  Note: segment_id {segment_id} exceeds 15; using center hole only")
                hole_mesh = self._create_index_hole_for_sign(sign_length, sign_height, point_left)
            new_mesh = sign_base.difference(hole_mesh)
            if new_mesh is not None and len(new_mesh.faces) > 0:
                sign_base = new_mesh
            else:
                self._print(f"  Warning: Index hole boolean returned empty mesh")
        except Exception as e:
            self._print(f"  Warning: Index hole boolean operation failed: {e}")
        
        # Add embossed text using vector-based rendering
        if not FREETYPE_AVAILABLE:
            self._print(f"  Warning: freetype-py not installed - text embossing unavailable")
            self._print(f"  Install with: pip install freetype-py shapely")
            sign_mesh = sign_base
        else:
            try:
                self._print(f"  Creating high-quality vector text...")
                
                # Create main text mesh
                # Position: near square end (attachment point), vertically centered
                if point_left:
                    text_x = sign_length - attach_pad - main_text_width
                else:
                    text_x = attach_pad
                text_y = (sign_height / 2) - (font_size / 2.8)  # Adjusted for baseline offset
                text_z = self.sign_thickness - self.boolean_overlap
                
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
                sign_mesh = self._union_meshes([sign_base, text_mesh] + distance_meshes)
                self._print(f"  Text embossed: '{text}'")
                
            except Exception as e:
                import traceback
                self._print(f"  Warning: Could not create vector text: {e}")
                self._print(f"  Details: {traceback.format_exc()}")
                self._print(f"  Saving blank sign")
                sign_mesh = sign_base
        
        # Export
        self._log_components(sign_mesh, f"sign '{text}'")
        sign_mesh.export(output_path)
        self._print(f"  Saved: {output_path}")
    
    def generate_arrow(self, output_path: str):
        """
        Generate an arrow pointer for the sign.
        
        Args:
            output_path: Path to save the STL file
        """
        # TODO: Implement arrow generation
        self._print(f"Generating arrow pointer...")
        self._print(f"  Output: {output_path}")
