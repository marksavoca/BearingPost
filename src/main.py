"""
Direction Sign Generator
Creates STL files for 3D-printable directional signs with geographic accuracy.
"""

import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from geo_utils import haversine_distance, calculate_bearing, format_distance
from stl_generator import DirectionSignGenerator
import os


@dataclass
class Location:
    """Represents a location with name, coordinates, and optional distance."""
    name: str
    latitude: float
    longitude: float
    distance_km: float = None  # Optional: can be calculated or provided
    bearing: float = None  # Bearing in degrees from home (0-360)
    font: str = "Arial"  # Font for the sign text
    color: str = "blue"  # Color for the sign


# Home location (reference point where the sign will sit)
HOME = Location(
    name="711 Green Lane",
    latitude=39.73059345761767,
    longitude=-75.16805997015658
)

# Destinations to point to
LOCATIONS = [
    Location("Albany", 42.697749760384546, -73.96640153872819, font="Arial", color="blue"),
    Location("Wharton", 40.89469275367825, -74.57707989642749, font="Arial", color="green"),
    Location("Stone Harbor", 39.05421572551508, -74.75888740116326, font="Arial", color="teal"),
    Location("Rome", 41.9028, 12.4964, font="Arial", color="red"),
    Location("Boston", 42.33813538280124, -71.09011637177542, font="Arial", color="maroon"),
    Location("Philadelphia", 39.93617180632561, -75.16416042623342, font="Arial", color="orange"),
    Location("Los Angeles", 34.06965322948626, -118.44078368287897, font="Arial", color="purple"),
]


def main():
    """Main entry point for generating direction sign STLs."""
    print("Direction Sign Generator")
    print("=" * 70)
    print(f"Home Location: {HOME.name}")
    print(f"  Coordinates: {HOME.latitude}, {HOME.longitude}")
    print("\n" + "=" * 70)
    print("Calculating bearings and distances...\n")
    
    # Calculate distance and bearing for each location
    for loc in LOCATIONS:
        loc.distance_km = haversine_distance(
            HOME.latitude, HOME.longitude,
            loc.latitude, loc.longitude
        )
        loc.bearing = calculate_bearing(
            HOME.latitude, HOME.longitude,
            loc.latitude, loc.longitude
        )
    
    # Display results
    print(f"{'Location':<35} {'Distance':<20} {'Bearing':<15} {'Font':<10} {'Color':<10}")
    print("=" * 70)
    
    for loc in LOCATIONS:
        distance_str = format_distance(loc.distance_km, units='mi')
        bearing_str = f"{loc.bearing:.1f}°"
        print(f"{loc.name:<35} {distance_str:<20} {bearing_str:<15} {loc.font:<10} {loc.color:<10}")
    
    print("\n" + "=" * 70)
    print("\nGenerating STL files...")
    print("=" * 70)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the STL generator
    generator = DirectionSignGenerator()
    
    # Generate the post with flat surfaces at each bearing
    # The first (taller) post includes the base with north arrow
    # Adjust bearings > 180° to keep signs on same side of post (front hemisphere)
    # Bearings > 180° are in the rear, so subtract 180° to mirror them to the front
    post_bearings = [loc.bearing if loc.bearing <= 180 else loc.bearing - 180 for loc in LOCATIONS]
    print(f"\nOriginal bearings: {[f'{loc.bearing:.1f}' for loc in LOCATIONS]}")
    print(f"Adjusted bearings: {[f'{b:.1f}' for b in post_bearings]}")
    post_path = os.path.join(output_dir, "post.stl")
    generator.generate_post(post_bearings, post_path)
    
    # Generate individual sign plates for each location
    print("\nGenerating sign plates...")
    for i, loc in enumerate(LOCATIONS):
        sign_filename = f"sign_{i+1}_{loc.name.replace(' ', '_').replace(',', '')}.stl"
        sign_path = os.path.join(output_dir, sign_filename)
        distance_str = format_distance(loc.distance_km, units='mi')
        # Pass bearing to determine sign direction
        generator.generate_sign(loc.name, distance_str, sign_path, loc.bearing)
    
    print("\n" + "=" * 70)
    print("\nGeneration complete!")
    print(f"Files saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - post.stl (two-piece post with flat indents, first post includes base with north arrow)")
    print(f"  - sign_*.stl ({len(LOCATIONS)} sign plates)")
    print("\nNext steps:")
    print("1. 3D print the components")
    print("2. Attach signs to the post at the flat surfaces")
    print("3. Assemble the directional sign")


if __name__ == "__main__":
    main()
