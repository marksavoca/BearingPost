"""
Direction Sign Generator
Creates STL files for 3D-printable directional signs with geographic accuracy.
"""

import argparse
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from geo_utils import haversine_distance, calculate_bearing, format_distance, geocode_place
from stl_generator import DirectionSignGenerator
import os


@dataclass
class Location:
    """Represents a location with name, coordinates, and optional distance."""
    name: str
    latitude: float
    longitude: float
    location: str = None  # Full location name for lookup/display context
    distance_km: float = None  # Optional: can be calculated or provided
    bearing: float = None  # Bearing in degrees from home (0-360)
    font: str = "Arial"  # Font for the sign text
    sign_color: str = "blue"  # Color for the sign
    text_color: str = "white"  # Color for the sign text


def load_config(path: str) -> Tuple[Location, List[Location], str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    units = config.get("units", "mi")
    user_agent = config.get("user_agent", "direction_sign/1.0")
    home_cfg = config["home"]
    config_dirty = False
    if "latitude" not in home_cfg or "longitude" not in home_cfg:
        home_query = home_cfg.get("location") or home_cfg["name"]
        home_lat, home_lon = geocode_place(home_query, user_agent=user_agent)
        home_cfg["latitude"] = home_lat
        home_cfg["longitude"] = home_lon
        config_dirty = True
    else:
        home_lat, home_lon = home_cfg["latitude"], home_cfg["longitude"]
    home = Location(
        name=home_cfg["name"],
        latitude=home_lat,
        longitude=home_lon,
        location=home_cfg.get("location", home_cfg["name"]),
        font=home_cfg.get("font", "Arial"),
        sign_color=home_cfg.get("sign_color", "blue"),
        text_color=home_cfg.get("text_color", "white"),
    )
    locations = []
    for entry in config.get("locations", []):
        if "latitude" not in entry or "longitude" not in entry:
            query = entry.get("location") or entry["name"]
            lat, lon = geocode_place(query, user_agent=user_agent)
            entry["latitude"] = lat
            entry["longitude"] = lon
            config_dirty = True
        else:
            lat, lon = entry["latitude"], entry["longitude"]
        locations.append(
            Location(
                name=entry["name"],
                latitude=lat,
                longitude=lon,
                location=entry.get("location", entry["name"]),
                font=entry.get("font", "Arial"),
                sign_color=entry.get("sign_color", "blue"),
                text_color=entry.get("text_color", "white"),
            )
        )
    if config_dirty:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        print(f"Updated config with geocoded lat/long: {path}")
    return home, locations, units, user_agent

def main():
    """Main entry point for generating direction sign STLs."""
    parser = argparse.ArgumentParser(description="Direction Sign Generator")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--spacers", type=int, default=0, help="Number of spacer segments to add")
    args = parser.parse_args()
    HOME, LOCATIONS, units, user_agent = load_config(args.config)

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
    print(f"{'Location':<35} {'Distance':<20} {'Bearing':<15} {'Font':<10} {'Sign':<10} {'Text':<10}")
    print("=" * 70)
    
    for loc in LOCATIONS:
        distance_str = format_distance(loc.distance_km, units=units)
        bearing_str = f"{loc.bearing:.1f}°"
        print(f"{loc.name:<35} {distance_str:<20} {bearing_str:<15} {loc.font:<10} {loc.sign_color:<10} {loc.text_color:<10}")
    
    print("\n" + "=" * 70)
    print("\nGenerating STL files...")
    print("=" * 70)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the STL generator
    generator = DirectionSignGenerator()
    
    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    # Generate segmented posts with fixed home flat on south (180°)
    post_segments = []
    post_segments.append({"bearing": 180.0, "segment_id": 1})
    for i, loc in enumerate(LOCATIONS):
        bearing = loc.bearing if loc.bearing <= 180 else loc.bearing - 180
        post_segments.append({"bearing": bearing, "segment_id": i + 2})
    for _ in range(max(0, args.spacers)):
        post_segments.append({"spacer": True})
    print(f"\nOriginal bearings: {[f'{loc.bearing:.1f}' for loc in LOCATIONS]}")
    adjusted_bearings = [f"{s['bearing']:.1f}" for s in post_segments if "bearing" in s]
    print(f"Adjusted bearings: {adjusted_bearings}")
    post_path = os.path.join(output_dir, f"{config_basename}_post.stl")
    generator.generate_post(post_segments, post_path, HOME.latitude, HOME.longitude)
    
    # Generate individual sign plates for each location
    print("\nGenerating sign plates...")
    home_sign_path = os.path.join(
        output_dir,
        f"{config_basename}_sign_1_{HOME.name.replace(' ', '_').replace(',', '')}.stl"
    )
    generator.generate_sign(HOME.name, "", home_sign_path, 180.0, segment_id=1, arrowed=False)
    for i, loc in enumerate(LOCATIONS):
        sign_filename = f"{config_basename}_sign_{i+2}_{loc.name.replace(' ', '_').replace(',', '')}.stl"
        sign_path = os.path.join(output_dir, sign_filename)
        distance_str = format_distance(loc.distance_km, units=units)
        # Pass bearing to determine sign direction
        generator.generate_sign(loc.name, distance_str, sign_path, loc.bearing, segment_id=i + 2)
    
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
