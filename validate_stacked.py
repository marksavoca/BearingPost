#!/usr/bin/env python3
"""
Quick validation of stacked distance text calculations.
This script shows what distance text will look like without running full STL generation.
"""

def test_stacked_distance():
    """Test the stacked distance text layout."""
    
    # Test cases matching the locations in main.py
    test_cases = [
        ("Albany, NY", "214 mi"),
        ("Wharton, NJ", "40 mi"),
        ("Stone Harbor, NJ", "78 mi"),
        ("Rome, Italy", "4338 mi"),
        ("Boston, MA", "257 mi"),
        ("Philadelphia, PA", "14 mi"),
        ("Los Angeles, CA", "2409 mi"),
    ]
    
    print("Stacked Distance Text Layout Validation")
    print("=" * 80)
    print()
    
    for location, distance in test_cases:
        # Extract number from distance
        dist_number = distance.split()[0]
        dist_unit = distance.split()[1] if len(distance.split()) > 1 else ""
        
        # Calculate character widths
        main_len = len(location)
        num_len = len(dist_number)
        
        # Simulate font sizes (20mm main, 10mm distance number, 8mm MI)
        main_font = 20.0
        num_font = 10.0
        mi_font = 8.0
        
        # Approximate widths (0.6 * font_size per character)
        main_width = main_len * main_font * 0.6
        old_dist_width = len(distance) * num_font * 0.6  # Old horizontal layout
        new_num_width = max(num_len * num_font * 0.6, 2 * num_font * 0.6)  # New stacked (just number)
        mi_width = 2 * mi_font * 0.6
        
        savings = old_dist_width - new_num_width
        
        print(f"Location: {location}")
        print(f"  Distance: {distance}")
        print(f"  Stacked layout:")
        print(f"    Top:    '{dist_number}' ({num_len} chars @ {num_font}mm = {new_num_width:.1f}mm wide)")
        print(f"    Bottom: '{dist_unit.upper()}' (2 chars @ {mi_font}mm = {mi_width:.1f}mm wide)")
        print(f"  Width savings: {old_dist_width:.1f}mm → {new_num_width:.1f}mm ({savings:.1f}mm saved)")
        print(f"  Main text: '{location}' ({main_len} chars @ {main_font}mm = {main_width:.1f}mm)")
        
        # Calculate if it fits (max 200mm sign, -30mm for point, -10% padding = ~160mm available)
        available = 160.0
        total_needed = main_width + 5.0 + new_num_width  # 5mm min gap
        fits = "✓ FITS" if total_needed <= available else "✗ NEEDS REDUCTION"
        
        print(f"  Total needed: {total_needed:.1f}mm / {available:.1f}mm available → {fits}")
        print()

if __name__ == "__main__":
    test_stacked_distance()
