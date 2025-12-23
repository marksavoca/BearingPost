"""
Geographic utility functions for calculating bearings and distances.
"""

import math
from typing import Tuple



def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in kilometers
    earth_radius = 6371.0
    
    return earth_radius * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing (forward azimuth) from point 1 to point 2.
    
    Args:
        lat1, lon1: Latitude and longitude of starting point in degrees
        lat2, lon2: Latitude and longitude of destination point in degrees
    
    Returns:
        Bearing in degrees (0-360, where 0 is North, 90 is East)
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    # Calculate bearing
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    
    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360
    return (bearing_deg + 360) % 360


def format_distance(distance_km: float, units: str = 'mi') -> str:
    """
    Format distance in a human-readable way.
    
    Args:
        distance_km: Distance in kilometers
        units: 'mi' for miles, 'km' for kilometers, 'both' for both
    
    Returns:
        Formatted string (e.g., "1,234 mi" or "5,678 km")
    """
    distance_mi = distance_km * 0.621371
    
    if units == 'mi':
        if distance_mi < 10:
            return f"{distance_mi:.1f} mi"
        else:
            return f"{distance_mi:,.0f} mi"
    elif units == 'km':
        if distance_km < 10:
            return f"{distance_km:.1f} km"
        else:
            return f"{distance_km:,.0f} km"
    else:  # both
        return f"{distance_km:,.0f} km ({distance_mi:,.0f} mi)"

