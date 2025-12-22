# Direction Sign Generator

A Python project that generates STL files for 3D-printable directional signs based on geographic coordinates.

## Overview

This project creates a directional sign system consisting of:
- **Post**: A central post with flat surfaces oriented toward each destination
- **Base**: A base platform with a north arrow indicator
- **Signs**: Directional sign plates with location names and distances
- **Arrows**: Arrow pointers to attach to the signs

The sign sits at a "home" location and points directional signs toward real geographic locations based on latitude/longitude coordinates.

## Project Structure

```
direction_sign/
├── src/
│   ├── main.py           # Main script with location data
│   ├── geo_utils.py      # Geographic calculations (bearing, distance)
│   └── stl_generator.py  # STL file generation
├── output/               # Generated STL files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Edit `src/main.py` to set your home location and destinations:
```python
HOME = Location(
    name="Home",
    latitude=40.7128,
    longitude=-74.0060
)

LOCATIONS = [
    Location("Paris", 48.8566, 2.3522),
    Location("Tokyo", 35.6762, 139.6503),
    # Add more locations...
]
```

2. Run the generator:
```bash
python src/main.py
```

3. Find the generated STL files in the `output/` directory

## How It Works

1. **Calculate Bearings**: Uses the haversine formula to calculate the bearing (compass direction) from home to each destination
2. **Calculate Distances**: Computes great circle distances between locations
3. **Generate Post**: Creates a post with flat mounting surfaces at the calculated bearings
4. **Generate Signs**: Creates sign plates with location names and distances
5. **Export STL**: Saves all components as STL files ready for 3D printing

## Customization

Adjust dimensions in `stl_generator.py`:
- `post_height`: Height of the main post (default: 150mm)
- `post_radius`: Radius of the post (default: 15mm)
- `sign_width`: Width of directional signs (default: 100mm)
- `sign_height`: Height of signs (default: 20mm)

## Future Enhancements

- [ ] Implement actual STL geometry generation
- [ ] Add text embossing/debossing on signs
- [ ] Support for custom fonts
- [ ] Assembly instructions generator
- [ ] Multi-level signs for many destinations

## License

MIT
