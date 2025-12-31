# BearingPost

BearingPost generates 3D-printable STL parts for a directional sign system based on real-world bearings and distances.

## Overview

The system consists of:
- **Base segment**: a compass-style base with raised N/E/S/W and degree ticks, plus an alignment peg.
- **Post segments**: one segment per sign (plus optional spacers), each with a single flat, alignment features, and magnet pockets.
- **Topper**: a cap that covers the top peg and includes a magnet pocket.
- **Signs**: directional sign plates with location names and two-line distance text. A home sign is also generated (no arrow or distance).

## Project Structure

```
direction_sign/
├── configs/              # JSON configs for locations and styling
├── src/
│   ├── main.py           # CLI entrypoint
│   ├── geo_utils.py      # Geographic calculations
│   └── stl_generator.py  # STL generation
├── output/               # Generated STL files
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

For cleaner unions (single-shell STLs), install the manifold boolean backend:

```bash
pip install manifold3d
```

## Usage

Generate parts from a config file:

```bash
python src/main.py --config configs/example.json
```

Add spacer segments:

```bash
python src/main.py --config configs/example.json --spacers 2
```

Emboss coordinates on the base (optional):

```bash
python src/main.py --config configs/example.json --coords
```

### Config format (JSON)

Each config provides a home location plus destinations. `name` is used on the sign, `location` is a fuller label for context, and lat/long can be filled in directly (or left for geocoding if enabled in the future).

```json
{
  "units": "mi",
  "home": {
    "name": "Mullica Hill",
    "location": "Mullica Hill, NJ",
    "latitude": 39.73059345761767,
    "longitude": -75.16805997015658,
    "font": "Arial",
    "sign_color": "blue",
    "text_color": "white"
  },
  "locations": [
    {
      "name": "Albany",
      "location": "Albany, NY",
      "latitude": 42.697749760384546,
      "longitude": -73.96640153872819,
      "font": "Arial",
      "sign_color": "blue",
      "text_color": "white"
    }
  ]
}
```

## Output

For a config named `example.json`, the generator outputs:
- `example_post_lower.stl` (base + lower post)
- `example_post_upper.stl` (upper post)
- `example_sign_1_<HOME>.stl` (home sign, no arrow/distance)
- `example_sign_2_<LOCATION>.stl`, ...

## Notes

- Segment IDs are encoded using 4 vertical ID pins (1–15). If you exceed 15 segments, the generator falls back to the single center pin/hole for that segment.
- The two post parts align using two asymmetrical glue pins for orientation.
- The base includes a bottom engraving: `Mark W Savoca © <current year>`.
- The generator requires the manifold boolean engine for clean unions.

## License

MIT
