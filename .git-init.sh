#!/bin/bash
cd "$(dirname "$0")"
git init
git add .
git commit -m "Initial commit: Direction sign generator with geographic positioning

- Main script with location data table (home + 7 destinations)
- Geographic utilities (haversine distance, bearing calculations)
- STL generator with post and base generation
- Post generation with single test box properly positioned
- Box correctly rotates around post at specified bearing
- Base with north arrow indicator
- Requirements and documentation"
