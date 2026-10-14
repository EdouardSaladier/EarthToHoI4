"""Hoi4renderer: the HOI4 political map look (zoomed out), in software.

A copy of the standalone Hoi4renderer project (C:/Users/Alexandre/Documents/
Hoi4renderer, 2026-10-13), vendored unchanged so the tool stays self-contained
-- its dependencies (numpy, scipy, opencv-python, Pillow) are already ours.
Only this file differs from upstream: re-copy the other modules to update.

EarthToHoI4 uses it for the mod's thumbnail (see modthumbnail.py); the whole
command line is still there: `python -m hoi4renderer "<mod folder>"`.
"""
