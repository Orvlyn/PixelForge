# PixelForge Split Tools

This folder contains the category-based standalone versions of PixelForge, split out from the main DNK app into separate focused tools.

The goal of this setup is to keep each app simpler, cleaner, and easier to ship on its own while still keeping the same overall PixelForge look, theme system, updater flow, and general user experience.

## Overview

Instead of one large all-in-one app, the toolset is separated into four standalone apps:

- `image_tools_standalone.py`
- `creative_tools_standalone.py`
- `color_lab_standalone.py`
- `utilities_standalone.py`

Each standalone is designed to feel like part of the same family while only exposing the tools relevant to its category.

## Shared Features

All standalone apps include:

- A dedicated `Home` page tailored to that category
- Quick-launch buttons for faster access to tools
- Theme presets shared with the wider PixelForge style direction
- GitHub-based icon/update flow
- In-app version checking tied to `version.json`
- A more focused layout than the original all-in-one DNK setup

## Apps And Included Tools

### Image Tools

`image_tools_standalone.py` is focused on image processing and visual asset workflows.

Included tools:

- Image Resizer
- Photo Editing
- Watermark
- Background Tools
- Vectorization

This standalone is the best fit for general image prep, cleanup, export adjustments, and asset conversion workflows.

### Creative Tools

`creative_tools_standalone.py` is focused on layout generation, pixel workflows, and presentation tools.

Included tools:

- Pixel Art Mode
- Power-of-Two
- Image Grid
- Border
- Texture Preview

This standalone is intended more for stylized asset creation, texture prep, presentation, and workflow helpers often used around games or digital art pipelines.

### Color Lab

`color_lab_standalone.py` is the palette and color utility app.

Included tools:

- Palette Extractor
- HEX Tool

This standalone is centered around palette generation, color picking, HEX workflows, and visual color experimentation in a smaller dedicated app.

### Utilities

`utilities_standalone.py` contains the more practical helper tools.

Included tools:

- Rename Tool
- Folder Analyzer
- Format Converter
- URL Scraper

This standalone is intended for cleanup, batch utility tasks, lightweight file handling, and general support workflows around media and folders.

## Notes

- The standalone apps are meant to stay visually consistent with the full PixelForge app, but each one has a narrower scope.
- `version.json` is used for in-app update/version checks across the split tools.
- The split structure is intended to make release management easier while keeping each download more focused.
