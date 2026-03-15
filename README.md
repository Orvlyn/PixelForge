# PixelForge Split Tools

This folder contains category-split standalone apps extracted from the DNK monolith.

## Apps

- `image_tools_standalone.py`
  - Image Resizer
  - Photo Editing
  - Watermark
  - Background Tools
  - Vectorization
- `creative_tools_standalone.py`
  - Pixel Art Mode
  - Power-of-Two
  - Image Grid
  - Border
  - Texture Preview
- `color_lab_standalone.py`
  - Palette Extractor
  - HEX Tool
- `utilities_standalone.py`
  - Duplicate Finder
  - Rename Tool
  - Folder Analyzer
  - Format Converter
  - URL Scraper

Each app now includes:

- A category-specific `Home` page with quick launch buttons
- In-app update checker wired to `version.json`
- Shared PixelForge styling and icon flow

## Build

From this folder:

```bat
build_standalone_exes.bat
```

Build with scraper dependencies included:

```bat
build_standalone_exes.bat with-scraper
```

By default, scraper dependencies are excluded to reduce package size.

## Scraper Recommendation

For an initial open-source release, ship without scraper dependencies (`default` build) and keep scraper as optional.

Why:

- Smaller download footprint
- Fewer dependency/runtime issues
- Less support burden from browser/site compatibility changes

If your audience needs scraping immediately, publish a second build variant with `with-scraper`.
