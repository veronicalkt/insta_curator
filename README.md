# Photo Dump Planner

`photo_dump_planner.py` analyzes a batch of photos, picks the best 10 for an Instagram-style carousel, orders them for a smoother visual flow, and builds a caption plus a collage preview image.

## Features
- Computes simple aesthetic metrics (brightness, contrast, sharpness, edge density, and colorfulness) to rank images.
- Runs torchvision's Faster R-CNN for subject detection and ResNet-50 for scene classification to understand image content.
- Extracts a five-color palette for each image and estimates an overall warm/cool tone.
- Picks the top 10 photos while nudging for subject diversity, then sequences them by maximizing feature contrast between neighbours.
- Generates a casual, diary-like caption that references dominant subjects and colors.
- Exports a `preview.jpg` collage (5×2 grid) with the caption overlaid.

## Requirements & Assumptions
- Python 3.9+
- Python packages: `torch`, `torchvision`, `Pillow`, `numpy`, `opencv-python`, `scikit-image` (install via `pip install torch torchvision Pillow numpy opencv-python scikit-image`).
- The first run may download pretrained weights if they are not cached yet.
- Provide at least 10 valid image paths (designed for ~50). When using the default library mode, the script scans `photo_library/` recursively for supported formats.

## Usage
Place your candidate photos inside the `photo_library/` directory (or supply a newline-delimited file of paths) and run:

```bash
python photo_dump_planner.py --library photo_library --output preview.jpg
```

You can still pass explicit paths to override the library discovery:

```bash
python photo_dump_planner.py img1.jpg img2.jpg ... --output preview.jpg
```

Notes:
- The script prints the curated list in order plus the generated caption.
- Adjust the weighting logic in `_score_image` or the sequencing heuristic in `arrange_sequence` to match your aesthetic preferences.
- The collage uses the default Pillow bitmap font; customise `create_preview` if you need a specific font.

## Next Steps Ideas
- Plug in a dedicated aesthetic scoring model.
- Swap the caption logic for an LLM-backed generator if available.
- Surface the per-image metrics in a JSON report for manual review.
