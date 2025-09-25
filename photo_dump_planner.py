#!/usr/bin/env python3
"""Photo dump planner script.

This tool analyzes a batch of images and curates a 10-photo Instagram carousel.
It computes simple aesthetic metrics, runs subject detection with torchvision
models, extracts dominant colors, sequences the photos for visual flow, and
produces a collage preview with an auto-generated caption.

Assumptions:
- Requires Pillow, numpy, torch, torchvision, opencv-python, and scikit-image.
- Pretrained torchvision weights are downloaded ahead of time (first run may
  download them if network access is available).
- Designed for local experimentation; the heuristics are intentionally simple
  and can be adjusted for specific needs.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from torchvision.models import (
    ResNet50_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    resnet50,
    fasterrcnn_resnet50_fpn,
)


@dataclass
class ImageAnalysisResult:
    """Holds analysis data for a single image."""

    path: Path
    brightness: float
    contrast: float
    edge_density: float
    sharpness: float
    colorfulness: float
    dominant_colors: List[Tuple[int, int, int]]
    color_temperature: float
    detection_labels: List[str]
    people_count: int
    subject_focus: float
    scene_label: str
    aesthetic_score: float


# Lazily initialised global models to avoid repeated loading.
_CLASSIFIER = None
_DETECTOR = None


def _load_models() -> Tuple[torch.nn.Module, torch.nn.Module]:
    global _CLASSIFIER, _DETECTOR
    if _CLASSIFIER is None:
        weights = ResNet50_Weights.DEFAULT
        _CLASSIFIER = resnet50(weights=weights)
        _CLASSIFIER.eval()
    if _DETECTOR is None:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        _DETECTOR = fasterrcnn_resnet50_fpn(weights=weights)
        _DETECTOR.eval()
    return _CLASSIFIER, _DETECTOR


def _preprocess_for_classifier(image: Image.Image) -> torch.Tensor:
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess(image).unsqueeze(0)


def _preprocess_for_detector(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [transforms.ToTensor()],
    )
    return transform(image)


def _brightness_contrast_sharpness(image_array: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute basic aesthetic metrics from the image.

    Returns brightness (0-1), contrast (0-1 scaled), edge density, and Laplacian
    sharpness variance.
    """

    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    brightness = float(np.mean(hsv[:, :, 2]) / 255.0)
    contrast = float(np.std(hsv[:, :, 2]) / 128.0)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edge_density = float(np.mean(edges > 0))

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = laplacian_var

    return brightness, contrast, edge_density, sharpness


def _colorfulness_metric(image_array: np.ndarray) -> float:
    """Compute colorfulness using Hasler & Süsstrunk metric."""

    (B, G, R) = cv2.split(cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    rb_mean = float(np.mean(rg))
    yb_mean = float(np.mean(yb))
    rb_std = float(np.std(rg))
    yb_std = float(np.std(yb))
    return math.sqrt(rb_std ** 2 + yb_std ** 2) + 0.3 * math.sqrt(rb_mean ** 2 + yb_mean ** 2)


def _dominant_colors(image_array: np.ndarray, k: int = 5) -> Tuple[List[Tuple[int, int, int]], float]:
    """Find dominant colors via k-means on a subsampled set of pixels.

    Returns (colors, color_temperature) where color_temperature is a heuristic
    in Kelvin-like scale derived from average hue.
    """

    pixels = image_array.reshape(-1, 3)
    sample_count = min(5000, len(pixels))
    if sample_count < len(pixels):
        idx = np.random.choice(len(pixels), sample_count, replace=False)
        pixels = pixels[idx]

    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    centers = centers.astype(int)
    colors = [tuple(map(int, center)) for center in centers]

    hsv_centers = cv2.cvtColor(np.array([centers], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0]
    avg_hue = float(np.mean(hsv_centers[:, 0]))
    # Rough mapping of hue (0-179 in OpenCV) to warm/cool scale (blue ~ cool, red/yellow ~ warm).
    color_temperature = 2000 + (avg_hue / 179.0) * 6500

    return colors, color_temperature


def _detection_summary(detector: torch.nn.Module, tensor: torch.Tensor, image_area: float, confidence: float = 0.6) -> Tuple[List[str], int, float]:
    """Run Faster R-CNN detection and summarise labels."""

    with torch.inference_mode():
        outputs = detector([tensor])[0]

    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    boxes = outputs["boxes"].cpu().numpy()

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    categories = weights.meta["categories"]

    label_counts: Dict[str, int] = {}
    people_count = 0
    best_focus = 0.0

    for label, score, box in zip(labels, scores, boxes):
        if score < confidence:
            continue
        name = categories[label]
        label_counts[name] = label_counts.get(name, 0) + 1
        if name == "person":
            people_count += 1
        box_area = float((box[2] - box[0]) * (box[3] - box[1]))
        focus_ratio = box_area / image_area
        best_focus = max(best_focus, focus_ratio)

    sorted_labels = sorted(label_counts.keys(), key=lambda key: (-label_counts[key], key))
    return sorted_labels, people_count, best_focus


def _scene_classification(classifier: torch.nn.Module, tensor: torch.Tensor) -> str:
    with torch.inference_mode():
        logits = classifier(tensor)
    weights = ResNet50_Weights.DEFAULT
    idx_to_label = weights.meta["categories"]
    top_idx = int(torch.argmax(logits))
    return idx_to_label[top_idx]


def _score_image(brightness: float, contrast: float, edge_density: float, sharpness: float, colorfulness: float, subject_focus: float) -> float:
    """Combine metrics into a heuristic aesthetic score (0-1)."""

    brightness_score = max(0.0, 1.0 - abs(brightness - 0.55) / 0.55)
    contrast_score = min(contrast, 1.5) / 1.5
    edge_score = min(edge_density * 3.0, 1.0)
    sharpness_score = min(sharpness / 800.0, 1.0)
    color_score = min(colorfulness / 100.0, 1.0)
    focus_score = min(subject_focus * 5.0, 1.0)

    weights = {
        "brightness": 0.15,
        "contrast": 0.2,
        "edge": 0.15,
        "sharpness": 0.15,
        "color": 0.2,
        "focus": 0.15,
    }

    score = (
        brightness_score * weights["brightness"]
        + contrast_score * weights["contrast"]
        + edge_score * weights["edge"]
        + sharpness_score * weights["sharpness"]
        + color_score * weights["color"]
        + focus_score * weights["focus"]
    )
    return score


def analyze_image(path: Path) -> ImageAnalysisResult:
    classifier, detector = _load_models()

    image = Image.open(path).convert("RGB")
    image_array = np.array(image)

    brightness, contrast, edge_density, sharpness = _brightness_contrast_sharpness(image_array)
    colorfulness = _colorfulness_metric(image_array)
    dominant_colors, color_temperature = _dominant_colors(image_array)

    detector_tensor = _preprocess_for_detector(image)
    labels, people_count, subject_focus = _detection_summary(detector, detector_tensor, float(image.width * image.height))

    classifier_tensor = _preprocess_for_classifier(image)
    scene_label = _scene_classification(classifier, classifier_tensor)

    aesthetic_score = _score_image(brightness, contrast, edge_density, sharpness, colorfulness, subject_focus)

    return ImageAnalysisResult(
        path=path,
        brightness=brightness,
        contrast=contrast,
        edge_density=edge_density,
        sharpness=sharpness,
        colorfulness=colorfulness,
        dominant_colors=dominant_colors,
        color_temperature=color_temperature,
        detection_labels=labels,
        people_count=people_count,
        subject_focus=subject_focus,
        scene_label=scene_label,
        aesthetic_score=aesthetic_score,
    )


def select_top_images(results: Sequence[ImageAnalysisResult], top_n: int = 10) -> List[ImageAnalysisResult]:
    """Pick top-n results while promoting subject diversity."""

    sorted_results = sorted(results, key=lambda item: item.aesthetic_score, reverse=True)
    seen_labels: Dict[str, int] = {}
    scored: List[Tuple[float, ImageAnalysisResult]] = []

    for result in sorted_results:
        penalty = 0.0
        for label in result.detection_labels[:3]:
            penalty += seen_labels.get(label, 0) * 0.05
        penalised_score = result.aesthetic_score - penalty
        scored.append((penalised_score, result))
        for label in result.detection_labels[:3]:
            seen_labels[label] = seen_labels.get(label, 0) + 1

    scored.sort(key=lambda pair: pair[0], reverse=True)
    curated = [dataclasses.replace(res, aesthetic_score=score) for score, res in scored[:top_n]]
    return curated


def _feature_vector(result: ImageAnalysisResult) -> np.ndarray:
    labels_hash = hash("|".join(result.detection_labels[:3])) % 97 / 96.0
    return np.array(
        [
            result.brightness,
            result.color_temperature / 8500.0,
            result.people_count / 5.0,
            labels_hash,
        ]
    )


def arrange_sequence(results: Sequence[ImageAnalysisResult]) -> List[ImageAnalysisResult]:
    if not results:
        return []

    remaining = list(results)
    ordered = [remaining.pop(0)]

    while remaining:
        last = ordered[-1]
        last_vec = _feature_vector(last)
        distances = []
        for candidate in remaining:
            vec = _feature_vector(candidate)
            distance = np.linalg.norm(last_vec - vec)
            distances.append((distance, -candidate.aesthetic_score, candidate))
        distances.sort(reverse=True)
        _best_distance, _neg_score, best_candidate = distances[0]
        ordered.append(best_candidate)
        remaining.remove(best_candidate)

    return ordered


def _color_name(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    # Simple approximate mapping to color families.
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 60 and g < 60 and b < 60:
        return "charcoal"
    if r > g and r > b:
        return "red" if g < 120 else "orange"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue"
    return "neutral"


def generate_caption(results: Sequence[ImageAnalysisResult]) -> str:
    if not results:
        return ""

    labels: Dict[str, int] = {}
    color_words: Dict[str, int] = {}

    for result in results:
        for label in result.detection_labels[:2]:
            labels[label] = labels.get(label, 0) + 1
        for color in result.dominant_colors[:2]:
            color_words[_color_name(color)] = color_words.get(_color_name(color), 0) + 1

    top_labels = sorted(labels, key=lambda key: labels[key], reverse=True)[:3]
    top_colors = sorted(color_words, key=lambda key: color_words[key], reverse=True)[:2]

    label_phrase = ", ".join(top_labels) if top_labels else "little moments"
    color_phrase = " & ".join(top_colors) if top_colors else "soft hues"

    intro = "photo dump vibes" if len(results) > 5 else "weekend snaps"
    closing = "grateful for the in-between bits" if "person" in top_labels else "catching feelings for these views"

    return f"{intro} • {label_phrase} and {color_phrase}. {closing}."


def create_preview(results: Sequence[ImageAnalysisResult], caption: str, output_path: Path, thumb_size: Tuple[int, int] = (540, 540)) -> None:
    if not results:
        raise ValueError("No results to render")

    columns = 5
    rows = math.ceil(len(results) / columns)
    thumb_w, thumb_h = thumb_size
    padding = 20
    caption_height = 120
    width = columns * thumb_w + (columns + 1) * padding
    height = rows * thumb_h + (rows + 1) * padding + caption_height

    collage = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(collage)
    font = ImageFont.load_default()

    for idx, result in enumerate(results):
        image = Image.open(result.path).convert("RGB")
        image.thumbnail(thumb_size)
        col = idx % columns
        row = idx // columns
        x = padding + col * (thumb_w + padding)
        y = padding + row * (thumb_h + padding)
        # Paste centered within the thumbnail cell.
        paste_x = x + (thumb_w - image.width) // 2
        paste_y = y + (thumb_h - image.height) // 2
        collage.paste(image, (paste_x, paste_y))

    text_x = padding
    text_y = rows * (thumb_h + padding) + padding
    draw.text((text_x, text_y), caption, fill=(240, 240, 240), font=font)

    collage.save(output_path)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate a photo dump from 50 images")
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Paths to image files (expects at least 10, ideally 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preview.jpg"),
        help="Path for the generated collage preview",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    image_paths = [path for path in args.images if path.exists()]
    if len(image_paths) < 10:
        raise SystemExit("Please provide at least 10 valid image paths")

    print(f"Analyzing {len(image_paths)} images…")
    results = [analyze_image(path) for path in image_paths]
    results.sort(key=lambda item: item.aesthetic_score, reverse=True)

    print("Selecting top shots…")
    top_results = select_top_images(results, top_n=10)

    print("Arranging for visual flow…")
    ordered = arrange_sequence(top_results)

    caption = generate_caption(ordered)
    create_preview(ordered, caption, args.output)

    print("Done! Selected images in order:")
    for result in ordered:
        print(f" - {result.path} (score={result.aesthetic_score:.2f}, scene={result.scene_label})")
    print(f"Caption: {caption}")
    print(f"Preview saved to {args.output}")


if __name__ == "__main__":
    main()
