"""
Image Enhancement Pipeline — Steps 31–55 of the 100-step recognition plan.
===========================================================================
All functions are pure PIL — no numpy, no OpenCV, no extra installs.
Every crop strategy + enhancement is composable.

The philosophy: surveillance cameras work because they capture at high resolution
then crop tightly on the region of interest. We replicate this here:
  1. Take a VRChat frame (typically 1600×900 or 1920×1080)
  2. Crop aggressively to just the nametag region
  3. Upscale 4–6× with Lanczos
  4. Sharpen, contrast-stretch, gamma-correct
  5. Feed the resulting crisp crop to the vision model with an OCR prompt

The 6 crop strategies cover all the different ways a name might appear:
  - nametag_strip  : full width, top 55% (default, catches all heights)
  - upper_half     : full width, top 50%
  - top_quarter    : full width, top 25% (very close players)
  - upper_center   : center 70%, top 45% (removes side clutter)
  - tiled_3x3      : 9 tiles (the nametag is somewhere in one of them)
  - grayscale_*    : grayscale copy of nametag_strip for OCR focus
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageStat


# ---------------------------------------------------------------------------
# Step 31: CropVariant dataclass
# ---------------------------------------------------------------------------

@dataclass
class CropVariant:
    """A single processed crop ready for a vision model query."""
    name:          str          # e.g. 'nametag_strip_4x', 'top_quarter_6x_gray'
    image:         Image.Image  # PIL Image (RGB)
    quality_score: float        # 0.0–1.0 (higher = better for OCR)
    source_path:   str  = ''    # original frame this came from
    saved_path:    str  = ''    # where the crop was saved on disk ('' if not saved)
    scale_factor:  int  = 1     # upscale factor applied
    strategy:      str  = ''    # crop strategy name (e.g. 'nametag_strip')
    metadata:      dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 32–36: Crop strategies
# ---------------------------------------------------------------------------

def crop_nametag_strip(img: Image.Image) -> Image.Image:
    """Full-width crop of the top 55% — catches all floating nametags."""
    w, h = img.size
    return img.crop((0, 0, w, int(h * 0.55)))


def crop_upper_half(img: Image.Image) -> Image.Image:
    """Full-width crop of the top 50%."""
    w, h = img.size
    return img.crop((0, 0, w, h // 2))


def crop_top_quarter(img: Image.Image) -> Image.Image:
    """Full-width crop of the top 25% — for very close players."""
    w, h = img.size
    return img.crop((0, 0, w, h // 4))


def crop_upper_center(img: Image.Image) -> Image.Image:
    """Center 70% horizontally, top 45% vertically — removes side clutter."""
    w, h = img.size
    x0   = int(w * 0.15)
    x1   = int(w * 0.85)
    y1   = int(h * 0.45)
    return img.crop((x0, 0, x1, y1))


# --- Vertically-divided strip crops (column × height band) ---
# VRChat nametags float above a player's head anywhere in the frame.
# Dividing the frame into left / center / right columns, each paired with
# a height band, lets the pipeline focus on one nametag at a time.

def crop_left_strip(img: Image.Image) -> Image.Image:
    """Left 40% of frame, top 55% — nametags on the left side of the screen."""
    w, h = img.size
    return img.crop((0, 0, int(w * 0.40), int(h * 0.55)))


def crop_center_strip(img: Image.Image) -> Image.Image:
    """Center 40% horizontally (x: 30–70%), top 55% — nametags directly ahead."""
    w, h = img.size
    return img.crop((int(w * 0.30), 0, int(w * 0.70), int(h * 0.55)))


def crop_right_strip(img: Image.Image) -> Image.Image:
    """Right 40% of frame, top 55% — nametags on the right side of the screen."""
    w, h = img.size
    return img.crop((int(w * 0.60), 0, w, int(h * 0.55)))


def crop_left_quarter(img: Image.Image) -> Image.Image:
    """Left 40%, top 25% — close players on the left."""
    w, h = img.size
    return img.crop((0, 0, int(w * 0.40), int(h * 0.25)))


def crop_center_quarter(img: Image.Image) -> Image.Image:
    """Center 40%, top 25% — close players directly ahead."""
    w, h = img.size
    return img.crop((int(w * 0.30), 0, int(w * 0.70), int(h * 0.25)))


def crop_right_quarter(img: Image.Image) -> Image.Image:
    """Right 40%, top 25% — close players on the right."""
    w, h = img.size
    return img.crop((int(w * 0.60), 0, w, int(h * 0.25)))


def crop_tiled_3x3(img: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    Split the top 60% of the frame into a 3×3 grid of equal tiles.
    Returns [(tile_label, PIL.Image), ...] — 9 tiles total.
    Nametags are somewhere in one of these tiles.
    """
    w, h   = img.size
    region = img.crop((0, 0, w, int(h * 0.60)))
    rw, rh = region.size
    tw, th = rw // 3, rh // 3
    tiles: List[Tuple[str, Image.Image]] = []
    for row in range(3):
        for col in range(3):
            x0     = col * tw
            y0     = row * th
            x1     = x0 + tw if col < 2 else rw
            y1     = y0 + th if row < 2 else rh
            tile   = region.crop((x0, y0, x1, y1))
            label  = f'tile_r{row}c{col}'
            tiles.append((label, tile))
    return tiles


# ---------------------------------------------------------------------------
# Step 37: upscale
# ---------------------------------------------------------------------------

def upscale(img: Image.Image, factor: int = 4) -> Image.Image:
    """Upscale image by integer factor using high-quality Lanczos resampling."""
    if factor <= 1:
        return img
    new_w = img.width  * factor
    new_h = img.height * factor
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Step 38–39: Sharpening
# ---------------------------------------------------------------------------

def apply_sharpen(img: Image.Image, passes: int = 2) -> Image.Image:
    """Apply PIL's standard SHARPEN kernel N times."""
    for _ in range(passes):
        img = img.filter(ImageFilter.SHARPEN)
    return img


def apply_unsharp_mask(
    img: Image.Image, radius: int = 2, percent: int = 150, threshold: int = 3
) -> Image.Image:
    """Apply unsharp mask — enhances fine text edges."""
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


# ---------------------------------------------------------------------------
# Step 40: apply_contrast_stretch — 2nd–98th percentile normalization
# ---------------------------------------------------------------------------

def apply_contrast_stretch(img: Image.Image) -> Image.Image:
    """Stretch luminance so the darkest 2% → black and brightest 2% → white."""
    gray  = img.convert('L')
    hist  = gray.histogram()
    total = sum(hist)
    lo_thresh = total * 0.02
    hi_thresh = total * 0.98
    cumulative = 0
    lo_val, hi_val = 0, 255
    for pixel_val, count in enumerate(hist):
        cumulative += count
        if cumulative <= lo_thresh:
            lo_val = pixel_val
        if cumulative <= hi_thresh:
            hi_val = pixel_val
    if hi_val <= lo_val:
        return img
    scale = 255.0 / (hi_val - lo_val)
    # Apply via Contrast enhancer scaled to the computed range
    # (full per-pixel LUT would need numpy; we approximate with PIL enhancer)
    factor = max(1.0, scale)
    return ImageEnhance.Contrast(img).enhance(min(factor, 3.0))


# ---------------------------------------------------------------------------
# Step 41: apply_brightness_normalize
# ---------------------------------------------------------------------------

def apply_brightness_normalize(img: Image.Image, target: float = 128.0) -> Image.Image:
    """Adjust brightness so the image mean luminance equals target (default 128)."""
    gray   = img.convert('L')
    mean   = ImageStat.Stat(gray).mean[0]
    if mean < 1.0:
        return img
    factor = target / mean
    factor = max(0.5, min(factor, 2.5))
    return ImageEnhance.Brightness(img).enhance(factor)


# ---------------------------------------------------------------------------
# Step 42: apply_gamma
# ---------------------------------------------------------------------------

def apply_gamma(img: Image.Image, gamma: float = 0.75) -> Image.Image:
    """Apply power-law gamma correction. gamma < 1.0 brightens; > 1.0 darkens."""
    # Build a 256-entry LUT
    lut = bytes([int((i / 255) ** gamma * 255) for i in range(256)])
    if img.mode == 'RGB':
        lut = lut * 3   # R, G, B channels
    elif img.mode == 'RGBA':
        lut = lut * 3 + bytes(range(256))  # preserve alpha
    try:
        return img.point(lut)
    except Exception:
        return img


# ---------------------------------------------------------------------------
# Step 43: apply_denoise
# ---------------------------------------------------------------------------

def apply_denoise(img: Image.Image) -> Image.Image:
    """Apply a 3×3 median-like filter to reduce VRChat UI noise."""
    return img.filter(ImageFilter.MedianFilter(size=3))


# ---------------------------------------------------------------------------
# Step 44: apply_edge_enhance
# ---------------------------------------------------------------------------

def apply_edge_enhance(img: Image.Image) -> Image.Image:
    """Aggressive edge enhancement — makes text outlines crisp."""
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


# ---------------------------------------------------------------------------
# Step 45: to_grayscale_rgb
# ---------------------------------------------------------------------------

def to_grayscale_rgb(img: Image.Image) -> Image.Image:
    """Convert to grayscale then back to RGB (models expect 3-channel input)."""
    return img.convert('L').convert('RGB')


# ---------------------------------------------------------------------------
# Step 46: text_region_boost — brighten pixels likely to be white nametag text
# ---------------------------------------------------------------------------

def text_region_boost(img: Image.Image, threshold: int = 180) -> Image.Image:
    """
    Boost pixels with high luminance (likely white VRChat name tags).
    Pixels with L >= threshold get extra brightness; dark background suppressed.
    """
    gray  = img.convert('L')
    data  = list(gray.getdata())
    boost = bytes([
        min(255, int(v * 1.35)) if v >= threshold else max(0, int(v * 0.7))
        for v in data
    ])
    enhanced_gray = Image.frombytes('L', gray.size, boost)
    # Merge back with original using a soft mask so color is partially preserved
    blended = Image.blend(img.convert('RGB'),
                          enhanced_gray.convert('RGB'), alpha=0.55)
    return blended


# ---------------------------------------------------------------------------
# Step 47–48: Quality scoring
# ---------------------------------------------------------------------------

def image_blur_score(img: Image.Image) -> float:
    """
    Estimate sharpness by computing the variance of edges (Laplacian proxy).
    Uses PIL's FIND_EDGES filter — returns 0.0 (very blurry) to 1.0 (very sharp).
    """
    gray   = img.convert('L')
    edges  = gray.filter(ImageFilter.FIND_EDGES)
    stat   = ImageStat.Stat(edges)
    stddev = stat.stddev[0]
    # Normalize: a crisp crop typically has stddev 10–60; blur = 0–5
    return min(1.0, stddev / 40.0)


def image_quality_score(img: Image.Image) -> float:
    """
    Combined quality score (0.0–1.0) mixing sharpness + contrast.
    Higher score = better candidate for OCR.
    """
    blur_s     = image_blur_score(img)
    gray       = img.convert('L')
    stat       = ImageStat.Stat(gray)
    contrast_s = min(1.0, stat.stddev[0] / 60.0)
    mean_ok    = 0.8 if 30 <= stat.mean[0] <= 220 else 0.4
    return (blur_s * 0.5 + contrast_s * 0.35 + mean_ok * 0.15)


# ---------------------------------------------------------------------------
# Step 49: enhancement_chain — ordered processing pipeline
# ---------------------------------------------------------------------------

def enhancement_chain(
    img:           Image.Image,
    scale_factor:  int   = 4,
    sharpen_passes: int  = 2,
    gamma:         float = 0.80,
    do_contrast:   bool  = True,
    do_brightness: bool  = True,
    do_denoise:    bool  = False,
    do_edge:       bool  = False,
    do_grayscale:  bool  = False,
    do_text_boost: bool  = True,
) -> Image.Image:
    """
    Run the full enhancement pipeline in a sensible order.
    Default settings are tuned for VRChat nametag legibility.
    """
    img = upscale(img, scale_factor)
    if do_denoise:
        img = apply_denoise(img)
    img = apply_sharpen(img, passes=sharpen_passes)
    img = apply_unsharp_mask(img)
    if do_contrast:
        img = apply_contrast_stretch(img)
    if do_brightness:
        img = apply_brightness_normalize(img)
    if gamma != 1.0:
        img = apply_gamma(img, gamma=gamma)
    if do_text_boost:
        img = text_region_boost(img)
    if do_edge:
        img = apply_edge_enhance(img)
    if do_grayscale:
        img = to_grayscale_rgb(img)
    return img


# ---------------------------------------------------------------------------
# Step 50–51: Build all variants (6 raw crops × 2 scales + grayscale version)
# ---------------------------------------------------------------------------

def all_crop_variants(source: Image.Image) -> Dict[str, Image.Image]:
    """
    Return a dict of {variant_name: raw_cropped_PIL_Image} (not yet enhanced).
    Calling code can then run enhancement_chain() on each.
    """
    variants: Dict[str, Image.Image] = {}

    variants['nametag_strip'] = crop_nametag_strip(source)
    variants['upper_half']    = crop_upper_half(source)
    variants['top_quarter']   = crop_top_quarter(source)
    variants['upper_center']  = crop_upper_center(source)

    # Column × height-band variants: target one nametag at a time
    variants['left_strip']     = crop_left_strip(source)
    variants['center_strip']   = crop_center_strip(source)
    variants['right_strip']    = crop_right_strip(source)
    variants['left_quarter']   = crop_left_quarter(source)
    variants['center_quarter'] = crop_center_quarter(source)
    variants['right_quarter']  = crop_right_quarter(source)

    # Grayscale version of the nametag strip (dedicated OCR pass)
    variants['nametag_strip_gray'] = to_grayscale_rgb(crop_nametag_strip(source))

    # Aggressive top strip (just the very top 20% — names at eye level)
    w, h = source.size
    variants['eye_level_strip'] = source.crop((0, 0, w, int(h * 0.20)))

    # Tiled crops — 9 tiles of the top 60%
    for label, tile in crop_tiled_3x3(source):
        variants[label] = tile

    return variants


def build_all_variants(
    source_path:  str,
    scale_factor: int  = 4,
    save_dir:     str  = '',
) -> List[CropVariant]:
    """
    Load source_path, generate all crop variants + enhancement, compute quality.
    Optionally save each variant to save_dir/<variant_name>.png.
    Returns list of CropVariant sorted by quality_score descending.
    """
    try:
        source = Image.open(source_path).convert('RGB')
    except Exception as e:
        print(f'[image_enhance] cannot open {source_path}: {e}', flush=True)
        return []

    raw = all_crop_variants(source)
    result: List[CropVariant] = []

    for variant_name, crop_img in raw.items():
        # Skip near-empty crops (tiles at the edge can be very small)
        if crop_img.width < 40 or crop_img.height < 20:
            continue

        # Enhance — grayscale variants skip the color text_boost step
        is_gray = 'gray' in variant_name
        enhanced = enhancement_chain(
            crop_img,
            scale_factor   = scale_factor,
            sharpen_passes = 2,
            gamma          = 0.80,
            do_contrast    = True,
            do_brightness  = True,
            do_text_boost  = not is_gray,
            do_grayscale   = is_gray,
        )

        quality = image_quality_score(enhanced)
        saved   = ''
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            saved = os.path.join(save_dir, f'{variant_name}.png')
            try:
                enhanced.save(saved)
            except Exception:
                saved = ''

        result.append(CropVariant(
            name         = variant_name,
            image        = enhanced,
            quality_score= quality,
            source_path  = source_path,
            saved_path   = saved,
            scale_factor = scale_factor,
            strategy     = variant_name.split('_')[0],
        ))

    # Sort best quality first so the pipeline tries the clearest crop first
    result.sort(key=lambda cv: cv.quality_score, reverse=True)
    return result


# ---------------------------------------------------------------------------
# Step 52: choose_best_variant
# ---------------------------------------------------------------------------

def choose_best_variant(variants: List[CropVariant]) -> Optional[CropVariant]:
    """Return the single highest-quality CropVariant from the list."""
    return variants[0] if variants else None


# ---------------------------------------------------------------------------
# Steps 53–54: Save/load with sidecar JSON
# ---------------------------------------------------------------------------

def save_enhanced_with_sidecar(
    img: Image.Image, save_path: str, metadata: dict
) -> None:
    """Save a PNG crop and a companion JSON sidecar containing metadata."""
    img.save(save_path)
    sidecar = save_path.replace('.png', '_meta.json')
    metadata_with_ts = dict(metadata)
    metadata_with_ts['saved_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    try:
        with open(sidecar, 'w', encoding='utf-8') as f:
            json.dump(metadata_with_ts, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_sidecar_metadata(png_path: str) -> dict:
    """Load the JSON sidecar for a saved crop. Returns {} if missing."""
    sidecar = png_path.replace('.png', '_meta.json')
    if not os.path.exists(sidecar):
        return {}
    try:
        with open(sidecar, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}
