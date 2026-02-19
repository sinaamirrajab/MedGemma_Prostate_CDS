#!/usr/bin/env python3
"""
Generate MedSigLIP input preview images for all patients in the UI CSV.

For each patient, produces a side-by-side center-slice montage of T2W, ADC, HBV
using the same preprocessing as the classifier (SimpleITK resample → 448-px crop
→ 1st-99th-percentile normalise).  The PNG is saved next to the existing .mha
files and a new column  cls_input_preview  is written back to the CSV.

Usage (run from the repo root):
    /mnt/data9/conda/medgemma/bin/python scripts/generate_previews.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image as PILImage, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent.parent
CSV_PATH    = REPO_ROOT / "imgs" / "df_ui_test_4cases.csv"
PREVIEW_COL = "cls_input_preview"

# ---------------------------------------------------------------------------
# Preprocessing  (mirrors classifier_api.py exactly)
# ---------------------------------------------------------------------------

def _resample(img: sitk.Image, spacing=(0.3125, 0.3125, 3.0)) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size    = img.GetSize()
    new_size = [
        int(round(orig_size[i] * (orig_spacing[i] / spacing[i])))
        for i in range(3)
    ]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(spacing)
    r.SetSize(new_size)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetInterpolator(sitk.sitkLinear)
    r.SetDefaultPixelValue(0)
    return r.Execute(img)


def _center_crop_pad(vol: np.ndarray, hw: int = 448) -> np.ndarray:
    z, h, w = vol.shape
    out = np.zeros((z, hw, hw), dtype=vol.dtype)
    h0 = max((h - hw) // 2, 0); w0 = max((w - hw) // 2, 0)
    h1 = min(h0 + hw, h);       w1 = min(w0 + hw, w)
    oh0 = max((hw - h) // 2, 0); ow0 = max((hw - w) // 2, 0)
    out[:, oh0:oh0+(h1-h0), ow0:ow0+(w1-w0)] = vol[:, h0:h1, w0:w1]
    return out


def _load_vol(path: str, hw: int = 448) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    img = _resample(img)
    vol = sitk.GetArrayFromImage(img)
    return _center_crop_pad(vol, hw)


def _norm_slice(sl: np.ndarray) -> np.ndarray:
    """1st-99th percentile normalise to [0, 255] uint8."""
    sl = sl.astype(np.float32)
    lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
    sl = np.clip(sl, lo, hi)
    if hi > lo:
        sl = (sl - lo) / (hi - lo) * 255.0
    else:
        sl = np.zeros_like(sl)
    return sl.astype(np.uint8)


# ---------------------------------------------------------------------------
# Montage builder
# ---------------------------------------------------------------------------

TILE    = 224   # px per modality panel
GAP     = 8     # px gap between panels
LABEL_H = 20    # px for label strip at top

LABELS  = ["T2W", "ADC", "HBV"]


def _build_montage(paths: list[str]) -> PILImage.Image:
    """
    Create a horizontal montage of center slices for the given modality paths.
    Missing/broken paths are replaced with black tiles.
    """
    total_w = len(paths) * TILE + (len(paths) - 1) * GAP
    total_h = TILE + LABEL_H
    canvas  = PILImage.new("RGB", (total_w, total_h), color=(0, 0, 0))
    draw    = ImageDraw.Draw(canvas)

    for i, (path, label) in enumerate(zip(paths, LABELS)):
        x = i * (TILE + GAP)
        # Try to load the volume; fall back to black tile
        tile = PILImage.new("L", (TILE, TILE), 0)
        if path and Path(path).exists():
            try:
                vol = _load_vol(path, hw=448)
                center_sl = vol[vol.shape[0] // 2]
                tile = PILImage.fromarray(_norm_slice(center_sl))
                tile = tile.resize((TILE, TILE), PILImage.LANCZOS)
            except Exception as exc:
                print(f"  WARNING: could not load {path}: {exc}", file=sys.stderr)
        else:
            print(f"  WARNING: path not found or empty: {path!r}", file=sys.stderr)

        # Paste the tile below the label strip (convert L→RGB for unified canvas)
        canvas.paste(tile.convert("RGB"), (x, LABEL_H))

        # Draw label centred above the tile
        # Approximate text width with default font (≈6 px/char)
        approx_w = len(label) * 7
        draw.text(
            (x + (TILE - approx_w) // 2, 4),
            label,
            fill=(200, 200, 200),
        )

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} patients from {CSV_PATH}")

    preview_paths: list[str] = []
    for _, row in df.iterrows():
        pid      = str(row["patient_id"])
        t2w_path = str(row.get("t2w", ""))
        adc_path = str(row.get("adc", ""))
        hbv_path = str(row.get("hbv", ""))

        # Save preview next to the existing MHA files
        # Use the folder of the t2w image if available, else imgs/{pid}/
        if t2w_path and Path(t2w_path).exists():
            out_dir = Path(t2w_path).parent
        else:
            out_dir = REPO_ROOT / "imgs" / pid
            out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{pid}_input_preview.png"

        print(f"  [{pid}] generating preview → {out_path}")
        montage = _build_montage([t2w_path, adc_path, hbv_path])
        montage.save(out_path)
        print(f"  [{pid}] saved {out_path.stat().st_size // 1024} kB")

        preview_paths.append(str(out_path))

    df[PREVIEW_COL] = preview_paths
    df.to_csv(CSV_PATH, index=False)
    print(f"\nCSV updated — column '{PREVIEW_COL}' added to {CSV_PATH}")


if __name__ == "__main__":
    main()
