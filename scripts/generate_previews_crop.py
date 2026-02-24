#!/usr/bin/env python3
"""
Generate PNG slices that match the exact MedSigLIP classifier crop pipeline.

For each patient:
1. Resample modalities at spacing (0.3125, 0.3125, 3.0) with linear interpolation.
2. Resample SEG with nearest-neighbour interpolation.
3. Center-crop/pad to 448x448.
4. Crop around prostate using SEG bbox with margin (2, 20, 20) in (z, y, x).
5. Apply per-slice normalization percentile(1) -> percentile(99) to [0, 255].
6. Concatenate modalities horizontally in model_config.json order.
7. Save only the center slice PNG (same slice-selection rule as generate_previews.py).

Outputs are saved under:
  imgs/{patient_id}/{patient_id}_classifier_input_pngs_{modality_tag}/
and two CSV columns are written:
  - cls_input_png_dir
  - cls_input_png_center
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image as PILImage


REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "imgs" / "df_ui_test_4cases.csv"
MODEL_CONFIG = REPO_ROOT / "backend" / "lr_medsiglip_features" / "model_config.json"

OUT_DIR_COL = "cls_input_png_dir"
OUT_CENTER_COL = "cls_input_png_center"

TARGET_SPACING = (0.3125, 0.3125, 3.0)
TARGET_HW = 448
CROP_MARGIN = (2, 20, 20)  # (z, y, x)


with open(MODEL_CONFIG) as fh:
    CFG = json.load(fh)
# MODALITIES: list[str] = CFG["modalities"]
MODALITIES: list[str] = ["t2w", "adc", "hbv"]  # hardcoded for now since model_config.json is not yet final


def _resample_sitk(img: sitk.Image, interp: int) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_size = [
        int(round(orig_size[i] * (orig_spacing[i] / TARGET_SPACING[i])))
        for i in range(3)
    ]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(TARGET_SPACING)
    r.SetSize(new_size)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetInterpolator(interp)
    r.SetDefaultPixelValue(0)
    return r.Execute(img)


def _center_crop_or_pad(vol: np.ndarray, target_hw: int = TARGET_HW) -> np.ndarray:
    z, h, w = vol.shape
    out = np.zeros((z, target_hw, target_hw), dtype=vol.dtype)
    h0 = max((h - target_hw) // 2, 0)
    w0 = max((w - target_hw) // 2, 0)
    h1 = min(h0 + target_hw, h)
    w1 = min(w0 + target_hw, w)
    oh0 = max((target_hw - h) // 2, 0)
    ow0 = max((target_hw - w) // 2, 0)
    out[:, oh0 : oh0 + (h1 - h0), ow0 : ow0 + (w1 - w0)] = vol[:, h0:h1, w0:w1]
    return out


def _load_vol(path: str) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    img = _resample_sitk(img, sitk.sitkLinear)
    return _center_crop_or_pad(sitk.GetArrayFromImage(img), TARGET_HW)


def _load_seg(path: str) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    img = _resample_sitk(img, sitk.sitkNearestNeighbor)
    return _center_crop_or_pad(sitk.GetArrayFromImage(img), TARGET_HW)


def _compute_bbox_3d(seg: np.ndarray) -> tuple | None:
    if not np.any(seg > 0):
        return None
    mz, my, mx = CROP_MARGIN
    z_idx, y_idx, x_idx = np.where(seg > 0)
    z0 = max(0, int(z_idx.min()) - mz)
    z1 = min(seg.shape[0], int(z_idx.max()) + 1 + mz)
    y0 = max(0, int(y_idx.min()) - my)
    y1 = min(seg.shape[1], int(y_idx.max()) + 1 + my)
    x0 = max(0, int(x_idx.min()) - mx)
    x1 = min(seg.shape[2], int(x_idx.max()) + 1 + mx)
    return (z0, z1), (y0, y1), (x0, x1)


def _norm_slice(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(np.float32)
    lo = float(np.percentile(sl, 1))
    hi = float(np.percentile(sl, 99))
    sl = np.clip(sl, lo, hi)
    if hi > lo:
        sl = (sl - lo) / (hi - lo) * 255.0
    else:
        sl = np.zeros_like(sl)
    return sl.astype(np.uint8)


def _build_classifier_like_stack(path_map: dict[str, str], seg_path: str) -> tuple[np.ndarray, int]:
    unique_mods = list(dict.fromkeys(MODALITIES))
    vols: dict[str, np.ndarray] = {}
    for mod in unique_mods:
        p = path_map.get(mod, "")
        if not p or not Path(p).exists():
            raise FileNotFoundError(f"Modality '{mod}' path not found: {p!r}")
        vols[mod] = _load_vol(p)

    # Match generate_previews.py slice selection: center of resampled+center-cropped
    # reference modality before seg-based z-cropping.
    ref_mod = MODALITIES[0]
    ref_center_pre_crop = vols[ref_mod].shape[0] // 2

    seg_vol = None
    z0_for_mapping = 0
    if seg_path and Path(seg_path).exists():
        seg_vol = _load_seg(seg_path)

    if seg_vol is not None:
        bbox = _compute_bbox_3d(seg_vol)
        if bbox is not None:
            (z0, z1), (y0, y1), (x0, x1) = bbox
            z0_for_mapping = z0
            for mod in unique_mods:
                vols[mod] = vols[mod][z0:z1, y0:y1, x0:x1]

    n_slices = min(v.shape[0] for v in vols.values())
    for mod in unique_mods:
        if vols[mod].shape[0] != n_slices:
            vols[mod] = vols[mod][:n_slices]

    normed: list[np.ndarray] = []
    for mod in MODALITIES:
        v = vols[mod]
        n = np.empty_like(v, dtype=np.uint8)
        for s in range(n_slices):
            n[s] = _norm_slice(v[s])
        normed.append(n)

    # Map center index from pre-seg-crop coordinates to cropped coordinates.
    mapped_center = ref_center_pre_crop - z0_for_mapping
    mapped_center = max(0, min(mapped_center, n_slices - 1))

    return np.concatenate(normed, axis=2), mapped_center


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} patients from {CSV_PATH}")
    print(f"Using modalities from model config: {MODALITIES}")

    out_dirs: list[str] = []
    out_center_pngs: list[str] = []
    unique_tag = "_".join(dict.fromkeys(MODALITIES))

    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        print(f"\n[{pid}]")

        path_map = {mod: str(row.get(mod, "")) for mod in dict.fromkeys(MODALITIES)}
        seg_path = str(row.get("seg", ""))

        patient_dir = REPO_ROOT / "imgs" / pid
        patient_dir.mkdir(parents=True, exist_ok=True)
        out_dir = patient_dir / f"{pid}_classifier_input_pngs_{unique_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            concat, center_idx = _build_classifier_like_stack(path_map, seg_path)
            n_slices = concat.shape[0]
            print(f"  concat stack shape: {concat.shape}")
            center_path = out_dir / f"{pid}_slice_{center_idx:03d}.png"
            PILImage.fromarray(concat[center_idx]).save(center_path)
            print(f"  saved center slice z={center_idx} to {center_path}")

            out_dirs.append(str(out_dir))
            out_center_pngs.append(str(center_path))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"  ERROR: {exc}", file=sys.stderr)
            out_dirs.append("")
            out_center_pngs.append("")

    df[OUT_DIR_COL] = out_dirs
    df[OUT_CENTER_COL] = out_center_pngs
    df.to_csv(CSV_PATH, index=False)
    print(f"\nCSV updated with '{OUT_DIR_COL}' and '{OUT_CENTER_COL}' in {CSV_PATH}")


if __name__ == "__main__":
    main()
