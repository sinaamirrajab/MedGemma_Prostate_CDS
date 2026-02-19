#!/usr/bin/env python3
"""
Generate scrollable NIfTI files that mirror the *exact* MedSigLIP model inputs.

Preprocessing follows medsiglip_multislice_classification_crop_fast_save.py exactly:

  1. Resample T2W/ADC/HBV to target_spacing (0.3125, 0.3125, 3.0 mm) with LINEAR
     interpolation; SEG with NEAREST-NEIGHBOUR at hw=512.
  2. Center-crop / zero-pad every volume to target_hw=448 px (512 for SEG).
  3. Crop all volumes to the segmentation bounding-box with margin=(2, 20, 20)
     (mz, my, mx) voxels — same as _compute_bbox_3d in the reference script.
  4. Per-slice normalization:
       minv = np.percentile(slice, 0)   <- true minimum  (NOT 1st percentile)
       maxv = np.percentile(slice, 99)
       norm  = clip -> subtract min -> divide by (max-min+1e-8) -> x 255
  5. Channel order from model_config.json modalities:
       ["hbv", "adc", "hbv"]  ->  col-0=HBV, col-1=ADC, col-2=HBV

  The three processed volumes are concatenated side-by-side along x so each
  NIfTI slice shows:

        [ HBV (448) | ADC (448) | HBV (448) ]

  Output shape: (n_slices, h_cropped, 448*3) float32, saved as .nii.gz.
  Column cls_input_nifti is written back to the CSV.

Usage (run from repo root):
    /mnt/data9/conda/medgemma/bin/python scripts/generate_classifier_inputs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).resolve().parent.parent
CSV_PATH     = REPO_ROOT / "imgs" / "df_ui_test_4cases.csv"
MODEL_CONFIG = REPO_ROOT / "backend" / "lr_medsiglip_features" / "model_config.json"
OUTPUT_COL   = "cls_input_nifti"

# ---------------------------------------------------------------------------
# Load model config
# ---------------------------------------------------------------------------

with open(MODEL_CONFIG) as fh:
    CFG = json.load(fh)

MODALITIES: list[str] = CFG["modalities"]           # e.g. ["hbv", "adc", "hbv"]
HW: int               = 448                         # MedSigLIP-448 input size
SEG_HW: int           = HW                          # seg loaded at SAME hw as modalities
                                                     # (reference script calls load_resample_crop_seg
                                                     #  with target_hw=config.target_hw=448)
TARGET_SPACING        = (0.3125, 0.3125, 3.0)       # mm (x, y, z)
CROP_MARGIN           = (2, 20, 20)                 # (mz, my, mx) voxels

print("Model config:")
print(f"  modalities  : {MODALITIES}")
print(f"  hw          : {HW}  |  seg_hw : {SEG_HW}")
print(f"  spacing     : {TARGET_SPACING}")
print(f"  crop_margin : {CROP_MARGIN}  (mz, my, mx)")
print(f"  norm        : percentile(0) -> percentile(99) per slice")
print()


# ---------------------------------------------------------------------------
# Steps 1+2: resample + center-crop  (mirrors load_resample_crop exactly)
# ---------------------------------------------------------------------------

def _resample_sitk(img: sitk.Image, interp: int) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size    = img.GetSize()
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


def _center_crop_or_pad(vol: np.ndarray, target_hw: int) -> np.ndarray:
    z, h, w = vol.shape
    out = np.zeros((z, target_hw, target_hw), dtype=vol.dtype)
    h0  = max((h - target_hw) // 2, 0); w0  = max((w - target_hw) // 2, 0)
    h1  = min(h0 + target_hw, h);       w1  = min(w0 + target_hw, w)
    oh0 = max((target_hw - h) // 2, 0); ow0 = max((target_hw - w) // 2, 0)
    out[:, oh0:oh0+(h1-h0), ow0:ow0+(w1-w0)] = vol[:, h0:h1, w0:w1]
    return out


def _load_vol(path: str) -> np.ndarray:
    """Load + resample LINEAR + center-crop to HW."""
    img = sitk.ReadImage(str(path))
    img = _resample_sitk(img, sitk.sitkLinear)
    return _center_crop_or_pad(sitk.GetArrayFromImage(img), HW)


def _load_seg(path: str) -> np.ndarray:
    """Load + resample NEAREST-NEIGHBOUR + center-crop to SEG_HW."""
    img = sitk.ReadImage(str(path))
    img = _resample_sitk(img, sitk.sitkNearestNeighbor)
    return _center_crop_or_pad(sitk.GetArrayFromImage(img), SEG_HW)


# ---------------------------------------------------------------------------
# Step 3: seg bounding-box crop  (mirrors _compute_bbox_3d)
# ---------------------------------------------------------------------------

def _compute_bbox_3d(seg: np.ndarray) -> tuple | None:
    if not np.any(seg > 0):
        return None
    mz, my, mx = CROP_MARGIN
    z_idx, y_idx, x_idx = np.where(seg > 0)
    z0 = max(0,            int(z_idx.min()) - mz)
    z1 = min(seg.shape[0], int(z_idx.max()) + 1 + mz)
    y0 = max(0,            int(y_idx.min()) - my)
    y1 = min(seg.shape[1], int(y_idx.max()) + 1 + my)
    x0 = max(0,            int(x_idx.min()) - mx)
    x1 = min(seg.shape[2], int(x_idx.max()) + 1 + mx)
    return (z0, z1), (y0, y1), (x0, x1)


# ---------------------------------------------------------------------------
# Step 4: per-slice normalization  (mirrors norm_mri + get_norm)
# ---------------------------------------------------------------------------

def _norm_slice(sl: np.ndarray) -> np.ndarray:
    """percentile(0) -> percentile(99) stretch to [0, 255] float32."""
    minv = float(np.percentile(sl, 0))    # true min
    maxv = float(np.percentile(sl, 99))
    sl   = np.clip(sl, minv, maxv).astype(np.float32)
    sl  -= minv
    sl  /= (maxv - minv + 1e-8)
    sl  *= 255.0
    return sl


# ---------------------------------------------------------------------------
# Full pipeline for one patient
# ---------------------------------------------------------------------------

def _build_nifti(path_map: dict[str, str], seg_path: str) -> nib.Nifti1Image:
    # --- load unique modalities ---
    unique_mods = list(dict.fromkeys(MODALITIES))
    vols: dict[str, np.ndarray] = {}
    for mod in unique_mods:
        p = path_map.get(mod, "")
        if not p or not Path(p).exists():
            raise FileNotFoundError(f"Modality '{mod}' path not found: {p!r}")
        print(f"    [{mod}]  {p}")
        vols[mod] = _load_vol(p)
        print(f"          shape after resample+crop: {vols[mod].shape}")

    # --- load seg and compute crop bbox ---
    seg_vol = None
    if seg_path and Path(seg_path).exists():
        print(f"    [seg]  {seg_path}")
        seg_vol = _load_seg(seg_path)
        print(f"          shape: {seg_vol.shape}")

    if seg_vol is not None:
        bbox = _compute_bbox_3d(seg_vol)
        if bbox is not None:
            (z0, z1), (y0, y1), (x0, x1) = bbox
            print(f"    seg bbox (margin={CROP_MARGIN}): "
                  f"z {z0}:{z1}  y {y0}:{y1}  x {x0}:{x1}")
            for mod in unique_mods:
                vols[mod] = vols[mod][z0:z1, y0:y1, x0:x1]
                print(f"    [{mod}]  after seg-crop: {vols[mod].shape}")
        else:
            print("    [seg]  mask empty — seg-crop skipped")
    else:
        print("    [seg]  not found — seg-crop skipped")

    # --- synchronise slice counts ---
    n_slices = min(v.shape[0] for v in vols.values())
    for mod in unique_mods:
        if vols[mod].shape[0] != n_slices:
            vols[mod] = vols[mod][:n_slices]
            print(f"    [{mod}]  truncated to {n_slices} slices")

    # --- per-slice normalisation in MODALITIES order ---
    normed: list[np.ndarray] = []
    for mod in MODALITIES:                  # allows duplicates (e.g. hbv twice)
        v = vols[mod]
        n = np.empty_like(v, dtype=np.float32)
        for s in range(n_slices):
            n[s] = _norm_slice(v[s])
        normed.append(n)

    # --- side-by-side concat along x-axis ---
    concat = np.concatenate(normed, axis=2)   # (z, y, x_total)
    print(f"    concatenated shape: {concat.shape}  "
          f"[{'|'.join(MODALITIES)}]")

    # nibabel expects (x, y, z); our array is (z, y, x) → transpose
    nifti_arr = concat.transpose(2, 1, 0).astype(np.float32)
    sx, sy, sz = TARGET_SPACING
    img = nib.Nifti1Image(nifti_arr, affine=np.diag([sx, sy, sz, 1.0]))
    img.header.set_data_dtype(np.float32)
    img.header.set_xyzt_units("mm")
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} patients from {CSV_PATH}\n")

    out_paths: list[str] = []
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        print(f"[{pid}]")

        path_map = {mod: str(row.get(mod, "")) for mod in dict.fromkeys(MODALITIES)}
        seg_path  = str(row.get("seg", ""))

        t2w_path = str(row.get("t2w", ""))
        if t2w_path and Path(t2w_path).exists():
            out_dir = Path(t2w_path).parent
        else:
            out_dir = REPO_ROOT / "imgs" / pid
            out_dir.mkdir(parents=True, exist_ok=True)

        modality_tag = "_".join(dict.fromkeys(MODALITIES))
        out_path = out_dir / f"{pid}_classifier_input_{modality_tag}.nii.gz"

        try:
            nii = _build_nifti(path_map, seg_path)
            nib.save(nii, str(out_path))
            print(f"  saved {out_path}  ({out_path.stat().st_size // 1024} kB)\n")
            out_paths.append(str(out_path))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {exc}\n", file=sys.stderr)
            out_paths.append("")

    df[OUTPUT_COL] = out_paths
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV updated — '{OUTPUT_COL}' written to {CSV_PATH}")


if __name__ == "__main__":
    main()
