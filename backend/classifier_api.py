#!/usr/bin/env python3
"""
Standalone classification API — MedSigLIP mpMRI feature extraction + ensemble prediction.

Endpoints:
  GET  /health                          → liveness check
  POST /api/classifier/predict          → run full pipeline on a CSV row or batch
  POST /api/classifier/predict_csv      → run on all rows of the configured CSV

Usage:
  python classifier_api.py
  # or with custom port:
  CLASSIFIER_API_PORT=8001 python classifier_api.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Paths & constants — mirror the training config exactly
# ---------------------------------------------------------------------------

ARTIFACT_DIR = Path("lr_medsiglip_features")
MODEL_CONFIG_PATH = ARTIFACT_DIR / "model_config.json"
ENSEMBLE_PATH     = ARTIFACT_DIR / "lr_case_csPCa_binary_ensemble.joblib"

# Default CSV to run when /api/classifier/predict_csv is called with no body
DEFAULT_CSV = Path("../imgs/df_ui_test_4cases.csv")

LABEL_COL   = "case_csPCa_binary"
FEATURE_COL = "feature"

HOST = os.environ.get("CLASSIFIER_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("CLASSIFIER_API_PORT", "8001"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("classifier_api")

# ---------------------------------------------------------------------------
# Load model config + ensemble at startup
# ---------------------------------------------------------------------------

def _load_artifacts():
    with open(MODEL_CONFIG_PATH) as f:
        cfg = json.load(f)
    ensemble = joblib.load(ENSEMBLE_PATH)
    LOGGER.info("Loaded ensemble of %d fold-models from %s", len(ensemble), ENSEMBLE_PATH)
    LOGGER.info(
        "Config: modalities=%s  aggregation=%s  sigma=%s  normalize=%s  threshold=%.3f",
        cfg["modalities"], cfg["modality_aggregation"],
        cfg["slice_gaussian_sigma"], cfg["normalize_features"],
        cfg["thresholds"][LABEL_COL],
    )
    return cfg, ensemble


try:
    MODEL_CFG, ENSEMBLE = _load_artifacts()
except Exception as exc:
    LOGGER.error("Failed to load model artifacts: %s", exc)
    raise SystemExit(1) from exc

# Unpack config values used in feature extraction
_MODALITIES        = MODEL_CFG["modalities"]                  # e.g. ["hbv", "adc", "hbv"]
_AGGREGATION       = MODEL_CFG["modality_aggregation"]        # "mean" or "concat"
_SIGMA             = MODEL_CFG["slice_gaussian_sigma"]        # int or None
_NORMALIZE         = MODEL_CFG["normalize_features"]          # bool
_THRESHOLD         = MODEL_CFG["thresholds"][LABEL_COL]       # float
_FEATURE_DIRS      = MODEL_CFG["feature_dirs"]                # dict modality->dir

# ---------------------------------------------------------------------------
# Feature extraction helpers  (adapted from medsiglip_multislice_classification_crop_fast_save.py)
# ---------------------------------------------------------------------------

def _resample(img: sitk.Image, spacing=(0.3125, 0.3125, 3.0), interp=sitk.sitkLinear) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size    = img.GetSize()
    new_size = [int(round(orig_size[i] * (orig_spacing[i] / spacing[i]))) for i in range(3)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(spacing)
    r.SetSize(new_size)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetInterpolator(interp)
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


def _load_vol(path: str, hw: int = 448, interp=sitk.sitkLinear) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    img = _resample(img, interp=interp)
    vol = sitk.GetArrayFromImage(img)
    return _center_crop_pad(vol, hw)


def _load_seg(path: str, hw: int = 448) -> np.ndarray:
    """Load segmentation with nearest-neighbour resampling at the same hw as modalities."""
    img = sitk.ReadImage(str(path))
    img = _resample(img, interp=sitk.sitkNearestNeighbor)
    vol = sitk.GetArrayFromImage(img)
    return _center_crop_pad(vol, hw)


def _compute_bbox_3d(seg: np.ndarray, margin=(2, 20, 20)):
    """Bounding box around seg > 0 with (mz, my, mx) margin. Returns None if mask is empty."""
    if not np.any(seg > 0):
        return None
    mz, my, mx = margin
    z_idx, y_idx, x_idx = np.where(seg > 0)
    z0 = max(0,            int(z_idx.min()) - mz)
    z1 = min(seg.shape[0], int(z_idx.max()) + 1 + mz)
    y0 = max(0,            int(y_idx.min()) - my)
    y1 = min(seg.shape[1], int(y_idx.max()) + 1 + my)
    x0 = max(0,            int(x_idx.min()) - mx)
    x1 = min(seg.shape[2], int(x_idx.max()) + 1 + mx)
    return (z0, z1), (y0, y1), (x0, x1)


def _gaussian_weights(n: int, sigma) -> np.ndarray:
    if sigma is None:
        sigma = max(n / 6.0, 1e-6)
    idx    = np.arange(n, dtype=np.float32)
    center = (n - 1) / 2.0
    w = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
    w /= w.sum()
    return w.astype(np.float32)


def _pool_slices(feat: np.ndarray, sigma) -> np.ndarray:
    """Gaussian-weighted pooling over slice axis → (feature_dim,)"""
    n = feat.shape[0]
    w = _gaussian_weights(n, sigma)
    return (feat * w[:, None]).sum(axis=0)


# ---------------------------------------------------------------------------
# MedSigLIP model — loaded lazily on first feature-extraction request
# ---------------------------------------------------------------------------

_VISION_MODEL   = None
_IMAGE_PROC     = None
_DEVICE         = None


def _ensure_vision_model():
    global _VISION_MODEL, _IMAGE_PROC, _DEVICE
    if _VISION_MODEL is not None:
        return
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image  # noqa: F401 (ensure PIL available)

    model_name = "google/medsiglip-448"
    LOGGER.info("Loading vision model: %s", model_name)
    _IMAGE_PROC  = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    _VISION_MODEL = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _VISION_MODEL = _VISION_MODEL.to(_DEVICE).eval()
    LOGGER.info("Vision model ready on %s", _DEVICE)


def _extract_features_from_paths(t2w_path: str, adc_path: str, hbv_path: str, seg_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline matching medsiglip_multislice_classification_crop_fast_save.py:
      1. Resample all volumes to 0.3125×0.3125×3 mm, center-crop to 448 px.
      2. Load seg with nearest-neighbour at same hw=448; compute bbox with margin=(2,20,20).
      3. Crop all modality volumes to bbox.
      4. Per-slice norm: percentile(0)→percentile(99) stretch to [0,255].
      5. Extract MedSigLIP features per slice, Gaussian-pool, aggregate modalities.
    """
    import torch
    from PIL import Image as PILImage

    _ensure_vision_model()

    path_map = {"t2w": t2w_path, "adc": adc_path, "hbv": hbv_path}

    # -- Step 1: load + resample + center-crop all unique modalities --
    unique_mods = list(dict.fromkeys(_MODALITIES))
    vol_cache: dict[str, np.ndarray] = {}
    for mod in unique_mods:
        vol_cache[mod] = _load_vol(path_map[mod])
        LOGGER.debug("%s shape after resample+crop: %s", mod, vol_cache[mod].shape)

    # -- Step 2: seg-based bbox crop (same hw=448, nearest-neighbour) --
    if seg_path and Path(seg_path).exists():
        try:
            seg_vol = _load_seg(seg_path)
            bbox = _compute_bbox_3d(seg_vol, margin=(2, 20, 20))
            if bbox is not None:
                (z0, z1), (y0, y1), (x0, x1) = bbox
                LOGGER.info("seg bbox: z %d:%d  y %d:%d  x %d:%d", z0, z1, y0, y1, x0, x1)
                for mod in unique_mods:
                    vol_cache[mod] = vol_cache[mod][z0:z1, y0:y1, x0:x1]
                    LOGGER.debug("%s shape after seg-crop: %s", mod, vol_cache[mod].shape)
            else:
                LOGGER.warning("seg mask empty — skipping seg-crop")
        except Exception as exc:
            LOGGER.warning("seg load failed (%s) — skipping seg-crop", exc)
    else:
        LOGGER.warning("seg path not provided or missing — skipping seg-crop")

    # Align slice counts across modalities
    n_slices_map = {mod: vol_cache[mod].shape[0] for mod in unique_mods}
    n_slices_min = min(n_slices_map.values())
    for mod in unique_mods:
        if vol_cache[mod].shape[0] != n_slices_min:
            vol_cache[mod] = vol_cache[mod][:n_slices_min]

    # -- Steps 3+4: per-slice norm → RGB → MedSigLIP features --
    def _norm_slice(sl: np.ndarray) -> np.ndarray:
        """percentile(0) → percentile(99) stretch to [0,255] uint8, per reference script."""
        minv = float(np.percentile(sl, 0))
        maxv = float(np.percentile(sl, 99))
        sl = np.clip(sl, minv, maxv).astype(np.float32)
        sl -= minv
        sl /= (maxv - minv + 1e-8)
        sl *= 255.0
        return sl.astype(np.uint8)

    modality_features: list[np.ndarray] = []
    for mod in _MODALITIES:
        vol = vol_cache[mod]
        n_slices = vol.shape[0]
        slice_feats: list[np.ndarray] = []

        for s_idx in range(n_slices):
            sl  = _norm_slice(vol[s_idx])              # per-slice norm
            rgb = PILImage.fromarray(sl).convert("RGB")
            inputs = _IMAGE_PROC(images=rgb, return_tensors="pt").to(_DEVICE)
            with torch.no_grad():
                out = _VISION_MODEL.get_image_features(**inputs)
            feat = out.squeeze(0).cpu().float().numpy()
            if _NORMALIZE:
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm
            slice_feats.append(feat)

        stacked = np.vstack(slice_feats)               # (n_slices, dim)
        pooled  = _pool_slices(stacked, _SIGMA)        # (dim,)
        modality_features.append(pooled)

    if _AGGREGATION == "concat":
        return np.concatenate(modality_features, axis=0)
    elif _AGGREGATION == "mean":
        return np.mean(modality_features, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {_AGGREGATION}")


# ---------------------------------------------------------------------------
# Ensemble prediction + uncertainty
# ---------------------------------------------------------------------------

def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of a Bernoulli distribution. Max = 1.0 at p=0.5."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _ensemble_predict(X: np.ndarray) -> dict:
    """
    X: (n_samples, feature_dim)

    Returns dict with per-sample arrays:
      mean_prob       — average probability across folds (primary score)
      prediction      — binary label at optimized threshold
      fold_probs      — (n_folds, n_samples) individual fold probabilities

    Uncertainty metrics (all in [0, 1], higher = more uncertain):
      uncertainty_std          — std of fold probabilities (epistemic spread)
      uncertainty_entropy      — entropy of mean_prob  (aleatoric, peaks at p=0.5)
      uncertainty_disagreement — fraction of folds that disagree with majority vote
      uncertainty_ci_width     — width of 95% bootstrap CI across folds (fold_max-fold_min proxy)
    """
    fold_probs_matrix = np.array([m.predict_proba(X)[:, 1] for m in ENSEMBLE])  # (n_folds, n_samples)
    mean_prob = fold_probs_matrix.mean(axis=0)                                   # (n_samples,)
    prediction = (mean_prob >= _THRESHOLD).astype(int)                           # (n_samples,)

    # 1. Epistemic: std of fold probabilities — main uncertainty signal
    unc_std = fold_probs_matrix.std(axis=0)                                      # (n_samples,)

    # 2. Aleatoric: entropy of the mean prediction — peaks at p=0.5 regardless of fold spread
    unc_entropy = _binary_entropy(mean_prob)                                     # (n_samples,)

    # 3. Fold disagreement: fraction of folds whose binary vote differs from ensemble vote
    fold_votes = (fold_probs_matrix >= _THRESHOLD).astype(int)                  # (n_folds, n_samples)
    majority = prediction[np.newaxis, :]                                         # broadcast
    unc_disagreement = (fold_votes != majority).mean(axis=0)                     # (n_samples,)

    # 4. 95% CI width across folds (uses 2.5th and 97.5th percentile of fold probs)
    ci_lo = np.percentile(fold_probs_matrix, 2.5, axis=0)
    ci_hi = np.percentile(fold_probs_matrix, 97.5, axis=0)
    unc_ci_width = ci_hi - ci_lo                                                 # (n_samples,)

    return {
        "mean_prob":               mean_prob,
        "prediction":              prediction,
        "fold_probs":              fold_probs_matrix,
        "uncertainty_std":         unc_std,
        "uncertainty_entropy":     unc_entropy,
        "uncertainty_disagreement":unc_disagreement,
        "uncertainty_ci_width":    unc_ci_width,
        "ci_lo":                   ci_lo,
        "ci_hi":                   ci_hi,
    }


# ---------------------------------------------------------------------------
# High-level pipeline: CSV row → feature → prediction
# ---------------------------------------------------------------------------

def _process_row(row: dict) -> dict:
    """Extract features and predict for a single patient row (dict)."""
    pid = row.get("patient_id", "unknown")
    LOGGER.info("Processing patient %s", pid)

    feat = _extract_features_from_paths(
        t2w_path=row["t2w"],
        adc_path=row["adc"],
        hbv_path=row["hbv"],
        seg_path=row.get("seg", ""),
    )

    X   = feat[np.newaxis, :]   # (1, feature_dim)
    res = _ensemble_predict(X)

    return {
        "patient_id":               pid,
        # Primary prediction
        "csPC_probability":         float(res["mean_prob"][0]),
        "csPC_prediction":          int(res["prediction"][0]),
        "threshold_used":           float(_THRESHOLD),
        # Per-fold probabilities (useful for plotting)
        "fold_probabilities":       [float(p) for p in res["fold_probs"][:, 0]],
        # Uncertainty scores — all normalised to [0, 1]
        # Higher = more uncertain
        "uncertainty_std":          float(res["uncertainty_std"][0]),
        "uncertainty_entropy":      float(res["uncertainty_entropy"][0]),
        "uncertainty_disagreement": float(res["uncertainty_disagreement"][0]),
        "uncertainty_ci_width":     float(res["uncertainty_ci_width"][0]),
        "uncertainty_ci_lo":        float(res["ci_lo"][0]),
        "uncertainty_ci_hi":        float(res["ci_hi"][0]),
    }


def _create_preview(t2w_path: str, adc_path: str, hbv_path: str) -> str:
    """
    Generate a side-by-side montage of the center slice of each modality,
    preprocessed exactly as MedSigLIP model input (resample → center-crop 448 →
    1st-99th percentile normalise).  Returns a data-URI PNG string.
    """
    import base64
    import io

    from PIL import Image as PILImage, ImageDraw

    TILE = 224          # display size (pixels) per modality
    GAP  = 6            # pixels between tiles
    LABEL_H = 18        # pixel height reserved for text label above each tile
    labels = ["T2W", "ADC", "HBV"]
    paths  = [t2w_path, adc_path, hbv_path]

    tiles: list[PILImage.Image] = []
    for path in paths:
        if path and Path(path).exists():
            try:
                vol = _load_vol(path, hw=448)          # same preprocessing as model
                n   = vol.shape[0]
                sl  = vol[n // 2].astype(np.float32)   # center slice
                lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
                sl  = np.clip(sl, lo, hi)
                if hi > lo:
                    sl = ((sl - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    sl = np.zeros_like(sl, dtype=np.uint8)
                tile = PILImage.fromarray(sl, mode="L").resize(
                    (TILE, TILE), PILImage.LANCZOS
                )
            except Exception as exc:
                LOGGER.warning("Preview failed for %s: %s", path, exc)
                tile = PILImage.new("L", (TILE, TILE), 0)
        else:
            tile = PILImage.new("L", (TILE, TILE), 0)
        tiles.append(tile)

    total_w = len(tiles) * TILE + (len(tiles) - 1) * GAP
    total_h = TILE + LABEL_H
    montage = PILImage.new("L", (total_w, total_h), color=0)
    draw    = ImageDraw.Draw(montage)

    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x = i * (TILE + GAP)
        montage.paste(tile, (x, LABEL_H))
        # Draw label centred above the tile (approximate char width = 7 px)
        text_w = len(label) * 7
        draw.text((x + (TILE - text_w) // 2, 3), label, fill=200)

    buf = io.BytesIO()
    montage.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def run_csv_pipeline(csv_path: str | Path) -> pd.DataFrame:
    """
    Run the full pipeline on every row in csv_path.
    Returns the input DataFrame with added columns:
        csPC_probability, csPC_prediction
    """
    df = pd.read_csv(csv_path)
    LOGGER.info("Running pipeline on %d patients from %s", len(df), csv_path)

    results = []
    for _, row in df.iterrows():
        try:
            res = _process_row(row.to_dict())
        except Exception as exc:
            LOGGER.warning("Failed patient %s: %s", row.get("patient_id", "?"), exc)
            res = {
                "patient_id":       row.get("patient_id", "?"),
                "csPC_probability": None,
                "csPC_prediction":  None,
                "threshold_used":   float(_THRESHOLD),
                "error":            str(exc),
            }
        results.append(res)

    res_df = pd.DataFrame(results)
    merged = df.merge(res_df.drop(columns=["threshold_used"], errors="ignore"),
                      on="patient_id", how="left")
    return merged


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _json_response(handler: BaseHTTPRequestHandler, code: int, body: Any) -> None:
    payload = json.dumps(body, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(payload)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8")) if raw else {}


class ClassifierHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        LOGGER.debug(fmt, *args)

    def do_OPTIONS(self) -> None:  # noqa: N802  CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, 200, {
                "status":      "ok",
                "model":       str(ENSEMBLE_PATH),
                "n_folds":     len(ENSEMBLE),
                "modalities":  _MODALITIES,
                "aggregation": _AGGREGATION,
                "threshold":   _THRESHOLD,
            })
        else:
            _json_response(self, 404, {"error": f"Unknown path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/classifier/predict":
                # Body: single patient dict with keys t2w, adc, hbv, seg, patient_id
                body = _read_body(self)
                result = _process_row(body)
                _json_response(self, 200, result)

            elif self.path == "/api/classifier/preview":
                # Body: {t2w, adc, hbv} – file paths on disk
                body = _read_body(self)
                image_data = _create_preview(
                    t2w_path=body.get("t2w", ""),
                    adc_path=body.get("adc", ""),
                    hbv_path=body.get("hbv", ""),
                )
                _json_response(self, 200, {"image": image_data})

            elif self.path == "/api/classifier/predict_csv":
                # Body (optional): {"csv_path": "/path/to/file.csv"}
                body    = _read_body(self)
                csv_path = body.get("csv_path", str(DEFAULT_CSV))
                if not Path(csv_path).exists():
                    _json_response(self, 400, {"error": f"CSV not found: {csv_path}"})
                    return
                result_df = run_csv_pipeline(csv_path)
                # Save alongside the input CSV
                out_path = Path(csv_path).with_stem(Path(csv_path).stem + "_predictions")
                result_df.to_csv(out_path, index=False)
                LOGGER.info("Predictions saved to %s", out_path)
                _json_response(self, 200, {
                    "csv_path":        csv_path,
                    "output_csv":      str(out_path),
                    "n_patients":      len(result_df),
                    "predictions":     result_df[[
                        "patient_id",
                        "csPC_probability",
                        "csPC_prediction",
                        "uncertainty_std",
                        "uncertainty_entropy",
                        "uncertainty_disagreement",
                        "uncertainty_ci_width",
                        "uncertainty_ci_lo",
                        "uncertainty_ci_hi",
                    ]].to_dict(orient="records"),
                })

            else:
                _json_response(self, 404, {"error": f"Unknown path: {self.path}"})

        except Exception:
            tb = traceback.format_exc()
            LOGGER.error("Unhandled error:\n%s", tb)
            _json_response(self, 500, {"error": tb})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), ClassifierHandler)
    LOGGER.info("Classifier API listening on http://%s:%s", HOST, PORT)
    LOGGER.info("  GET  /health")
    LOGGER.info("  POST /api/classifier/predict        — single patient (JSON body)")
    LOGGER.info("  POST /api/classifier/preview        — center-slice montage PNG")
    LOGGER.info("  POST /api/classifier/predict_csv   — full CSV pipeline")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
        server.shutdown()
