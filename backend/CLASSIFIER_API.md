# Classifier API — Documentation

Standalone HTTP backend for **clinically significant prostate cancer (csPCa) classification**
from biparametric MRI (bpMRI) using MedSigLIP features + 5-fold ensemble prediction.

---

## Architecture Overview

```
mpMRI volumes (T2W / ADC / HBV)
        │
        ▼
MedSigLIP per-slice feature extraction   ← "google/medsiglip-448"
        │
        ▼
Gaussian slice pooling  →  modality aggregation (mean)
        │
        ▼
5-fold ensemble (StandardScaler → PCA → LR)   ← lr_case_csPCa_binary_ensemble.joblib
        │
        ▼
mean probability  +  uncertainty metrics  +  binary prediction
```

Model config is read from `model_config.json` at startup — no hardcoded hyperparameters
in the API itself.

---

## Prerequisites

### 1. Conda environment
The API requires the `medgemma` conda environment:
```bash
conda activate medgemma
```

Required packages (already present in the environment):
- `torch`, `transformers`
- `scikit-learn`, `joblib`
- `SimpleITK`, `numpy`, `pandas`, `Pillow`

### 2. Model artifacts
Must exist at:
```
/mnt/data9/projects/medgemma_challenge/medsiglib/results/lr_medsiglip_features_bpMRI_ft_concat_mpMRI/
  ├── model_config.json                     ← hyperparameters, thresholds, modality config
  └── lr_case_csPCa_binary_ensemble.joblib  ← list of 5 trained sklearn Pipelines
```

---

## Starting the Server

```bash
cd /mnt/data9/projects/medgemma_challenge_UI/cds/backend
conda activate medgemma
python classifier_api.py
```

The server starts on **port 8001** by default:
```
2026-02-19 12:00:00 [INFO] classifier_api — Classifier API listening on http://0.0.0.0:8001
```

### Custom port / host
```bash
CLASSIFIER_API_PORT=9000 CLASSIFIER_API_HOST=127.0.0.1 python classifier_api.py
```

### Background (nohup)
```bash
nohup python classifier_api.py > ./log/classifier_api.log 2>&1 &
```

---

## Endpoints

### `GET /health`
Liveness check. Returns server status and loaded model config.

**Request:**
```bash
curl http://localhost:8001/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "/mnt/.../lr_case_csPCa_binary_ensemble.joblib",
  "n_folds": 5,
  "modalities": ["hbv", "adc", "hbv"],
  "aggregation": "mean",
  "threshold": 0.52
}
```

---

### `POST /api/classifier/predict`
Run the full pipeline on a **single patient**. Provide paths to the MRI files in the JSON body.

**Request:**
```bash
curl -X POST http://localhost:8001/api/classifier/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "11375",
    "t2w": "/mnt/data9/data/picai/.../11375_1001398_t2w.mha",
    "adc": "/mnt/data9/data/picai/.../11375_1001398_adc.mha",
    "hbv": "/mnt/data9/data/picai/.../11375_1001398_hbv.mha",
    "seg": "/mnt/data9/projects/medgemma_challenge/picai_labels/.../11375_1001398.nii.gz"
  }'
```

**Required fields:** `t2w`, `adc`, `hbv` (absolute paths to .mha or .nii.gz files)  
**Optional fields:** `patient_id`, `seg`

**Response:**
```json
{
  "patient_id": "11375",

  "csPC_probability": 0.734,
  "csPC_prediction": 1,
  "threshold_used": 0.52,

  "fold_probabilities": [0.71, 0.78, 0.69, 0.76, 0.73],

  "uncertainty_std": 0.033,
  "uncertainty_entropy": 0.872,
  "uncertainty_disagreement": 0.0,
  "uncertainty_ci_width": 0.091,
  "uncertainty_ci_lo": 0.693,
  "uncertainty_ci_hi": 0.784
}
```

---

### `POST /api/classifier/predict_csv`
Run the pipeline on **all patients** in a CSV file.

**Request (default CSV):**
```bash
curl -X POST http://localhost:8001/api/classifier/predict_csv
```

**Request (custom CSV):**
```bash
curl -X POST http://localhost:8001/api/classifier/predict_csv \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "/mnt/data9/projects/medgemma_challenge/df_ui_test_4cases.csv"}'
```

The CSV must contain columns: `patient_id`, `t2w`, `adc`, `hbv`, `seg`

**Response:**
```json
{
  "csv_path": "/mnt/.../df_ui_test_4cases.csv",
  "output_csv": "/mnt/.../df_ui_test_4cases_predictions.csv",
  "n_patients": 4,
  "predictions": [
    {
      "patient_id": "11375",
      "csPC_probability": 0.734,
      "csPC_prediction": 1,
      "uncertainty_std": 0.033,
      "uncertainty_entropy": 0.872,
      "uncertainty_disagreement": 0.0,
      "uncertainty_ci_width": 0.091,
      "uncertainty_ci_lo": 0.693,
      "uncertainty_ci_hi": 0.784
    },
    ...
  ]
}
```

Predictions are also **saved to disk** at `<input_csv>_predictions.csv`.

---

## Response Fields Reference

### Prediction fields

| Field | Type | Description |
|---|---|---|
| `csPC_probability` | float [0–1] | Mean probability across all 5 fold models |
| `csPC_prediction` | int (0 or 1) | Binary label: 1 = clinically significant PCa, 0 = no csPCa |
| `threshold_used` | float | Decision threshold (0.52, optimised on OOF F1) |
| `fold_probabilities` | float[5] | Individual probability from each of the 5 fold models |

### Uncertainty fields

All uncertainty scores are in **[0, 1]**. Higher = more uncertain.

| Field | Description | When high |
|---|---|---|
| `uncertainty_std` | Std of the 5 fold probabilities — **primary uncertainty score** | Fold models strongly disagree |
| `uncertainty_entropy` | Shannon entropy of `csPC_probability` — peaks at p = 0.5 | Prediction is close to the decision boundary |
| `uncertainty_disagreement` | Fraction of folds whose binary vote differs from the majority | At least 1 fold "flipped" — flag for clinical review |
| `uncertainty_ci_width` | Width of 95% CI across folds (p97.5 − p2.5) | Large spread in fold estimates |
| `uncertainty_ci_lo` | Lower bound of 95% CI | — |
| `uncertainty_ci_hi` | Upper bound of 95% CI | — |

**Interpretation guide:**

```
uncertainty_std < 0.05  → High confidence  (folds agree)
uncertainty_std 0.05–0.15 → Moderate uncertainty
uncertainty_std > 0.15  → Low confidence   (flag for review)

uncertainty_disagreement > 0  → At least one fold contradicts the prediction
uncertainty_disagreement = 0.4+ → Majority is slim — borderline case
```

---

## Integration with the UI Backend

Both APIs can run simultaneously on different ports:

```bash
# Terminal 1 — MedGemma recommendation API
conda activate medgemma
python medgemma_api.py          # port 8000

# Terminal 2 — Classifier API
conda activate medgemma
python classifier_api.py        # port 8001
```

From the frontend (React/Vite), call:
```js
// Single patient prediction
const res = await fetch("http://localhost:8001/api/classifier/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ patient_id, t2w, adc, hbv, seg })
});
const data = await res.json();
// data.csPC_probability, data.uncertainty_std, etc.

// Batch CSV prediction
const res = await fetch("http://localhost:8001/api/classifier/predict_csv", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ csv_path: "/path/to/patients.csv" })
});
```

---

## Error Handling

If a patient fails (missing file, loading error), the API returns a partial result
with `null` probability and an `error` field rather than crashing the whole batch:

```json
{
  "patient_id": "99999",
  "csPC_probability": null,
  "csPC_prediction": null,
  "error": "FileNotFoundError: /mnt/.../99999_t2w.mha"
}
```

HTTP error codes:
- `400` — CSV file not found
- `404` — Unknown endpoint
- `500` — Unhandled server error (full traceback in response body)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: joblib` | Run with `conda activate medgemma` |
| `Failed to load model artifacts` | Check `ARTIFACT_DIR` path exists and contains both `.json` and `.joblib` files |
| Vision model download on first request | Ensure HuggingFace credentials are set: `huggingface-cli login` |
| Slow first request | Vision model loads on first call (~30s on GPU, ~2min on CPU) — `GET /health` to warm up |
| CORS error from frontend | Already handled — all responses include `Access-Control-Allow-Origin: *` |
