# CDS App (MedGemma Treatment Recommendations)

This app provides a prostate cancer clinical decision support UI with imaging preview, clinical inputs, and treatment recommendations.

## MedGemma Recommendation Feature

- A local Python API endpoint is available at `POST /api/medgemma/recommend`.
- The app requests treatment recommendations from `google/medgemma-1.5-4b-it`.
- The recommendation panel renders the top 3 options.
- Each option has a `Show details` / `Hide details` button.
- If MedGemma is unavailable, the API returns a structured fallback recommendation.

## Run

1. Start the Python API in the notebook environment:

```bash
conda activate /mnt/data9/conda/medgemma
python backend/medgemma_api.py
```

2. Start frontend:

```bash
npm run dev
```

Vite proxies `/api/*` to `http://127.0.0.1:8000` by default.
You can override proxy target:

```bash
VITE_MEDGEMMA_API_TARGET=http://127.0.0.1:8000 npm run dev
```
