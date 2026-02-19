# MedGemma CDS: A Clinical Decision Support System for Prostate Cancer Treatment Planning

**Team:** Precision Medicine Maastricht University  
**Challenge:** The MedGemma Impact Challenge 2026  
**Task:** AI-assisted, guideline-aligned treatment recommendation for prostate cancer using multiparametric MRI and large language models

---

## 1. Introduction & Problem Statement

Prostate cancer is the most common malignancy in men worldwide, and its management is inherently complex. Clinicians must reconcile multiparametric MRI findings, serum biomarkers (PSA, PSA density), TNM staging, patient comorbidities, and personal preferences before recommending a treatment pathway — all within a constrained consultation window. Guidelines from the PDQ® (Physician Data Query) provide evidence-based recommendations, but translating those guidelines into a personalised, patient-specific plan at the point of care remains a cognitive bottleneck.

This submission introduces a **full-stack Clinical Decision Support (CDS) web application** that fuses two complementary AI models:

1. **MedSigLIP-448** — a vision encoder fine-tuned on medical imaging — driving a five-fold logistic regression ensemble that estimates the probability of clinically significant prostate cancer (csPCa) directly from mpMRI.
2. **MedGemma-1.5-4B-IT** — a medical large language model — that converts the structured patient record and AI predictions into ranked, free-text treatment recommendations, augmented with relevant PDQ guideline passages via Retrieval-Augmented Generation (RAG), and that also serves as an interactive clinical assistant chatbot.

---

## 2. Solution Overview

The system is a single-page React application backed by two independent Python HTTP services. The workflow for each patient case proceeds in three stages:

```
mpMRI volumes (T2W · ADC · HBV)
        │
        ▼
┌─────────────────────────────────┐
│  MedSigLIP Classifier API       │  ← port 8001
│  · Resample → 0.3125×0.3125×3mm│
│  · Seg-based prostate crop      │
│  · Per-slice percentile norm    │
│  · 448-px centre crop           │
│  · google/medsiglip-448 encoder │
│  · Gaussian slice pooling (σ=2) │
│  · 5-fold LR ensemble           │
│  → P(csPCa), uncertainty CI     │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Clinical Inputs (UI form)      │
│  · TNM staging, PSA, age        │
│  · Comorbidities, LUTS          │
│  · Patient preferences          │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  MedGemma Recommendation API    │  ← port 8000
│  · BM25 RAG over PDQ guidelines │
│  · Structured JSON prompt       │
│  · google/medgemma-1.5-4b-it    │
│  → Ranked treatment options     │
│  → Interactive chat assistant   │
└─────────────────────────────────┘
```

---

## 3. System Architecture

### 3.1 Frontend — React + Vite

The UI is built with React 19 and bundled by Vite 7. It consists of six panels arranged in a responsive grid:

| Panel | Component | Role |
|---|---|---|
| **Imaging Review** | `ImagingPanel.jsx` | 4-up NiiVue viewer: T2W · ADC · HBV · MedSigLIP input |
| **MedSigLIP Predictions** | `ModelPredictionsPanel.jsx` | csPCa probability, binary prediction, PSA density, 5-fold uncertainty |
| **Clinical Inputs** | `ClinicalInputsPanel.jsx` | TNM staging, PSA, age, comorbidities, patient preferences |
| **Treatment Recommendations** | `RecommendationPanel.jsx` | Ranked options with rationale and PDQ citations |
| **Clinical Assistant** | `ClinicalAssistantPanel.jsx` | MedGemma-powered free-form chat grounded on the active patient |
| **Top Bar / Hero** | `TopBar.jsx`, `HeroSection.jsx` | Navigation branding |

**State management** is handled entirely in `App.jsx` via React `useState`. All patient form fields and classifier outputs are mirrored into the CSV row and persisted to `localStorage` (key `cds_session_v1`). On navigation between patients, the app asymmetrically:
- **Next patient**: clears the destination's UI fields so a fresh evaluation is prompted.
- **Previous patient**: restores the saved recommendation and classifier results for that patient.

The `ClinicalAssistantPanel` is force-remounted with `key={patientIndex}` to prevent chat history from leaking between patients.

**Volume viewing** uses [NiiVue](https://github.com/niivue/niivue). All four panels stream `.mha` / `.nii.gz` volumes directly from the Vite dev server filesystem; Vite's `fs.strict: false` setting enables serving volumes from absolute paths on disk. The fourth panel shows the exact MedSigLIP model input (HBV|ADC|HBV concatenated side-by-side, seg-cropped, normalised) generated offline and stored as `.nii.gz` per patient.

### 3.2 MedSigLIP Classifier Backend (`classifier_api.py`, port 8001)

A pure-Python `ThreadingHTTPServer` wrapping the following pipeline:

**Preprocessing** (matches the training script exactly):
1. Read each volume with SimpleITK; resample to **0.3125 × 0.3125 × 3.0 mm** isotropic spacing using bilinear interpolation (nearest-neighbour for the segmentation mask).
2. Centre-crop / zero-pad to **448 × 448 px** in the axial plane.
3. Load the prostate segmentation mask at the same 448-px grid with nearest-neighbour resampling.
4. Compute the 3-D bounding box of the mask with a `(mz=2, my=20, mx=20)` voxel margin.
5. Crop all modality volumes to this bounding box, isolating the prostate region.
6. **Per-slice normalisation**: for each axial slice independently, stretch the intensity range `[percentile(0), percentile(99)]` linearly to `[0, 255]` (true minimum, not percentile 1).

**Feature extraction** (per slice, per modality):
- Each normalised slice is passed through `google/medsiglip-448` (`AutoModel.get_image_features`) with the standard image processor.
- Features are L2-normalised (unit sphere).
- Slices are **Gaussian-pooled** along the z-axis with σ = 2, centred at the middle slice.
- Modalities are aggregated by **mean** pooling across `[HBV, ADC, HBV]`, yielding a single 1152-dimensional feature vector.

**Prediction**:
- A 5-fold logistic regression ensemble (trained on 1 254 PI-CAI cases) predicts `P(csPCa)`.
- Uncertainty is reported as: cross-fold standard deviation, predictive entropy, and 95% confidence interval.
- Threshold: **0.52** (optimised on the validation set).

**Endpoints**:
- `GET /health` — liveness probe
- `POST /api/classifier/predict` — single patient (JSON body with file paths)
- `POST /api/classifier/predict_csv` — batch run over the configured CSV

### 3.3 MedGemma Recommendation Backend (`medgemma_api.py`, port 8000)

A `ThreadingHTTPServer` hosting `google/medgemma-1.5-4b-it`. Key design decisions:

**RAG over PDQ guidelines**: The NCI PDQ® prostate cancer treatment document is chunked (900-token chunks, 160-token overlap) at startup. At inference time, a BM25-style term-overlap retrieval selects the top-4 most relevant chunks given the patient's staging, PSA, and predicted risk. These are injected into the prompt as `[PDQ p.X]` citations.

**Structured prompt**: The LLM receives a strict instruction to return only a JSON object matching a schema with `summary`, `options[].title`, `options[].reasoning`, and `options[].description`. A robust JSON extractor handles code-fenced output, trailing commas, and newlines inside strings.

**Retry logic**: If generation exceeds the context budget or produces malformed JSON, the API retries with an increased token budget.

**GPU routing**: The backend detects available CUDA GPUs, tracks failures, and automatically routes to the next available device on CUDA assertion errors.

**Endpoints**:
- `GET /api/medgemma/recommend` — treatment recommendation (structured JSON)
- `POST /api/medgemma/chat` — multi-turn clinical assistant chat, grounded on the active patient record and current recommendation

---

## 4. Screenshots

### 4.1 Imaging Review + MedSigLIP Classifier

The imaging panel shows all three mpMRI modalities alongside the exact model input (HBV|ADC|HBV concatenated after seg-based prostate crop). Clicking **▶ Run MedSigLIP Classifier** calls the classifier API and populates the predictions panel with the csPCa probability, binary classification, PSA density, and 5-fold uncertainty metrics.

![MedSigLIP Image Classifier](dist/imgs/medsiglip%20image%20classifier.png)

### 4.2 MedGemma Treatment Recommendation Engine

After the clinician fills in the staging, PSA, comorbidities, and patient preferences and presses **Get Recommendation**, MedGemma generates a personalised ranked list of treatment options. Each option includes a PDQ-cited rationale and a practical description. The options are rendered as collapsible cards.

![MedGemma Treatment Recommendation Engine](dist/imgs/medgemma%20TRE.png)

### 4.3 MedGemma Clinical Assistant Chatbot

The assistant panel provides a free-form conversational interface grounded on the full patient context (form fields, AI prediction, recommendation already generated). Clinicians can ask follow-up questions such as *"What is the expected continence recovery after prostatectomy for this patient?"* and receive contextualised, markdown-formatted responses.

![MedGemma Clinical Assistant Chatbot](dist/imgs/medgemma%20charbot.png)

---

## 5. Model Performance

The MedSigLIP + logistic regression ensemble was trained and evaluated on the [PI-CAI](https://pi-cai.grand-challenge.org/) dataset (1 254 training+validation / 222 test cases, 5-fold cross-validation).

| Split | AUROC | Average Precision | Accuracy | Recall |
|---|---|---|---|---|
| Train | 0.753 | 0.530 | 0.711 | 0.621 |
| Validation | 0.753 | 0.542 | 0.725 | 0.698 |
| **Test** | **0.767** | **0.582** | **0.716** | **0.651** |

Cross-validation AUROC: **0.756 ± 0.033** (5-fold)

The classifier uses class-weighted logistic regression (L2, C=0.01, SAGA solver) trained on 1 152-dimensional MedSigLIP features, without PCA dimensionality reduction. The decision threshold of **0.52** was chosen to balance precision and recall on the validation set.

---

## 6. Key Technical Decisions

### Exact Preprocessing Fidelity
The preprocessing pipeline in `classifier_api.py` was carefully reverse-engineered from the original training script (`medsiglip_multislice_classification_crop_fast_save.py`). Critical correctness details:
- Seg mask loaded at **hw=448** (same as modalities), not a separate 512-px grid, so bounding box indices are directly applicable.
- Per-**slice** normalisation using true `percentile(0)` (minimum), not a volume-wide `percentile(1)`, ensuring no information loss from intensity clipping before normalisation.
- Channel order `[HBV, ADC, HBV]` matching the model training configuration exactly.

### Session Persistence Without a Database
All user inputs and classifier results are stored as extra columns in the in-memory patient array, serialised to `localStorage` as JSON. On restore, only the 15 UI-editable columns are overlaid onto the fresh CSV row — ensuring that newly-added image path columns (like `cls_input_nifti`) are always up-to-date even after the CSV changes, without requiring a localStorage version bump.

### NiiVue Volume Viewer Integration
Each of the four imaging panels uses the `@niivue/niivue` WebGL viewer embedded in a React component (`NiiViewer.jsx`). Volumes are streamed directly from the filesystem via Vite's dev server. The component resets state on URL changes and guards against null canvas references to prevent the `toUpperCase` crash that occurs when Niivue tries to infer the file format before the DOM is ready.

### RAG-Augmented LLM Inference
MedGemma alone has no explicit PDQ knowledge. The BM25-style retriever injects the 4 most relevant guideline passages into every prompt, allowing the model to cite specific pages (`[PDQ p.X]`) and grounding its recommendations in authoritative clinical guidelines rather than parametric memory.

---

## 7. Reproducibility & Running the System

### Prerequisites
- Conda environment `/mnt/data9/conda/medgemma` with PyTorch, Transformers, SimpleITK, joblib, scikit-learn, pandas, numpy
- Node.js ≥ 20 (via nvm)
- NVIDIA GPU (≥ 24 GB VRAM recommended for MedGemma-4B)
- HuggingFace token with access to `google/medgemma-1.5-4b-it` and `google/medsiglip-448`

### Startup
```bash
# 1. MedGemma recommendation API (port 8000)
nohup /mnt/data9/conda/medgemma/bin/python \
  /path/to/cds/backend/medgemma_api.py > /tmp/medgemma_api.log 2>&1 &

# 2. MedSigLIP classifier API (port 8001)
#    Must run from backend/ so lr_medsiglip_features/ paths resolve
cd /path/to/cds/backend
nohup /mnt/data9/conda/medgemma/bin/python classifier_api.py \
  > /tmp/classifier_api.log 2>&1 &

# 3. Vite dev server (port 5173)
cd /path/to/cds
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

### Pre-generating Classifier Input NIfTIs (optional)
```bash
# Generates HBV|ADC|HBV seg-cropped NIfTIs for the scrollable 4th panel
python scripts/generate_classifier_inputs.py
```

### Project Structure
```
cds/
├── backend/
│   ├── medgemma_api.py          # MedGemma recommendation + chat API
│   ├── classifier_api.py        # MedSigLIP feature extraction + LR ensemble
│   └── lr_medsiglip_features/   # Trained model artifacts (joblib + config)
├── scripts/
│   └── generate_classifier_inputs.py  # Offline NIfTI preview generation
├── src/
│   ├── App.jsx                  # Root component + all state management
│   ├── components/
│   │   ├── ImagingPanel.jsx     # 4-up NiiVue viewer
│   │   ├── NiiViewer.jsx        # Single NiiVue canvas component
│   │   ├── ModelPredictionsPanel.jsx  # csPCa probability + uncertainty
│   │   ├── ClinicalInputsPanel.jsx    # TNM / PSA / comorbidity form
│   │   ├── RecommendationPanel.jsx    # Ranked treatment options
│   │   └── ClinicalAssistantPanel.jsx # MedGemma chat interface
│   └── data/
│       ├── patientCsv.js        # CSV parser + toViewerUrl helper
│       └── initialForm.js       # Default form field values
└── imgs/
    └── df_ui_test_4cases.csv    # Patient metadata + image paths
```

---

## 8. Limitations & Future Work

- **Not for clinical use.** The system is a research prototype. All outputs carry the disclaimer *"Demo outputs only. Not for clinical use."*
- **Scale**: The current demo loads four PI-CAI cases. Extending to a full cohort requires a server-side patient store (e.g. a lightweight SQLite database) rather than in-browser CSV.
- **MedSigLIP fine-tuning**: The vision encoder is used zero-shot (no task-specific fine-tuning of the encoder weights). End-to-end fine-tuning on the full PI-CAI dataset is expected to substantially improve AUROC.
- **Report generation**: A PDF export pathway (combining the imaging panel screenshot, predictions, and recommendation) would improve clinical utility.
- **T2W modality**: The current classifier configuration uses `[HBV, ADC, HBV]` and omits T2W. Adding T2W as a fourth modality may improve sensitivity for low-ADC lesions.
- **Multimodal MedGemma**: Once multimodal (image + text) versions of MedGemma are available, the vision and language reasoning could be unified in a single model rather than a two-stage pipeline.

---

## 9. Conclusion

We presented a complete, end-to-end clinical decision support system for prostate cancer that integrates state-of-the-art medical vision and language models in a clinician-facing web interface. MedSigLIP provides interpretable, quantified imaging biomarkers (csPCa probability with 5-fold uncertainty) while MedGemma translates the full clinical picture into personalised, guideline-grounded treatment recommendations and supports follow-up consultation via a context-aware chatbot. The system is fully open-source, reproducible, and designed with clinical workflow in mind.

---

*Built with: React 19 · Vite 7 · NiiVue · google/medsiglip-448 · google/medgemma-1.5-4b-it · SimpleITK · scikit-learn · PDQ® NCI Guidelines*
