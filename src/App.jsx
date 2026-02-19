import { useEffect, useMemo, useState } from "react";
import "./App.css";
import AppFooter from "./components/AppFooter";
import ClinicalAssistantPanel from "./components/ClinicalAssistantPanel";
import ClinicalInputsPanel from "./components/ClinicalInputsPanel";
import HeroSection from "./components/HeroSection";
import ImagingPanel from "./components/ImagingPanel";
import ModelPredictionsPanel from "./components/ModelPredictionsPanel";
import RecommendationPanel from "./components/RecommendationPanel";
import TopBar from "./components/TopBar";
import { initialForm } from "./data/initialForm";
import { loadPatientsFromCsv, serializeCsv } from "./data/patientCsv";

// Map: form field name → CSV column name (covers ALL UI inputs)
const FIELD_TO_CSV = {
  tStage: "clinical_t_stage",
  nStage: "clinical_n_stage",
  mStage: "clinical_m_stage",
  psa: "clinical_psa",
  performance: "clinical_performance",
  age: "clinical_age",
  familyHistory: "ui_family_history",
  lowerUrinarySymptoms: "ui_lower_urinary_symptoms",
  priorBiopsy: "ui_prior_biopsy",
  comorbidities: "ui_comorbidities", // stored as pipe-separated string
  preferencesSexualFunction: "ui_pref_sexual_function",
  preferencesUrinaryContinence: "ui_pref_urinary_continence",
  preferencesTreatmentIntensity: "ui_pref_treatment_intensity",
  preferencesFollowUpBurden: "ui_pref_follow_up_burden",
  additionalPreferences: "ui_additional_preferences",
};

// All CSV columns that correspond to UI inputs + persisted outputs
const UI_CSV_COLUMNS = [
  ...Object.values(FIELD_TO_CSV),
  "ui_recommendation",
  // MedSigLIP classifier outputs (written when the user presses Run)
  "cls_probability",
  "cls_prediction",
  "cls_uncertainty_std",
  "cls_uncertainty_entropy",
  "cls_uncertainty_ci_lo",
  "cls_uncertainty_ci_hi",
];

// Read all form fields from a patient row (with initialForm fallbacks)
function formFromPatient(patient) {
  const rawComorbidities = patient?.ui_comorbidities;
  const comorbidities =
    rawComorbidities != null && rawComorbidities !== ""
      ? rawComorbidities.split("|").filter(Boolean)
      : [...initialForm.comorbidities];
  return {
    tStage: patient?.clinical_t_stage || initialForm.tStage,
    nStage: patient?.clinical_n_stage || initialForm.nStage,
    mStage: patient?.clinical_m_stage || initialForm.mStage,
    psa: patient?.clinical_psa || patient?.psa || initialForm.psa,
    performance: patient?.clinical_performance || initialForm.performance,
    age: patient?.clinical_age || patient?.patient_age || initialForm.age,
    familyHistory: patient?.ui_family_history || initialForm.familyHistory,
    lowerUrinarySymptoms: patient?.ui_lower_urinary_symptoms || initialForm.lowerUrinarySymptoms,
    priorBiopsy: patient?.ui_prior_biopsy || initialForm.priorBiopsy,
    comorbidities,
    preferencesSexualFunction: patient?.ui_pref_sexual_function || initialForm.preferencesSexualFunction,
    preferencesUrinaryContinence: patient?.ui_pref_urinary_continence || initialForm.preferencesUrinaryContinence,
    preferencesTreatmentIntensity: patient?.ui_pref_treatment_intensity || initialForm.preferencesTreatmentIntensity,
    preferencesFollowUpBurden: patient?.ui_pref_follow_up_burden || initialForm.preferencesFollowUpBurden,
    additionalPreferences: patient?.ui_additional_preferences ?? initialForm.additionalPreferences,
  };
}

// Reset all UI-editable CSV columns on a patient row to defaults.
// Clinical overrides are cleared so formFromPatient falls back to original
// CSV values (e.g. patient.psa, patient.patient_age).
function resetPatientFields(patient) {
  return {
    ...patient,
    clinical_t_stage: "",
    clinical_n_stage: "",
    clinical_m_stage: "",
    clinical_psa: "",
    clinical_performance: "",
    clinical_age: "",
    ui_family_history: initialForm.familyHistory,
    ui_lower_urinary_symptoms: initialForm.lowerUrinarySymptoms,
    ui_prior_biopsy: initialForm.priorBiopsy,
    ui_comorbidities: initialForm.comorbidities.join("|"),
    ui_pref_sexual_function: initialForm.preferencesSexualFunction,
    ui_pref_urinary_continence: initialForm.preferencesUrinaryContinence,
    ui_pref_treatment_intensity: initialForm.preferencesTreatmentIntensity,
    ui_pref_follow_up_burden: initialForm.preferencesFollowUpBurden,
    ui_additional_preferences: initialForm.additionalPreferences,
    ui_recommendation: "",
    // Classifier outputs – cleared on reset so a fresh run is always possible
    cls_probability: "",
    cls_prediction: "",
    cls_uncertainty_std: "",
    cls_uncertainty_entropy: "",
    cls_uncertainty_ci_lo: "",
    cls_uncertainty_ci_hi: "",
  };
}

// localStorage key – bump the version suffix if the schema changes
const STORAGE_KEY = "cds_session_v1";

// Try to restore the session from localStorage.
// Falls back to CSV + defaults when no saved state exists or the patient list changed.
function loadInitialPatients(basePatients) {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const { patientIds, patients: saved } = JSON.parse(raw);
      const currentIds = basePatients.map((p) => p.patient_id).join(",");
      if (
        patientIds === currentIds &&
        Array.isArray(saved) &&
        saved.length === basePatients.length
      ) {
        // Always start from the *fresh* base CSV row so that newly-added
        // columns (e.g. cls_input_nifti, image paths) are never stale.
        // Only overlay the UI-editable columns from the saved session.
        return basePatients.map((basePt, i) => {
          const savedPt = saved[i] ?? {};
          const uiOverrides = Object.fromEntries(
            UI_CSV_COLUMNS
              .filter((col) => savedPt[col] !== undefined && savedPt[col] !== null)
              .map((col) => [col, savedPt[col]])
          );
          return { ...basePt, ...uiOverrides };
        });
      }
    }
  } catch {
    // ignore malformed cache
  }
  // First visit or patient list changed: apply defaults to every patient
  return basePatients.map(resetPatientFields);
}

export default function App() {
  const { headers: baseHeaders, rows: basePatients } = useMemo(() => loadPatientsFromCsv(), []);
  const [patients, setPatients] = useState(() => loadInitialPatients(basePatients));

  // Auto-save the full patients array to localStorage on every change
  useEffect(() => {
    try {
      const patientIds = basePatients.map((p) => p.patient_id).join(",");
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ patientIds, patients }));
    } catch {
      // ignore storage quota errors
    }
  }, [patients, basePatients]);
  const [patientIndex, setPatientIndex] = useState(0);
  const [result, setResult] = useState(null);
  const currentPatient = patients[patientIndex];
  // All form state lives in the patients array – no separate localForm needed
  const form = formFromPatient(currentPatient);

  const csvHeaders = useMemo(() => {
    const merged = [...baseHeaders];
    UI_CSV_COLUMNS.forEach((col) => {
      if (!merged.includes(col)) merged.push(col);
    });
    return merged;
  }, [baseHeaders]);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [isClassifying, setIsClassifying] = useState(false);
  const [classifierError, setClassifierError] = useState("");

  // Derive live classifier result from patient row (null if classifier hasn't run yet)
  const clsProbabilityRaw = Number.parseFloat(currentPatient?.cls_probability);
  const classifierResult = Number.isFinite(clsProbabilityRaw)
    ? {
        probability: clsProbabilityRaw,
        prediction: String(currentPatient.cls_prediction) === "1",
        uncertaintyStd: Number.parseFloat(currentPatient?.cls_uncertainty_std),
        uncertaintyEntropy: Number.parseFloat(currentPatient?.cls_uncertainty_entropy),
        ciLo: Number.parseFloat(currentPatient?.cls_uncertainty_ci_lo),
        ciHi: Number.parseFloat(currentPatient?.cls_uncertainty_ci_hi),
      }
    : null;

  const csvProbability = Number.parseFloat(currentPatient?.case_csPCa_binary);
  const csvVolume = Number.parseFloat(currentPatient?.prostate_volume);
  const modelPrediction = {
    // null until the classifier has been run for this patient
    csPcaProbability: classifierResult ? classifierResult.probability : null,
    predictedProstateVolumeMl: Number.isFinite(csvVolume) && csvVolume > 0 ? csvVolume : 34,
  };
  const csPcaPrediction =
    modelPrediction.csPcaProbability === null
      ? null
      : modelPrediction.csPcaProbability >= 0.5 ? "Yes (>= 0.5 threshold)" : "No (< 0.5 threshold)";
  const csvPsaDensity = Number.parseFloat(currentPatient?.psad);
  const psaValue = Number.parseFloat(form.psa);
  const psaDensity = Number.isFinite(csvPsaDensity)
    ? csvPsaDensity.toFixed(3)
    : Number.isFinite(psaValue) && modelPrediction.predictedProstateVolumeMl > 0
      ? (psaValue / modelPrediction.predictedProstateVolumeMl).toFixed(3)
      : "";
  const casePatient = {
    ...form,
    psaDensity,
  };
  const caseModelPrediction = {
    ...modelPrediction,
    csPcaPrediction,
    csPcaDefinition:
      "Clinically significant prostate cancer: Gleason score 3+4 or higher (ISUP grade group >=2).",
  };

  const handleChange = (event) => {
    const { name, value, type, checked } = event.target;
    const csvCol = FIELD_TO_CSV[name];
    if (!csvCol) return;
    setPatients((prev) =>
      prev.map((patient, index) => {
        if (index !== patientIndex) return patient;
        if (type === "checkbox") {
          // comorbidities: stored as pipe-separated string in the CSV column
          const current = patient[csvCol]
            ? patient[csvCol].split("|").filter(Boolean)
            : [];
          const next = checked
            ? current.includes(value) ? current : [...current, value]
            : current.filter((item) => item !== value);
          return { ...patient, [csvCol]: next.join("|") };
        }
        return { ...patient, [csvCol]: value };
      })
    );
  };

  const handleRunClassifier = async () => {
    setIsClassifying(true);
    setClassifierError("");
    try {
      const response = await fetch("/api/classifier/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: currentPatient.patient_id,
          t2w: currentPatient.t2w,
          adc: currentPatient.adc,
          hbv: currentPatient.hbv,
          seg: currentPatient.seg || "",
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        setClassifierError(payload?.error || `Classifier failed: ${response.status}`);
        return;
      }
      // Persist classifier results into the patient row (auto-saved to localStorage + CSV)
      setPatients((prev) =>
        prev.map((patient, index) =>
          index === patientIndex
            ? {
                ...patient,
                cls_probability: String(payload.csPC_probability),
                cls_prediction: String(payload.csPC_prediction),
                cls_uncertainty_std: String(payload.uncertainty_std),
                cls_uncertainty_entropy: String(payload.uncertainty_entropy),
                cls_uncertainty_ci_lo: String(payload.uncertainty_ci_lo),
                cls_uncertainty_ci_hi: String(payload.uncertainty_ci_hi),
              }
            : patient
        )
      );
    } catch (err) {
      setClassifierError(`Classifier unavailable: ${err.message}`);
    } finally {
      setIsClassifying(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError("");
    try {
      const response = await fetch("/api/medgemma/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient: casePatient,
          modelPrediction: caseModelPrediction,
        }),
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const reason = payload?.error || `API request failed: ${response.status}`;
        if (payload?.parse_error && typeof payload?.rawOutput === "string" && payload.rawOutput.trim()) {
          setResult({
            parseError: true,
            summary: "MedGemma response could not be parsed as strict JSON.",
            fullDetails: payload.rawOutput,
            rawOutput: payload.rawOutput,
            options: [],
            structuredDetails: {},
            note: "Raw model output is shown below for debugging.",
          });
          setError(`MedGemma parsing error: ${reason}`);
          return;
        }
        setResult(null);
        setError(`MedGemma recommendation failed: ${reason}`);
        return;
      }

      // Persist recommendation into the patient row so it survives navigation
      setPatients((prev) =>
        prev.map((patient, index) =>
          index === patientIndex
            ? { ...patient, ui_recommendation: JSON.stringify(payload) }
            : patient
        )
      );
      setResult(payload);
    } catch (requestError) {
      setResult(null);
      setError(`MedGemma recommendation unavailable: ${requestError.message}`);
      console.error(requestError);
    } finally {
      setIsLoading(false);
    }
  };

  // NEXT: reset ALL fields (form, classifier, recommendation) for the destination patient.
  const handleNextPatient = () => {
    const nextIndex = Math.min(patientIndex + 1, patients.length - 1);
    if (nextIndex === patientIndex) return;
    setPatients((prev) =>
      prev.map((patient, index) =>
        index === nextIndex ? resetPatientFields(patient) : patient
      )
    );
    setResult(null);
    setError("");
    setClassifierError("");
    setIsClassifying(false);
    setIsLoading(false);
    setPatientIndex(nextIndex);
  };

  // PREVIOUS: restore saved state for the previous patient (no reset).
  const handlePreviousPatient = () => {
    const prevIndex = Math.max(patientIndex - 1, 0);
    if (prevIndex === patientIndex) return;
    const savedRec = patients[prevIndex]?.ui_recommendation;
    setResult(savedRec ? JSON.parse(savedRec) : null);
    setError("");
    setClassifierError("");
    setIsClassifying(false);
    setIsLoading(false);
    setPatientIndex(prevIndex);
  };

  const handleSaveCsv = () => {
    const content = serializeCsv(csvHeaders, patients);
    const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "df_ui_test_4cases_with_clinical.csv";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="page">
      <TopBar />
      <HeroSection />

      <main className="grid">
        <ImagingPanel
          patient={currentPatient}
          patientIndex={patientIndex}
          patientCount={patients.length}
          onPreviousPatient={handlePreviousPatient}
          onNextPatient={handleNextPatient}
          isFirstPatient={patientIndex <= 0}
          isLastPatient={patientIndex >= Math.max(patients.length - 1, 0)}
        />
        <ModelPredictionsPanel
          modelPrediction={modelPrediction}
          csPcaPrediction={csPcaPrediction}
          psaDensity={psaDensity}
          classifierResult={classifierResult}
          isClassifying={isClassifying}
          onRunClassifier={handleRunClassifier}
          classifierError={classifierError}
        />
        <ClinicalInputsPanel
          form={form}
          onChange={handleChange}
          onSubmit={handleSubmit}
          psaDensity={psaDensity}
          predictedProstateVolumeMl={modelPrediction.predictedProstateVolumeMl}
          onSaveCsv={handleSaveCsv}
          patientId={currentPatient?.patient_id}
        />
        <RecommendationPanel result={result} isLoading={isLoading} error={error} />
        <ClinicalAssistantPanel
          key={patientIndex}
          patient={casePatient}
          modelPrediction={caseModelPrediction}
          recommendation={result || {}}
        />
      </main>

      <AppFooter />
    </div>
  );
}
