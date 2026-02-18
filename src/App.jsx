import { useState } from "react";
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

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const modelPrediction = {
    csPcaProbability: 0.7,
    predictedProstateVolumeMl: 34,
  };
  const csPcaPrediction =
    modelPrediction.csPcaProbability >= 0.5 ? "Yes (>= 0.5 threshold)" : "No (< 0.5 threshold)";
  const psaValue = Number.parseFloat(form.psa);
  const psaDensity =
    Number.isFinite(psaValue) && modelPrediction.predictedProstateVolumeMl > 0
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
    setForm((prev) => {
      if (type === "checkbox") {
        const current = Array.isArray(prev[name]) ? prev[name] : [];
        const nextValues = checked
          ? current.includes(value)
            ? current
            : [...current, value]
          : current.filter((item) => item !== value);
        return { ...prev, [name]: nextValues };
      }
      return { ...prev, [name]: value };
    });
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

      setResult(payload);
    } catch (requestError) {
      setResult(null);
      setError(`MedGemma recommendation unavailable: ${requestError.message}`);
      console.error(requestError);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <TopBar />
      <HeroSection />

      <main className="grid">
        <ImagingPanel />
        <ModelPredictionsPanel
          modelPrediction={modelPrediction}
          csPcaPrediction={csPcaPrediction}
          psaDensity={psaDensity}
        />
        <ClinicalInputsPanel
          form={form}
          onChange={handleChange}
          onSubmit={handleSubmit}
          psaDensity={psaDensity}
          predictedProstateVolumeMl={modelPrediction.predictedProstateVolumeMl}
        />
        <RecommendationPanel result={result} isLoading={isLoading} error={error} />
        <ClinicalAssistantPanel
          patient={casePatient}
          modelPrediction={caseModelPrediction}
          recommendation={result || {}}
        />
      </main>

      <AppFooter />
    </div>
  );
}
