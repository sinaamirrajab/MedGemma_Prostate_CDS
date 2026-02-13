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
import { modelSnapshot } from "./data/modelSnapshot";
import { recommendationTemplate } from "./data/recommendationTemplate";

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setResult(recommendationTemplate);
  };

  return (
    <div className="page">
      <TopBar />
      <HeroSection />

      <main className="grid">
        <ImagingPanel />
        <ModelPredictionsPanel modelSnapshot={modelSnapshot} />
        <ClinicalInputsPanel form={form} onChange={handleChange} onSubmit={handleSubmit} />
        <RecommendationPanel result={result} />
        <ClinicalAssistantPanel />
      </main>

      <AppFooter />
    </div>
  );
}
