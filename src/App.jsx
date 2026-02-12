import { useEffect, useMemo, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import "./App.css";

const initialForm = {
  tStage: "T2",
  nStage: "N0",
  mStage: "M0",
  psa: "8.6",
  performance: "ECOG 0",
  prostateVolume: "42",
  age: "66",
};

const NiiViewer = ({ title, subtitle, url }) => {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const nv = new Niivue();

    const init = async () => {
      try {
        await nv.attachToCanvas(canvasRef.current);
        await nv.loadVolumes([
          {
            url,
            colorMap: "gray",
          },
        ]);
        if (!active) return;
        if (typeof nv.setSliceType === "function") {
          nv.setSliceType(nv.sliceTypeAxial);
        }
        setStatus("ready");
      } catch (loadError) {
        if (!active) return;
        setError(loadError?.message ?? "Failed to load volume.");
        setStatus("error");
      }
    };

    init();

    return () => {
      active = false;
      if (typeof nv.destroy === "function") {
        nv.destroy();
      }
    };
  }, [url]);

  return (
    <div className="image-panel">
      <div className="image-placeholder image-canvas niivue">
        <canvas ref={canvasRef} aria-label={`${title} volume`} />
        {status !== "ready" ? (
          <div className="viewer-overlay">
            <span>{title}</span>
            <p>{status === "error" ? "Unable to load" : "Loading volume..."}</p>
          </div>
        ) : (
          <div className="viewer-hint">Scroll or drag to navigate slices</div>
        )}
      </div>
      <div className="image-meta">
        <span>{subtitle}</span>
        <span>{title}</span>
      </div>
      {error ? <div className="viewer-error">{error}</div> : null}
    </div>
  );
};

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);

  const modelSnapshot = useMemo(
    () => [
      { label: "csPCa probability", value: "0.72", sub: "High" },
      { label: "Lesion count", value: "2", sub: "Dominant lesion" },
      { label: "Model volume", value: "34 cc", sub: "Imaging-derived" },
    ],
    []
  );

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setResult({
      summary:
        "Guideline-aligned recommendation for localized prostate cancer based on clinical staging and MedGemma VLM imaging context.",
      options: [
        "MRI-targeted + systematic biopsy for confirmation",
        "Radical prostatectomy with pelvic lymph node assessment",
        "External beam radiotherapy + short-term ADT",
      ],
      note:
        "This is a demo recommendation aligned to EAU guideline structure. Replace with your clinical rules engine.",
    });
  };

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">MG</span>
          <div>
            <p className="brand-title">MedGemma Impact Challenge</p>
            <p className="brand-subtitle">Clinical Decision Support Prototype</p>
          </div>
        </div>
        <button className="ghost-btn" type="button">
          Export Report
        </button>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">MedGemma VLM + Clinical Data</p>
          <h1>
            Treatment recommendation support for prostate cancer care teams.
          </h1>
          <p className="lead">
            A scalable, guideline-aligned interface for imaging review, model
            predictions, and evidence-based treatment pathways.
          </p>
        </div>
      </section>

      <main className="grid">
        <section className="card imaging">
          <div className="card-header">
            <h2>Imaging Review</h2>
            <p>T2W + ADC volumes with slice navigation.</p>
          </div>
          <div className="image-grid">
            <NiiViewer
              title="T2W"
              subtitle="Axial T2-weighted"
              url="/imgs/t2w.nii.gz"
            />
            <NiiViewer
              title="ADC"
              subtitle="Diffusion-weighted"
              url="/imgs/adc.nii.gz"
            />
          </div>
        </section>

        <section className="card model">
          <div className="card-header">
            <h2>Model Predictions</h2>
            <p>MedGemma inference snapshot.</p>
          </div>
          <div className="metrics">
            {modelSnapshot.map((metric) => (
              <div className="metric" key={metric.label}>
                <p>{metric.label}</p>
                <h3>{metric.value}</h3>
                <span>{metric.sub}</span>
              </div>
            ))}
          </div>
          <div className="disclaimer">
            Demo outputs only. Not for clinical use.
          </div>
        </section>

        <section className="card form">
          <div className="card-header">
            <h2>Clinical Inputs</h2>
            <p>Capture TNM staging and key variables.</p>
          </div>
          <form onSubmit={handleSubmit} className="input-grid">
            <label>
              cT stage
              <select name="tStage" value={form.tStage} onChange={handleChange}>
                <option>T1</option>
                <option>T2</option>
                <option>T3</option>
                <option>T4</option>
              </select>
            </label>
            <label>
              cN stage
              <select name="nStage" value={form.nStage} onChange={handleChange}>
                <option>N0</option>
                <option>N1</option>
              </select>
            </label>
            <label>
              cM stage
              <select name="mStage" value={form.mStage} onChange={handleChange}>
                <option>M0</option>
                <option>M1</option>
              </select>
            </label>
            <label>
              PSA (ng/mL)
              <input name="psa" value={form.psa} onChange={handleChange} />
            </label>
            <label>
              Performance status
              <select
                name="performance"
                value={form.performance}
                onChange={handleChange}
              >
                <option>ECOG 0</option>
                <option>ECOG 1</option>
                <option>ECOG 2</option>
                <option>ECOG 3</option>
              </select>
            </label>
            <label>
              Age
              <input name="age" value={form.age} onChange={handleChange} />
            </label>
            <button className="primary-btn" type="submit">
              Generate Recommendation
            </button>
          </form>
        </section>

        <section className="card recommendation">
          <div className="card-header">
            <h2>EAU Recommendation</h2>
            <p>Guideline-aligned treatment pathway.</p>
          </div>
          {!result ? (
            <div className="empty-state">
              Submit clinical inputs to generate a recommendation.
            </div>
          ) : (
            <div className="recommendation-body">
              <p className="summary">{result.summary}</p>
              <div className="option-list">
                {result.options.map((option) => (
                  <div className="option" key={option}>
                    {option}
                  </div>
                ))}
              </div>
              <p className="note">{result.note}</p>
            </div>
          )}
        </section>

        <section className="card chatbot">
          <div className="card-header">
            <h2>Clinical Assistant</h2>
            <p>Ask MedGemma about the case, guidelines, or next steps.</p>
          </div>
          <div className="chatbot-shell">
            <div className="chatbot-messages">
              <div className="chatbot-empty">
                Chat history will appear here.
              </div>
            </div>
            <div className="chatbot-input">
              <input
                type="text"
                placeholder="Ask about treatment options, evidence, or imaging findings..."
              />
              <button type="button">Send</button>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>
          Built for MedGemma Impact Challenge. Replace placeholder images and
          integrate your model + guideline engine.
        </p>
        <div className="footer-tags">
          <span>FHIR-ready</span>
          <span>Audit trail</span>
          <span>Explainability hooks</span>
        </div>
      </footer>
    </div>
  );
}
