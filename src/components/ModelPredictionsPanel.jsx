export default function ModelPredictionsPanel({ modelPrediction, csPcaPrediction, psaDensity }) {
  const snapshot = [
    {
      label: "csPCa probability",
      value: modelPrediction.csPcaProbability.toFixed(2),
      sub: "Threshold 0.5",
    },
    {
      label: "csPCa prediction",
      value: csPcaPrediction.startsWith("Yes") ? "Yes" : "No",
      sub: "Gleason >= 3+4 (ISUP >= 2)",
    },
    {
      label: "Predicted prostate volume",
      value: `${modelPrediction.predictedProstateVolumeMl} mL`,
      sub: `PSA density: ${psaDensity || "n/a"} ng/mL/mL`,
    },
  ];

  return (
    <section className="card model">
      <div className="card-header">
        <h2>AI Predictions</h2>
        <p>MedGemma inference snapshot.</p>
      </div>
      <div className="metrics">
        {snapshot.map((metric) => (
          <div className="metric" key={metric.label}>
            <p>{metric.label}</p>
            <h3>{metric.value}</h3>
            <span>{metric.sub}</span>
          </div>
        ))}
      </div>
      <div className="disclaimer">Demo outputs only. Not for clinical use.</div>
    </section>
  );
}
