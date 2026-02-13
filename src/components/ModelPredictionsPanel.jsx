export default function ModelPredictionsPanel({ modelSnapshot }) {
  return (
    <section className="card model">
      <div className="card-header">
        <h2>AI Predictions</h2>
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
      <div className="disclaimer">Demo outputs only. Not for clinical use.</div>
    </section>
  );
}
