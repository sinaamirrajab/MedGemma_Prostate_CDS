export default function ModelPredictionsPanel({
  modelPrediction,
  psaDensity,
  classifierResult,
  isClassifying,
  onRunClassifier,
  classifierError,
}) {
  const hasResult = classifierResult !== null;
  const probability = hasResult ? classifierResult.probability : null;
  const predictionYes = hasResult ? classifierResult.prediction : null;

  const snapshot = [
    {
      label: "csPCa probability",
      value: probability !== null ? probability.toFixed(3) : "—",
      sub: hasResult ? "MedSigLIP ensemble · 5-fold" : "Run classifier to compute",
    },
    {
      label: "csPCa prediction",
      value: predictionYes === null ? "—" : predictionYes ? "Yes" : "No",
      sub: predictionYes === null
        ? "Run classifier to compute"
        : predictionYes
          ? "Gleason \u2265\u20093+4\u2002(ISUP \u2265\u20092)"
          : "Gleason \u2264\u20093+3\u2002(ISUP \u2264\u20091)",
      highlight: predictionYes === null ? null : predictionYes ? "positive" : "negative",
    },
    {
      label: "Predicted prostate volume",
      value: `${modelPrediction.predictedProstateVolumeMl} mL`,
      sub: `PSA density:\u2002${psaDensity || "n/a"}\u2002ng/mL/mL`,
    },
  ];

  return (
    <section className="card model">
      <div className="card-header">
        <h2>MedSigLIP Predictions</h2>
      </div>

      <div className="metrics">
        {snapshot.map((metric) => (
          <div
            className={`metric${metric.highlight ? ` metric--${metric.highlight}` : ""}`}
            key={metric.label}
          >
            <p>{metric.label}</p>
            <h3>{metric.value}</h3>
            <span>{metric.sub}</span>
          </div>
        ))}
      </div>

      {/* Uncertainty block — shown only after classifier has run */}
      {classifierResult && (
        <div className="uncertainty-block">
          <h4 className="uncertainty-title">Ensemble uncertainty (5 folds)</h4>
          <div className="uncertainty-grid">
            <div className="unc-item">
              <span className="unc-label">Epistemic std</span>
              <span className="unc-value">{classifierResult.uncertaintyStd.toFixed(3)}</span>
              <div className="unc-bar">
                <div
                  className="unc-fill"
                  style={{ width: `${Math.min(classifierResult.uncertaintyStd * 200, 100)}%` }}
                />
              </div>
            </div>
            <div className="unc-item">
              <span className="unc-label">Entropy</span>
              <span className="unc-value">{classifierResult.uncertaintyEntropy.toFixed(3)}</span>
              <div className="unc-bar">
                <div
                  className="unc-fill"
                  style={{ width: `${Math.min(classifierResult.uncertaintyEntropy * 100, 100)}%` }}
                />
              </div>
            </div>
            <div className="unc-item">
              <span className="unc-label">95% CI</span>
              <span className="unc-value">
                [{Number.isFinite(classifierResult.ciLo) ? classifierResult.ciLo.toFixed(2) : "—"},&nbsp;
                 {Number.isFinite(classifierResult.ciHi) ? classifierResult.ciHi.toFixed(2) : "—"}]
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Run button */}
      <div className="classifier-action">
        <button
          type="button"
          className={`btn-classify${classifierResult ? " btn-classify--rerun" : ""}`}
          onClick={onRunClassifier}
          disabled={isClassifying}
        >
          {isClassifying
            ? "⏳ Running MedSigLIP…"
            : classifierResult
              ? "↺ Re-run MedSigLIP Classifier"
              : "▶ Run MedSigLIP Classifier"}
        </button>
        {classifierError && <p className="classifier-error">{classifierError}</p>}
      </div>

      <div className="disclaimer">Demo outputs only. Not for clinical use.</div>
    </section>
  );
}
