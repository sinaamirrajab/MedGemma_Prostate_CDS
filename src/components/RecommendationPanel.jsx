export default function RecommendationPanel({ result }) {
  return (
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
  );
}
