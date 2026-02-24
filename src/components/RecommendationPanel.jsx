import { useMemo, useState } from "react";
import { normalizeRecommendation } from "../utils/recommendation";

export default function RecommendationPanel({ result, isLoading, error }) {
  const [isModalOpen, setModalOpen] = useState(false);
  const normalizedResult = useMemo(() => normalizeRecommendation(result), [result]);
  const isParseError = Boolean(normalizedResult?.parseError);

  const options = (normalizedResult?.options || []).slice(0, 3);
  const displayedOptions =
    isParseError
      ? []
      : options.length === 3
        ? options
        : [
            ...options,
            ...Array.from({ length: Math.max(0, 3 - options.length) }, () => ({
              title: "Treatment option pending",
              rationale: "Awaiting model output.",
              details: "No additional detail available.",
            })),
          ];

  const getOptionTitle = (option) => option?.title?.trim() || "Treatment option";
  const getOptionReasoning = (option) =>
    option?.rationale?.trim() || "Reasoning not provided by model.";
  const getOptionDescription = (option) =>
    option?.details?.trim() || "Treatment description not provided by model.";

  return (
    <section className="card recommendation">
      <div className="card-header">
        <h2>MedGemma Treatment Recommendations</h2>
        <p>Top 3 ranked options personalized to imaging + clinical context.</p>
      </div>
      {isLoading ? (
        <div className="empty-state">MedGemma is generating recommendations...</div>
      ) : !normalizedResult ? (
        <div className="empty-state">{error || "Submit clinical inputs to generate a recommendation."}</div>
      ) : (
        <div className="recommendation-body">
          {error ? <p className="note warning-note">{error}</p> : null}
          <p className="summary">{normalizedResult.summary}</p>
          {isParseError ? (
            <>
              <p className="note">{normalizedResult.note}</p>
              <div className="raw-output-panel">
                <p className="raw-output-label">Raw MedGemma output (parse failed):</p>
                <pre className="raw-model-output">
                  {normalizedResult.rawOutput ||
                    normalizedResult.fullDetails ||
                    normalizedResult.summary}
                </pre>
              </div>
            </>
          ) : (
            <>
              <div className="option-grid">
                {displayedOptions.map((option, index) => (
                  <div className="option-tile" key={`${index}-${getOptionTitle(option)}`}>
                    <span className="option-rank">Option {index + 1}</span>
                    <p className="option-title">{getOptionTitle(option)}</p>
                    <p className="option-preview">{getOptionReasoning(option)}</p>
                  </div>
                ))}
              </div>
              <button type="button" className="details-btn" onClick={() => setModalOpen(true)}>
                View full details
              </button>
              <p className="note">{normalizedResult.note}</p>

              {isModalOpen ? (
                <div
                  className="details-modal-backdrop"
                  role="presentation"
                  onClick={() => setModalOpen(false)}
                >
                  <div
                    className="details-modal"
                    role="dialog"
                    aria-modal="true"
                    aria-label="Full recommendation details"
                    onClick={(event) => event.stopPropagation()}
                  >
                    <div className="details-modal-header">
                      <h3>Detailed Recommendation Review</h3>
                      <button
                        type="button"
                        className="modal-close-btn"
                        onClick={() => setModalOpen(false)}
                      >
                        Close
                      </button>
                    </div>
                    {normalizedResult.fullDetails &&
                    normalizedResult.fullDetails !== normalizedResult.summary ? (
                      <p className="summary">{normalizedResult.fullDetails}</p>
                    ) : null}
                    <div className="details-stack">
                      <h4 className="stack-heading">Treatment-by-treatment details</h4>
                      {displayedOptions.map((option, index) => (
                        <div className="details-block" key={`details-${index}-${getOptionTitle(option)}`}>
                          <h4>
                            {index + 1}. {getOptionTitle(option)}
                          </h4>
                          <p>
                            <strong>Reasoning:</strong> {getOptionReasoning(option)}
                          </p>
                          <p>
                            <strong>Description:</strong> {getOptionDescription(option)}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
            </>
          )}
        </div>
      )}
    </section>
  );
}
