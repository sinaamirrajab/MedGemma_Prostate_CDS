import { useMemo, useState } from "react";

const stripCodeFences = (value) =>
  value
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

const cleanText = (value) => {
  if (typeof value !== "string") {
    if (value === undefined || value === null) {
      return "";
    }
    return String(value);
  }
  const text = stripCodeFences(value.trim());
  if (text.startsWith('"') && text.endsWith('"')) {
    try {
      const unwrapped = JSON.parse(text);
      return typeof unwrapped === "string" ? unwrapped.trim() : text;
    } catch {
      return text;
    }
  }
  return text;
};

const escapeJsonControlCharsInStrings = (value) => {
  let output = "";
  let inString = false;
  let escaped = false;

  for (let index = 0; index < value.length; index += 1) {
    const char = value[index];
    if (inString) {
      if (escaped) {
        output += char;
        escaped = false;
        continue;
      }
      if (char === "\\") {
        output += char;
        escaped = true;
        continue;
      }
      if (char === '"') {
        output += char;
        inString = false;
        continue;
      }
      if (char === "\n") {
        output += "\\n";
        continue;
      }
      if (char === "\r") {
        continue;
      }
      if (char === "\t") {
        output += "\\t";
        continue;
      }
      output += char;
      continue;
    }

    if (char === '"') {
      inString = true;
    }
    output += char;
  }

  return output;
};

const extractJsonObject = (value) => {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }

  const trimmed = value.trim();
  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidates = [];
  if (fencedMatch?.[1]) {
    candidates.push(fencedMatch[1]);
  }
  candidates.push(trimmed);

  for (const candidate of candidates) {
    const firstBrace = candidate.indexOf("{");
    const lastBrace = candidate.lastIndexOf("}");
    if (firstBrace < 0 || lastBrace <= firstBrace) {
      continue;
    }
    const jsonLike = candidate.slice(firstBrace, lastBrace + 1);
    try {
      const parsed = JSON.parse(jsonLike);
      if (parsed && typeof parsed === "object") {
        return parsed;
      }
    } catch {
      try {
        const repaired = escapeJsonControlCharsInStrings(jsonLike);
        const parsed = JSON.parse(repaired);
        if (parsed && typeof parsed === "object") {
          return parsed;
        }
      } catch {
        // continue scanning
      }
    }
  }

  return null;
};

const normalizeRecommendation = (result) => {
  if (!result) {
    return null;
  }

  let normalized = typeof result === "object" ? { ...result } : { summary: String(result) };
  if (normalized.parseError) {
    return {
      parseError: true,
      summary: cleanText(normalized.summary || "MedGemma response could not be parsed as strict JSON."),
      fullDetails: typeof normalized.fullDetails === "string" ? normalized.fullDetails : "",
      rawOutput: typeof normalized.rawOutput === "string" ? normalized.rawOutput : "",
      options: [],
      note: cleanText(normalized.note || "Raw MedGemma output shown for debugging."),
    };
  }

  const parsedFromSummary = extractJsonObject(normalized.summary);
  const parsedFromDetails = extractJsonObject(normalized.fullDetails);
  const parsedObject =
    parsedFromDetails || parsedFromSummary || extractJsonObject(normalized.note);
  if (parsedObject) {
    normalized = { ...normalized, ...parsedObject };
  }

  normalized.summary = cleanText(normalized.summary || normalized.fullDetails || "");
  normalized.fullDetails = cleanText(normalized.fullDetails || normalized.summary);
  normalized.note = cleanText(normalized.note);

  const options = Array.isArray(normalized.options) ? normalized.options : [];
  normalized.options = options
    .map((option) => {
      if (typeof option === "string") {
        const title = cleanText(option);
        return { title, rationale: "", details: "" };
      }
      if (!option || typeof option !== "object") {
        return null;
      }
      return {
        title: cleanText(option.title || option.name),
        rationale: cleanText(option.reasoning || option.rationale),
        details: cleanText(option.description || option.details),
      };
    })
    .filter((option) => option && (option.title || option.rationale || option.details));

  return normalized;
};

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
