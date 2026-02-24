const stripCodeFences = (value) =>
  String(value ?? "")
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

export const cleanText = (value, fallback = "") => {
  if (value === undefined || value === null) return fallback;
  if (typeof value !== "string") return String(value);
  const text = stripCodeFences(value.trim());
  if (!text) return fallback;
  if (text.startsWith('"') && text.endsWith('"')) {
    try {
      const unwrapped = JSON.parse(text);
      if (typeof unwrapped === "string") {
        const normalized = unwrapped.trim();
        return normalized || fallback;
      }
      return text;
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

export const extractJsonObject = (value) => {
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
        // keep scanning
      }
    }
  }

  return null;
};

const parseRecommendation = (recommendation) => {
  if (!recommendation) return null;
  if (typeof recommendation === "object") return recommendation;
  if (typeof recommendation !== "string") return null;
  try {
    const parsed = JSON.parse(recommendation);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return {
      summary: cleanText(recommendation, "Recommendation text was not JSON formatted."),
      options: [],
      note: "",
    };
  }
};

export const normalizeRecommendation = (result) => {
  const parsedInput = parseRecommendation(result);
  if (!parsedInput) {
    return null;
  }

  let normalized =
    typeof parsedInput === "object" ? { ...parsedInput } : { summary: String(parsedInput) };

  if (normalized.parseError) {
    return {
      parseError: true,
      summary: cleanText(
        normalized.summary || "MedGemma response could not be parsed as strict JSON."
      ),
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
        reasoning: cleanText(option.reasoning || option.rationale),
        description: cleanText(option.description || option.details),
      };
    })
    .filter((option) => option && (option.title || option.rationale || option.details));

  return normalized;
};
