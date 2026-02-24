import { cleanText, normalizeRecommendation } from "./recommendation";

const PAGE_WIDTH = 595.28; // A4 portrait width in points
const PAGE_HEIGHT = 841.89; // A4 portrait height in points
const MARGIN_X = 44;
const MARGIN_TOP = 46;
const MARGIN_BOTTOM = 46;
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN_X * 2;

const FONT_REGULAR = "F1";
const FONT_BOLD = "F2";

const formatPercent = (value) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Not available";

const formatNumber = (value, digits = 3) =>
  Number.isFinite(value) ? Number(value).toFixed(digits) : "Not available";

const sanitizePdfString = (value) =>
  String(value ?? "")
    .replace(/\\/g, "\\\\")
    .replace(/\(/g, "\\(")
    .replace(/\)/g, "\\)")
    .replace(/[\r\n\t]/g, " ")
    .replace(/[^\x20-\x7E]/g, "?")
    .trim();

const charWidth = (character, fontSize) => {
  if ("il.,'`|:;! ".includes(character)) return fontSize * 0.28;
  if ("MW@#%&".includes(character)) return fontSize * 0.82;
  return fontSize * 0.54;
};

const estimateTextWidth = (text, fontSize) =>
  Array.from(String(text ?? "")).reduce((sum, ch) => sum + charWidth(ch, fontSize), 0);

const splitLongWord = (word, maxWidth, fontSize) => {
  const parts = [];
  let current = "";
  for (const ch of Array.from(word)) {
    const next = current + ch;
    if (current && estimateTextWidth(next, fontSize) > maxWidth) {
      parts.push(current);
      current = ch;
    } else {
      current = next;
    }
  }
  if (current) parts.push(current);
  return parts;
};

const wrapText = (text, maxWidth, fontSize) => {
  const normalized = cleanText(text).replace(/\s+/g, " ").trim();
  if (!normalized) return [];
  const words = normalized.split(" ");
  const lines = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (estimateTextWidth(candidate, fontSize) <= maxWidth) {
      current = candidate;
      continue;
    }
    if (current) lines.push(current);
    if (estimateTextWidth(word, fontSize) <= maxWidth) {
      current = word;
      continue;
    }
    const chunks = splitLongWord(word, maxWidth, fontSize);
    if (!chunks.length) continue;
    for (let i = 0; i < chunks.length - 1; i += 1) lines.push(chunks[i]);
    current = chunks[chunks.length - 1];
  }

  if (current) lines.push(current);
  return lines;
};

const byteLength = (text) => new TextEncoder().encode(text).length;

const toPdfY = (topY, height = 0) => PAGE_HEIGHT - topY - height;

class PdfComposer {
  constructor() {
    this.pages = [];
    this.currentPage = [];
    this.cursorTop = MARGIN_TOP;
    this.pageNumber = 0;
    this.addPage();
  }

  addPage() {
    this.currentPage = [];
    this.pages.push(this.currentPage);
    this.pageNumber += 1;

    this.drawTopRect(MARGIN_X, 18, CONTENT_WIDTH, 24, [0.12, 0.31, 0.24]);
    this.drawText("Clinical Decision Support Report", {
      x: MARGIN_X + 10,
      top: 24,
      font: FONT_BOLD,
      size: 12,
      color: [1, 1, 1],
    });
    this.drawText(`Page ${this.pageNumber}`, {
      x: PAGE_WIDTH - MARGIN_X - 44,
      top: 24,
      font: FONT_REGULAR,
      size: 10,
      color: [0.91, 0.97, 0.93],
    });
    this.cursorTop = 56;
  }

  ensureSpace(requiredHeight) {
    if (this.cursorTop + requiredHeight > PAGE_HEIGHT - MARGIN_BOTTOM) {
      this.addPage();
    }
  }

  drawTopRect(x, top, width, height, color) {
    const y = toPdfY(top, height);
    this.currentPage.push(
      `${color[0].toFixed(3)} ${color[1].toFixed(3)} ${color[2].toFixed(3)} rg`,
      `${x.toFixed(2)} ${y.toFixed(2)} ${width.toFixed(2)} ${height.toFixed(2)} re f`
    );
  }

  drawLine(x1, x2, top, color = [0.82, 0.88, 0.84], width = 1) {
    const y = toPdfY(top);
    this.currentPage.push(
      `${width.toFixed(2)} w`,
      `${color[0].toFixed(3)} ${color[1].toFixed(3)} ${color[2].toFixed(3)} RG`,
      `${x1.toFixed(2)} ${y.toFixed(2)} m ${x2.toFixed(2)} ${y.toFixed(2)} l S`
    );
  }

  drawText(
    text,
    {
      x = MARGIN_X,
      top = this.cursorTop,
      font = FONT_REGULAR,
      size = 11,
      color = [0.12, 0.17, 0.15],
    } = {}
  ) {
    const safe = sanitizePdfString(text);
    if (!safe) return;
    const y = toPdfY(top + size);
    this.currentPage.push(
      "BT",
      `/${font} ${size.toFixed(2)} Tf`,
      `${color[0].toFixed(3)} ${color[1].toFixed(3)} ${color[2].toFixed(3)} rg`,
      `1 0 0 1 ${x.toFixed(2)} ${y.toFixed(2)} Tm`,
      `(${safe}) Tj`,
      "ET"
    );
  }

  addParagraph(
    text,
    {
      x = MARGIN_X,
      width = CONTENT_WIDTH,
      font = FONT_REGULAR,
      size = 11,
      lineHeight = size * 1.35,
      color = [0.12, 0.17, 0.15],
      spacingAfter = 6,
    } = {}
  ) {
    const lines = wrapText(text, width, size);
    if (!lines.length) {
      this.cursorTop += spacingAfter;
      return;
    }
    const required = lines.length * lineHeight + spacingAfter;
    this.ensureSpace(required);
    lines.forEach((line, index) => {
      this.drawText(line, {
        x,
        top: this.cursorTop + index * lineHeight,
        font,
        size,
        color,
      });
    });
    this.cursorTop += lines.length * lineHeight + spacingAfter;
  }

  addSection(title) {
    this.ensureSpace(30);
    this.drawTopRect(MARGIN_X, this.cursorTop, CONTENT_WIDTH, 22, [0.93, 0.97, 0.94]);
    this.drawText(title, {
      x: MARGIN_X + 10,
      top: this.cursorTop + 5,
      font: FONT_BOLD,
      size: 11.5,
      color: [0.13, 0.31, 0.24],
    });
    this.cursorTop += 28;
  }

  addField(label, value) {
    const labelWidth = 160;
    const lineHeight = 14;
    const labelText = `${cleanText(label, "Field")}:`;
    const valueText = cleanText(value, "Not available");
    const lines = wrapText(valueText, CONTENT_WIDTH - labelWidth, 10.5);
    const rowCount = Math.max(1, lines.length);
    this.ensureSpace(rowCount * lineHeight + 2);

    this.drawText(labelText, {
      x: MARGIN_X,
      top: this.cursorTop,
      font: FONT_BOLD,
      size: 10.5,
      color: [0.17, 0.25, 0.21],
    });

    if (!lines.length) {
      this.drawText("Not available", {
        x: MARGIN_X + labelWidth,
        top: this.cursorTop,
        font: FONT_REGULAR,
        size: 10.5,
      });
    } else {
      lines.forEach((line, index) => {
        this.drawText(line, {
          x: MARGIN_X + labelWidth,
          top: this.cursorTop + index * lineHeight,
          font: FONT_REGULAR,
          size: 10.5,
        });
      });
    }

    this.cursorTop += rowCount * lineHeight + 2;
  }

  addOption(index, option) {
    const title = cleanText(option?.title, "Treatment option");
    const reasoning = cleanText(option?.reasoning, "Not provided.");
    const description = cleanText(option?.description, "Not provided.");

    this.ensureSpace(30);
    this.drawTopRect(MARGIN_X, this.cursorTop, CONTENT_WIDTH, 20, [0.95, 0.98, 0.96]);
    this.drawText(`Option ${index + 1}: ${title}`, {
      x: MARGIN_X + 8,
      top: this.cursorTop + 4,
      font: FONT_BOLD,
      size: 10.8,
      color: [0.13, 0.31, 0.24],
    });
    this.cursorTop += 24;

    this.addParagraph(`Reasoning: ${reasoning}`, {
      size: 10.3,
      lineHeight: 13.6,
      spacingAfter: 4,
    });
    this.addParagraph(`Description: ${description}`, {
      size: 10.3,
      lineHeight: 13.6,
      spacingAfter: 6,
    });
  }

  pageStreams() {
    return this.pages.map((commands) => commands.join("\n"));
  }
}

const buildPdf = (pageStreams) => {
  const objects = [];
  const addObject = (body) => {
    objects.push(body);
    return objects.length;
  };

  const catalogId = addObject("<< /Type /Catalog /Pages 2 0 R >>");
  const pagesId = addObject("<< /Type /Pages /Kids [] /Count 0 >>");
  const regularFontId = addObject("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
  const boldFontId = addObject("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>");

  const pageIds = [];
  for (const stream of pageStreams) {
    const contentId = addObject(
      `<< /Length ${byteLength(stream)} >>\nstream\n${stream}\nendstream`
    );
    const pageId = addObject(
      `<< /Type /Page /Parent ${pagesId} 0 R /MediaBox [0 0 ${PAGE_WIDTH.toFixed(2)} ${PAGE_HEIGHT.toFixed(
        2
      )}] /Resources << /Font << /${FONT_REGULAR} ${regularFontId} 0 R /${FONT_BOLD} ${boldFontId} 0 R >> >> /Contents ${contentId} 0 R >>`
    );
    pageIds.push(pageId);
  }

  objects[pagesId - 1] = `<< /Type /Pages /Kids [${pageIds
    .map((id) => `${id} 0 R`)
    .join(" ")}] /Count ${pageIds.length} >>`;

  let pdf = "%PDF-1.4\n";
  const offsets = [0];
  objects.forEach((objectBody, index) => {
    offsets.push(byteLength(pdf));
    pdf += `${index + 1} 0 obj\n${objectBody}\nendobj\n`;
  });

  const xrefOffset = byteLength(pdf);
  pdf += `xref\n0 ${objects.length + 1}\n`;
  pdf += "0000000000 65535 f \n";
  for (let i = 1; i < offsets.length; i += 1) {
    pdf += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root ${catalogId} 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`;
  return new TextEncoder().encode(pdf);
};

const triggerDownload = (bytes, filename) => {
  const blob = new Blob([bytes], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1200);
};

export function openCaseReportPdf({
  patientId,
  patient,
  modelPrediction,
  classifierResult,
  recommendation,
}) {
  try {
    const recommendationView = normalizeRecommendation(recommendation) || {
      parseError: false,
      summary: "No recommendation generated yet.",
      fullDetails: "No detailed recommendation generated yet.",
      options: [],
      note: "",
      rawOutput: "",
    };
    const generatedAt = new Date().toLocaleString();
    const comorbidities = Array.isArray(patient?.comorbidities) ? patient.comorbidities : [];
    const tStage = cleanText(patient?.tStage, "N/A");
    const nStage = cleanText(patient?.nStage, "N/A");
    const mStage = cleanText(patient?.mStage, "N/A");
    const clinicalStage =
      tStage === "N/A" && nStage === "N/A" && mStage === "N/A"
        ? "Not available"
        : `cT ${tStage}, cN ${nStage}, cM ${mStage}`;
    const comorbidityList = comorbidities.length ? comorbidities.join(", ") : "None reported";
    const comorbidityBurden = `${comorbidities.length} selected`;

    const composer = new PdfComposer();

    composer.addParagraph(`Patient ID: ${cleanText(patientId, "N/A")}`, {
      font: FONT_BOLD,
      size: 12,
      lineHeight: 15,
      spacingAfter: 2,
      color: [0.13, 0.24, 0.2],
    });
    composer.addParagraph(`Generated: ${generatedAt}`, {
      size: 10.5,
      spacingAfter: 8,
      color: [0.32, 0.4, 0.36],
    });
    composer.drawLine(MARGIN_X, PAGE_WIDTH - MARGIN_X, composer.cursorTop);
    composer.cursorTop += 8;

    composer.addSection("Clinical Snapshot");
    composer.addField("Clinical stage (TNM)", clinicalStage);
    composer.addField("cT stage", tStage);
    composer.addField("cN stage", nStage);
    composer.addField("cM stage", mStage);
    composer.addField("Age", cleanText(patient?.age, "N/A"));
    composer.addField("PSA (ng/mL)", cleanText(patient?.psa, "N/A"));
    composer.addField("PSA density (PSA/volume)", cleanText(patient?.psaDensity, "N/A"));
    composer.addField("Performance", cleanText(patient?.performance, "N/A"));
    composer.addField("Family history", cleanText(patient?.familyHistory, "N/A"));
    composer.addField("LUTS", cleanText(patient?.lowerUrinarySymptoms, "N/A"));
    composer.addField("Prior biopsy", cleanText(patient?.priorBiopsy, "N/A"));
    composer.addField("Comorbidity burden", comorbidityBurden);
    composer.addField("Comorbidities", comorbidityList);
    composer.addField(
      "Sexual function preservation",
      cleanText(patient?.preferencesSexualFunction, "N/A")
    );
    composer.addField(
      "Urinary continence preservation",
      cleanText(patient?.preferencesUrinaryContinence, "N/A")
    );
    composer.addField(
      "Treatment intensity preference",
      cleanText(patient?.preferencesTreatmentIntensity, "N/A")
    );
    composer.addField(
      "Follow-up burden tolerance",
      cleanText(patient?.preferencesFollowUpBurden, "N/A")
    );
    composer.addField(
      "Additional priorities",
      cleanText(patient?.additionalPreferences, "None provided")
    );

    composer.addSection("Model Outputs");
    composer.addField("csPCa probability", formatPercent(modelPrediction?.csPcaProbability));
    composer.addField("csPCa prediction", cleanText(modelPrediction?.csPcaPrediction, "Not available"));
    composer.addField(
      "Predicted prostate volume (mL)",
      formatNumber(modelPrediction?.predictedProstateVolumeMl, 1)
    );
    composer.addField("Uncertainty (STD)", formatNumber(classifierResult?.uncertaintyStd));
    composer.addField("Uncertainty (Entropy)", formatNumber(classifierResult?.uncertaintyEntropy));
    composer.addField(
      "95% CI",
      Number.isFinite(classifierResult?.ciLo) && Number.isFinite(classifierResult?.ciHi)
        ? `${Number(classifierResult.ciLo).toFixed(3)} - ${Number(classifierResult.ciHi).toFixed(3)}`
        : "Not available"
    );

    composer.addSection("Detailed Treatment Recommendation");
    if (
      recommendationView.summary &&
      recommendationView.fullDetails &&
      recommendationView.summary !== recommendationView.fullDetails
    ) {
      composer.addField("Recommendation summary", recommendationView.summary);
    }
    composer.addParagraph(recommendationView.fullDetails || recommendationView.summary, {
      size: 11,
      lineHeight: 15,
      spacingAfter: 8,
    });

    if (recommendationView.options.length) {
      recommendationView.options.forEach((option, index) => composer.addOption(index, option));
    } else {
      composer.addParagraph("No ranked treatment options available for this case yet.", {
        size: 10.5,
      });
    }

    if (recommendationView.note) {
      composer.addParagraph(`Safety note: ${recommendationView.note}`, {
        font: FONT_BOLD,
        size: 10,
        lineHeight: 13.5,
        spacingAfter: 8,
        color: [0.35, 0.34, 0.22],
      });
    }

    if (recommendationView.parseError && recommendationView.rawOutput) {
      composer.addSection("Raw Model Output");
      const truncatedRaw =
        recommendationView.rawOutput.length > 2600
          ? `${recommendationView.rawOutput.slice(0, 2600)} ... (truncated)`
          : recommendationView.rawOutput;
      composer.addParagraph(truncatedRaw, {
        size: 9.2,
        lineHeight: 12.5,
        spacingAfter: 6,
        color: [0.2, 0.2, 0.2],
      });
    }

    const pdfBytes = buildPdf(composer.pageStreams());
    const safePatientId = cleanText(patientId, "case").replace(/[^a-zA-Z0-9_-]+/g, "_");
    const datePart = new Date().toISOString().slice(0, 10);
    const filename = `cds_report_${safePatientId}_${datePart}.pdf`;
    triggerDownload(pdfBytes, filename);
    return { ok: true };
  } catch (error) {
    return {
      ok: false,
      error: `Could not generate PDF report: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}
