import test from "node:test";
import assert from "node:assert/strict";
import {
  cleanText,
  extractJsonObject,
  normalizeRecommendation,
} from "../src/utils/recommendation.js";

test("extractJsonObject parses fenced json payload", () => {
  const parsed = extractJsonObject('```json\n{"summary":"ok","options":[]}\n```');
  assert.equal(parsed.summary, "ok");
});

test("extractJsonObject repairs newlines inside JSON strings", () => {
  const payload = '{"summary":"line1\nline2","options":[]}';
  const parsed = extractJsonObject(payload);
  assert.equal(parsed.summary, "line1\nline2");
});

test("normalizeRecommendation normalizes option field aliases", () => {
  const normalized = normalizeRecommendation({
    summary: "s",
    options: [{ title: "A", reasoning: "r", description: "d" }],
    note: "",
  });
  assert.equal(normalized.options.length, 1);
  assert.equal(normalized.options[0].rationale, "r");
  assert.equal(normalized.options[0].details, "d");
});

test("normalizeRecommendation keeps parse error shape", () => {
  const normalized = normalizeRecommendation({
    parseError: true,
    summary: "bad",
    rawOutput: "raw",
  });
  assert.equal(normalized.parseError, true);
  assert.equal(normalized.rawOutput, "raw");
});

test("cleanText returns fallback for blank strings", () => {
  assert.equal(cleanText("   ", "fallback"), "fallback");
});
