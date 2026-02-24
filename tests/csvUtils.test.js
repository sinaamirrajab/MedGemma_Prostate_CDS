import test from "node:test";
import assert from "node:assert/strict";
import { parseCsvLine, serializeCsv } from "../src/data/csvUtils.js";

test("parseCsvLine handles quotes and commas", () => {
  const values = parseCsvLine('a,"b,c","d""e"');
  assert.deepEqual(values, ["a", "b,c", 'd"e']);
});

test("serializeCsv escapes commas, quotes and newlines", () => {
  const csv = serializeCsv(
    ["name", "note"],
    [{ name: 'A"B', note: "line1\nline2,ok" }],
  );
  assert.equal(csv, 'name,note\n"A""B","line1\nline2,ok"');
});
