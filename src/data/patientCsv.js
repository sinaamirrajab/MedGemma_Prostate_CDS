import csvText from "../../imgs/df_ui_test_4cases.csv?raw";
import { parseCsvLine, serializeCsv } from "./csvUtils";
export { serializeCsv };

export function loadPatientsFromCsv() {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) return { headers: [], rows: [] };

  const headers = parseCsvLine(lines[0]);
  const rows = lines.slice(1).map((line) => {
    const cols = parseCsvLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = cols[index] ?? "";
    });
    return row;
  });

  return { headers, rows };
}

const configuredRoots = [
  import.meta.env.VITE_PROJECT_ROOT,
  import.meta.env.VITE_ADDITIONAL_VIEWER_ROOT,
]
  .map((value) => (typeof value === "string" ? value.trim() : ""))
  .filter(Boolean);

const PROJECT_ROOTS = configuredRoots;

export function toViewerUrl(pathValue) {
  if (!pathValue) return "";
  const normalizedPath = String(pathValue).trim();
  if (!normalizedPath) return "";

  // If the path is inside the project root, serve it as a root-relative URL
  // (Vite serves the project root via its static file server)
  for (const root of PROJECT_ROOTS) {
    const normalizedRoot = root.endsWith("/") ? root.slice(0, -1) : root;
    if (normalizedPath.startsWith(`${normalizedRoot}/`)) {
      return normalizedPath.slice(normalizedRoot.length); // e.g. /imgs/11375/...t2w.mha
    }
  }
  // Fallback: use Vite's /@fs endpoint for any other absolute path
  if (normalizedPath.startsWith("/")) return `/@fs${normalizedPath}`;
  return normalizedPath;
}
