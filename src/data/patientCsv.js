import csvText from "../../imgs/df_ui_test_4cases.csv?raw";

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current);
  return values;
}

function escapeCsvValue(value) {
  const stringValue = value == null ? "" : String(value);
  if (stringValue.includes('"') || stringValue.includes(",") || stringValue.includes("\n")) {
    return `"${stringValue.replaceAll('"', '""')}"`;
  }
  return stringValue;
}

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

export function serializeCsv(headers, rows) {
  const headerLine = headers.map(escapeCsvValue).join(",");
  const body = rows.map((row) => headers.map((header) => escapeCsvValue(row[header])).join(","));
  return [headerLine, ...body].join("\n");
}

// Project root as known at build/dev time – must match the Vite root
const PROJECT_ROOT = "/mnt/data9/projects/medgemma_challenge_UI/cds";

export function toViewerUrl(pathValue) {
  if (!pathValue) return "";
  // If the path is inside the project root, serve it as a root-relative URL
  // (Vite serves the project root via its static file server)
  if (pathValue.startsWith(PROJECT_ROOT + "/")) {
    return pathValue.slice(PROJECT_ROOT.length); // e.g. /imgs/11375/...t2w.mha
  }
  // Fallback: use Vite's /@fs endpoint for any other absolute path
  if (pathValue.startsWith("/")) return `/@fs${pathValue}`;
  return pathValue;
}
