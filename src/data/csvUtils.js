export function parseCsvLine(line) {
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

export function escapeCsvValue(value) {
  const stringValue = value == null ? "" : String(value);
  if (stringValue.includes('"') || stringValue.includes(",") || stringValue.includes("\n")) {
    return `"${stringValue.replaceAll('"', '""')}"`;
  }
  return stringValue;
}

export function serializeCsv(headers, rows) {
  const headerLine = headers.map(escapeCsvValue).join(",");
  const body = rows.map((row) => headers.map((header) => escapeCsvValue(row[header])).join(","));
  return [headerLine, ...body].join("\n");
}
