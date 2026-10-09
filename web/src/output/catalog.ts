import type { MediaKind } from "../export/models";

export const STATE_DIRECTORY = ".snapmemories";
export const CATALOG_FILE = "catalog.jsonl";

export interface CatalogRecord {
  item_id: string;
  relative_path: string;
  kind: MediaKind;
  taken_at: string;
  latitude: number | null;
  longitude: number | null;
}

export function serializeRecords(records: CatalogRecord[]): string {
  return records.map((record) => `${JSON.stringify(record)}\n`).join("");
}

export function parseCatalogIds(text: string): Set<string> {
  const ids = new Set<string>();
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      const record: unknown = JSON.parse(line);
      if (typeof record === "object" && record !== null && "item_id" in record) {
        ids.add(String(record.item_id));
      }
    } catch {}
  }
  return ids;
}
