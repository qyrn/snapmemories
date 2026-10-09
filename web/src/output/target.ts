import type { CatalogRecord } from "./catalog";

export interface SavedPart {
  name: string;
  url: string;
  size: number;
}

export interface OutputTarget {
  existingIds(): Promise<Set<string>>;
  write(folder: string, stem: string, extension: string, data: Blob): Promise<string>;
  record(record: CatalogRecord): Promise<void>;
  finish(): Promise<SavedPart[]>;
}

export function uniqueName(stem: string, extension: string, attempt: number): string {
  return attempt === 1 ? `${stem}.${extension}` : `${stem}_${attempt}.${extension}`;
}
