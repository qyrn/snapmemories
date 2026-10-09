import type { FileEntry } from "@zip.js/zip.js";

export type MediaKind = "photo" | "video";

export interface GeoPoint {
  latitude: number;
  longitude: number;
}

export interface MemoryEntry {
  takenAt: Date;
  kind: MediaKind;
  location: GeoPoint | null;
  mediaDownloadUrl: string;
  downloadLink: string;
}

export interface ArchiveMember {
  entry: FileEntry;
  name: string;
  size: number;
}

export interface EmbeddedMemory {
  key: string;
  kind: MediaKind;
  main: ArchiveMember;
  overlay: ArchiveMember | null;
  archivedAtMs: number;
}

export interface ImportItem {
  id: string;
  kind: MediaKind;
  takenAt: Date;
  location: GeoPoint | null;
  source: EmbeddedMemory;
}
