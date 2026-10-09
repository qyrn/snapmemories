import type { GeoPoint, MediaKind, MemoryEntry } from "./models";

const ENTRY_LIST_KEYS = ["Saved Media", "SavedMedia", "saved_media", "Memories", "memories"];
const DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/;
const COORDINATES_PATTERN = /(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)/;
const VIDEO_TYPES = new Set(["VIDEO", "MOVIE"]);

export type ExportErrorCode = "invalidZip" | "noMemories" | "invalidHistory";

export class ExportFormatError extends Error {
  constructor(
    readonly code: ExportErrorCode,
    readonly fileName = "",
  ) {
    super(code);
  }
}

export function parseMemoriesHistory(text: string): MemoryEntry[] {
  let document: unknown;
  try {
    document = JSON.parse(text.replace(/^﻿/, ""));
  } catch {
    throw new ExportFormatError("invalidHistory");
  }
  const entries: MemoryEntry[] = [];
  for (const item of findEntryList(document)) {
    if (isRecord(item)) {
      const entry = parseEntry(item);
      if (entry) entries.push(entry);
    }
  }
  return entries;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function findEntryList(document: unknown): unknown[] {
  if (Array.isArray(document)) return document;
  if (!isRecord(document)) throw new ExportFormatError("invalidHistory");
  for (const key of ENTRY_LIST_KEYS) {
    const value = document[key];
    if (Array.isArray(value)) return value;
  }
  const firstList = Object.values(document).find(Array.isArray);
  if (firstList) return firstList;
  throw new ExportFormatError("invalidHistory");
}

function text(item: Record<string, unknown>, key: string): string {
  const value = item[key];
  return typeof value === "string" ? value.trim() : "";
}

function parseEntry(item: Record<string, unknown>): MemoryEntry | null {
  const takenAt = parseSnapchatDate(text(item, "Date"));
  if (!takenAt) return null;
  const kind: MediaKind = VIDEO_TYPES.has(text(item, "Media Type").toUpperCase()) ? "video" : "photo";
  return {
    takenAt,
    kind,
    location: parseLocation(item["Location"]),
    mediaDownloadUrl: text(item, "Media Download Url"),
    downloadLink: text(item, "Download Link"),
  };
}

export function parseSnapchatDate(value: string): Date | null {
  const match = DATE_PATTERN.exec(value);
  if (!match) return null;
  const [year = NaN, month = NaN, day = NaN, hours = NaN, minutes = NaN, seconds = NaN] = match
    .slice(1)
    .map(Number);
  const date = new Date(Date.UTC(year, month - 1, day, hours, minutes, seconds));
  return Number.isNaN(date.getTime()) ? null : date;
}

export function parseLocation(value: unknown): GeoPoint | null {
  let latitude: number | null = null;
  let longitude: number | null = null;
  if (typeof value === "string") {
    const match = COORDINATES_PATTERN.exec(value);
    if (match) {
      latitude = Number(match[1]);
      longitude = Number(match[2]);
    }
  } else if (isRecord(value)) {
    latitude = coordinate(value["Latitude"] ?? value["latitude"]);
    longitude = coordinate(value["Longitude"] ?? value["longitude"]);
  }
  if (latitude === null || longitude === null) return null;
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  if (latitude === 0 && longitude === 0) return null;
  if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
  return { latitude, longitude };
}

function coordinate(value: unknown): number | null {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.trim() !== "") return Number(value);
  return null;
}
