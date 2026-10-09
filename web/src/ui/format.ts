import { language } from "./i18n";

const BYTE_UNITS = { en: ["B", "KB", "MB", "GB", "TB"], fr: ["o", "Ko", "Mo", "Go", "To"] };

export function formatCount(value: number): string {
  return new Intl.NumberFormat(language()).format(value);
}

export function formatBytes(bytes: number): string {
  const units = BYTE_UNITS[language()];
  let value = Math.max(0, bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || value >= 100 ? 0 : 1;
  return `${new Intl.NumberFormat(language(), { maximumFractionDigits: digits }).format(value)} ${units[unit] ?? ""}`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "-";
  if (seconds < 60) return `${Math.ceil(seconds)} s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return `${hours} h ${String(minutes).padStart(2, "0")}`;
}

export function formatDate(moment: Date): string {
  return new Intl.DateTimeFormat(language(), { dateStyle: "medium", timeStyle: "short" }).format(moment);
}
