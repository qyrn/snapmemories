const MONTH_NAMES = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"];

export interface LocalMoment {
  year: number;
  month: number;
  day: number;
  hours: string;
  minutes: string;
}

export function formatCount(value: number): string {
  return value.toLocaleString("en-US");
}

export function formatBytes(bytes: number): string {
  let value = Math.max(0, bytes);
  let unit = 0;
  while (value >= 1024 && unit < BYTE_UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || value >= 100 ? 0 : 1;
  return `${value.toFixed(digits)} ${BYTE_UNITS[unit] ?? "B"}`;
}

export function formatDuration(seconds: number): string {
  if (seconds <= 0) return "-";
  if (seconds < 60) return `${Math.ceil(seconds)} s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return `${hours} h ${String(minutes).padStart(2, "0")}`;
}

export function plural(count: number, singular: string, pluralForm = `${singular}s`): string {
  return `${formatCount(count)} ${count === 1 ? singular : pluralForm}`;
}

export function parseLocalMoment(iso: string): LocalMoment | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso);
  if (!match) return null;
  const [, year, month, day, hours, minutes] = match;
  if (!year || !month || !day || !hours || !minutes) return null;
  return { year: Number(year), month: Number(month), day: Number(day), hours, minutes };
}

export function monthLabel(moment: LocalMoment | null): string {
  if (!moment) return "Unknown date";
  return `${MONTH_NAMES[moment.month - 1] ?? ""} ${moment.year}`;
}

export function fullDateLabel(moment: LocalMoment | null): string {
  if (!moment) return "Unknown date";
  return `${moment.day} ${MONTH_NAMES[moment.month - 1] ?? ""} ${moment.year}, ${moment.hours}:${moment.minutes}`;
}
