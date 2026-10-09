export interface LocalTime {
  year: number;
  month: number;
  day: number;
  hours: number;
  minutes: number;
  seconds: number;
  offsetMinutes: number;
}

function pad(value: number, length = 2): string {
  return String(value).padStart(length, "0");
}

export function toLocalTime(moment: Date): LocalTime {
  return {
    year: moment.getFullYear(),
    month: moment.getMonth() + 1,
    day: moment.getDate(),
    hours: moment.getHours(),
    minutes: moment.getMinutes(),
    seconds: moment.getSeconds(),
    offsetMinutes: -moment.getTimezoneOffset(),
  };
}

export function formatOffset(local: LocalTime): string {
  const sign = local.offsetMinutes >= 0 ? "+" : "-";
  const absolute = Math.abs(local.offsetMinutes);
  return `${sign}${pad(Math.floor(absolute / 60))}:${pad(absolute % 60)}`;
}

export function exifDate(local: LocalTime): string {
  return `${local.year}:${pad(local.month)}:${pad(local.day)} ${pad(local.hours)}:${pad(local.minutes)}:${pad(local.seconds)}`;
}

export function isoWithOffset(local: LocalTime): string {
  return `${local.year}-${pad(local.month)}-${pad(local.day)}T${pad(local.hours)}:${pad(local.minutes)}:${pad(local.seconds)}${formatOffset(local)}`;
}

export function fileStem(local: LocalTime): string {
  return `${local.year}-${pad(local.month)}-${pad(local.day)}_${pad(local.hours)}-${pad(local.minutes)}-${pad(local.seconds)}`;
}

export function monthFolder(local: LocalTime): string {
  return `${local.year}/${local.year}-${pad(local.month)}`;
}
