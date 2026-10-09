export const DURATION_SECONDS = 45;
export const CHAPTER_STARTS = [0, 8, 14, 21, 29, 37];

export type Point = [seconds: number, frame: Keyframe];

export interface Track {
  target: string;
  points: Point[];
  easing?: string;
}

export interface Counter {
  target: string;
  from: number;
  to: number;
  start: number;
  end: number;
  linear?: boolean;
  format: (value: number) => string;
}

const FADE = 0.45;

export function sceneTrack(scene: number): Track {
  const start = CHAPTER_STARTS[scene - 1] ?? 0;
  const end = CHAPTER_STARTS[scene] ?? DURATION_SECONDS;
  const enter = start === 0 ? 0 : start;
  return {
    target: `[data-scene="${scene}"]`,
    points: [
      [enter, { opacity: start === 0 ? 1 : 0, transform: "scale(0.97)" }],
      [enter + FADE, { opacity: 1, transform: "scale(1)" }],
      [end - FADE, { opacity: 1, transform: "scale(1)" }],
      [end, { opacity: end === DURATION_SECONDS ? 1 : 0, transform: "scale(1.02)" }],
    ],
  };
}

export function chapterAt(seconds: number): number {
  let chapter = 0;
  CHAPTER_STARTS.forEach((start, index) => {
    if (seconds >= start) chapter = index;
  });
  return chapter;
}

export function chapterRestingTime(chapter: number): number {
  const end = CHAPTER_STARTS[chapter + 1] ?? DURATION_SECONDS;
  return end - 1;
}

export function toKeyframes(points: Point[], easing: string): Keyframe[] {
  const sorted = [...points].sort((a, b) => a[0] - b[0]);
  const first = sorted[0];
  const last = sorted.at(-1);
  if (!first || !last) return [];
  const padded: Point[] = [
    ...(first[0] > 0 ? [[0, first[1]] as Point] : []),
    ...sorted,
    ...(last[0] < DURATION_SECONDS ? [[DURATION_SECONDS, last[1]] as Point] : []),
  ];
  return padded.map(([seconds, frame]) => ({ ...frame, offset: seconds / DURATION_SECONDS, easing }));
}

export function counterValue(counter: Counter, seconds: number): number {
  if (seconds <= counter.start) return counter.from;
  if (seconds >= counter.end) return counter.to;
  const progress = (seconds - counter.start) / (counter.end - counter.start);
  const eased = counter.linear ? progress : 1 - (1 - progress) ** 3;
  return counter.from + (counter.to - counter.from) * eased;
}

export function formatClock(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}
