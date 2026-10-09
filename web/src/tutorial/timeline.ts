export const SECONDS_PER_STEP = 5.5;

export type Point = [seconds: number, frame: Keyframe];

export interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

const MAX_ZOOM = 1.8;
const TARGET_WIDTH = 0.72;
const TARGET_HEIGHT = 0.55;

export function parseBox(value: string): Box | null {
  const [left, top, right, bottom] = value.split(",").map(Number);
  if (left === undefined || top === undefined || right === undefined || bottom === undefined) return null;
  if (right <= left || bottom <= top) return null;
  return { left, top, right, bottom };
}

export function zoomFor(box: Box): { scale: number; transform: string } {
  const width = (box.right - box.left) / 100;
  const height = (box.bottom - box.top) / 100;
  const scale = Math.min(Math.max(Math.min(TARGET_WIDTH / width, TARGET_HEIGHT / height), 1), MAX_ZOOM);
  const limit = ((scale - 1) / 2) * 100;
  const clamp = (value: number): number => Math.min(Math.max(value, -limit), limit);
  const centerX = (box.left + box.right) / 200;
  const centerY = (box.top + box.bottom) / 200;
  const shiftX = clamp((0.5 - centerX) * scale * 100);
  const shiftY = clamp((0.5 - centerY) * scale * 100);
  return {
    scale,
    transform: `translate(${shiftX.toFixed(2)}%, ${shiftY.toFixed(2)}%) scale(${scale.toFixed(3)})`,
  };
}

export function chapterAt(seconds: number, steps: number): number {
  return Math.min(Math.floor(seconds / SECONDS_PER_STEP), steps - 1);
}

export function toKeyframes(points: Point[], duration: number, easing: string): Keyframe[] {
  const sorted = [...points].sort((a, b) => a[0] - b[0]);
  const first = sorted[0];
  const last = sorted.at(-1);
  if (!first || !last) return [];
  const padded: Point[] = [
    ...(first[0] > 0 ? [[0, first[1]] as Point] : []),
    ...sorted,
    ...(last[0] < duration ? [[duration, last[1]] as Point] : []),
  ];
  return padded.map(([seconds, frame]) => ({ ...frame, offset: seconds / duration, easing }));
}

export function formatClock(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}
