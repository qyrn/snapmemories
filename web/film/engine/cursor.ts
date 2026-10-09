import { h, icon } from "./dom.ts";
import type { Timeline } from "./timeline.ts";

const ARROW = `<svg viewBox="0 0 32 32"><path d="M6 3.5v21.2l5.3-5 3.6 8.3 3.6-1.6-3.6-8.1 7.3-.4z" fill="#fff" stroke="#111" stroke-width="1.8" stroke-linejoin="round"/></svg>`;
const HAND = `<svg viewBox="0 0 32 32"><path d="M12.5 4.5c-1.2 0-2 .9-2 2v11.2l-2-2.1c-.9-.9-2.3-.9-3.1 0-.8.8-.8 2 0 2.9l6.1 6.9c1.3 1.5 3.2 2.3 5.2 2.3h2.6c3.7 0 6.7-3 6.7-6.7v-6.3c0-1.1-.9-2-2-2s-1.9.9-1.9 2v-1.3c0-1.1-.9-2-2-2s-2 .9-2 2v-.9c0-1.1-.9-2-2-2s-2 .9-2 2V6.5c0-1.1-.8-2-1.6-2z" fill="#fff" stroke="#111" stroke-width="1.7" stroke-linejoin="round"/></svg>`;
const MOVE_X = "cubic-bezier(.55,0,.25,1)";
const MOVE_Y = "cubic-bezier(.3,0,.15,1)";

export interface Waypoint {
  start: number;
  at: number;
  target: Element | [x: number, y: number];
  anchor?: [x: number, y: number];
  hand?: boolean;
  click?: number;
}

export interface Cursor {
  root: HTMLElement;
  carry: HTMLElement;
}

export function createCursor(): Cursor & { mover: HTMLElement; pointer: HTMLElement; ripple: HTMLElement } {
  const ripple = h("span", "f-ripple");
  const carry = h("span", "f-carry");
  const pointer = h("span", "f-pointer", icon(ARROW, "f-arrow"), icon(HAND, "f-hand"));
  const mover = h("span", "f-cursor-y", ripple, carry, pointer);
  const root = h("span", "f-cursor-x", mover);
  return { root, carry, mover, pointer, ripple };
}

export function placeCursor(
  space: HTMLElement,
  scene: Timeline,
  cursorTimeline: Timeline,
  parts: ReturnType<typeof createCursor>,
  waypoints: Waypoint[],
  visibility: Array<[from: number, to: number]>,
): void {
  const positions = waypoints.map((waypoint) => {
    if (Array.isArray(waypoint.target)) return waypoint.target;
    scene.seek(waypoint.at);
    const box = space.getBoundingClientRect();
    const scale = box.width / space.offsetWidth;
    const rect = waypoint.target.getBoundingClientRect();
    const [ax, ay] = waypoint.anchor ?? [0.5, 0.5];
    return [
      (rect.left - box.left + rect.width * ax) / scale,
      (rect.top - box.top + rect.height * ay) / scale,
    ] as [number, number];
  });

  const xs: Array<[number, Keyframe]> = [];
  const ys: Array<[number, Keyframe]> = [];
  const presses: Array<[number, Keyframe]> = [];
  const ripples: Array<[number, Keyframe]> = [];
  waypoints.forEach((waypoint, index) => {
    const [x, y] = positions[index] ?? [0, 0];
    const [px, py] = positions[index - 1] ?? [x, y];
    xs.push(
      [waypoint.start, { transform: `translateX(${px}px)` }],
      [waypoint.at, { transform: `translateX(${x}px)` }],
    );
    ys.push(
      [waypoint.start, { transform: `translateY(${py}px)` }],
      [waypoint.at, { transform: `translateY(${y}px)` }],
    );
    const next = waypoints[index + 1];
    if (waypoint.hand) {
      const until = next ? next.start + 0.15 : Number.POSITIVE_INFINITY;
      cursorTimeline.at(waypoint.at - 0.12, (active) => {
        if (active) parts.pointer.dataset["hand"] = String(waypoint.at);
        else if (parts.pointer.dataset["hand"] === String(waypoint.at)) delete parts.pointer.dataset["hand"];
      });
      cursorTimeline.at(until, (active) => {
        if (active && parts.pointer.dataset["hand"] === String(waypoint.at))
          delete parts.pointer.dataset["hand"];
      });
    }
    if (waypoint.click !== undefined) {
      const time = waypoint.click;
      presses.push(
        [time - 0.01, { transform: "scale(1)" }],
        [time + 0.08, { transform: "scale(0.8)" }],
        [time + 0.24, { transform: "scale(1)" }],
      );
      ripples.push(
        [time, { opacity: 0, transform: "translate(-50%, -50%) scale(0.2)" }],
        [time + 0.05, { opacity: 0.9, transform: "translate(-50%, -50%) scale(0.5)" }],
        [time + 0.55, { opacity: 0, transform: "translate(-50%, -50%) scale(2.2)" }],
      );
      cursorTimeline.sound(time, "click", 0.9);
    }
  });
  cursorTimeline
    .animate(parts.root, xs, MOVE_X)
    .animate(parts.mover, ys, MOVE_Y)
    .animate(parts.pointer, presses)
    .animate(parts.ripple, ripples);
  const fades: Array<[number, Keyframe]> = [];
  for (const [from, to] of visibility) {
    fades.push(
      [from, { opacity: 0 }],
      [from + 0.25, { opacity: 1 }],
      [to - 0.25, { opacity: 1 }],
      [to, { opacity: 0 }],
    );
  }
  cursorTimeline.animate(parts.pointer.parentElement ?? parts.pointer, fades, "linear");
}
