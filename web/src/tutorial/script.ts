import { type Counter, type Point, sceneTrack, type Track } from "./timeline";

const CURSOR_HIDDEN = { opacity: 0, transform: "scale(1)" };
const SNAP_ORANGE = "#f2542d";
const TOGGLE_OFF = "#cfc6b4";
const KNOB_TRAVEL = "translateX(2.3cqw)";

interface Spot {
  left: string;
  top: string;
}

function spot(stage: HTMLElement, selector: string): Spot {
  const target = stage.querySelector(selector);
  if (!target) return { left: "50%", top: "50%" };
  const box = stage.getBoundingClientRect();
  const rect = target.getBoundingClientRect();
  const left = ((rect.left + rect.width / 2 - box.left) / box.width) * 100;
  const top = ((rect.top + rect.height / 2 - box.top) / box.height) * 100;
  return { left: `${left.toFixed(2)}%`, top: `${top.toFixed(2)}%` };
}

function offsetFromCenter(stage: HTMLElement, selector: string): string {
  const target = stage.querySelector(selector);
  if (!target) return "translate(0, 0)";
  const box = stage.getBoundingClientRect();
  const rect = target.getBoundingClientRect();
  const dx = ((rect.left + rect.width / 2 - (box.left + box.width / 2)) / box.width) * 100;
  const dy = ((rect.top + rect.height / 2 - (box.top + box.height / 2)) / box.width) * 100;
  return `translate(${dx.toFixed(2)}cqw, ${dy.toFixed(2)}cqw)`;
}

function cursorTrack(stage: HTMLElement): Track {
  const at = (seconds: number, selector: string, extra: Keyframe = {}): Point => [
    seconds,
    { ...spot(stage, selector), opacity: 1, transform: "scale(1)", ...extra },
  ];
  const press = (seconds: number, selector: string): Point[] => [
    at(seconds, selector),
    at(seconds + 0.12, selector, { transform: "scale(0.8)" }),
    at(seconds + 0.3, selector),
  ];
  return {
    target: '[data-anim="cursor"]',
    easing: "cubic-bezier(.45,0,.2,1)",
    points: [
      [0, { left: "82%", top: "86%", ...CURSOR_HIDDEN }],
      [0.6, { left: "82%", top: "86%", opacity: 1, transform: "scale(1)" }],
      ...press(1.8, '[data-anim="knob-1"]'),
      ...press(3.3, '[data-anim="knob-2"]'),
      ...press(4.8, '[data-anim="range"]'),
      ...press(6.2, '[data-anim="submit"]'),
      at(7.4, '[data-anim="submit"]', { opacity: 0 }),
      [29.2, { left: "78%", top: "84%", ...CURSOR_HIDDEN }],
      [29.5, { left: "78%", top: "84%", opacity: 1, transform: "scale(1)" }],
      ...press(30.3, '[data-anim="choose"]'),
      at(31.4, '[data-anim="choose"]', { opacity: 0 }),
    ],
  };
}

function toggleTracks(index: number, seconds: number): Track[] {
  return [
    {
      target: `[data-anim="toggle-${index}"]`,
      points: [
        [seconds, { backgroundColor: TOGGLE_OFF }],
        [seconds + 0.2, { backgroundColor: SNAP_ORANGE }],
      ],
    },
    {
      target: `[data-anim="knob-${index}"]`,
      points: [
        [seconds, { transform: "translateX(0)" }],
        [seconds + 0.2, { transform: KNOB_TRAVEL }],
      ],
    },
  ];
}

function pressTrack(target: string, seconds: number, after: Keyframe = {}): Track {
  return {
    target,
    points: [
      [seconds, { transform: "scale(1)" }],
      [seconds + 0.12, { transform: "scale(0.92)", ...after }],
      [seconds + 0.3, { transform: "scale(1)", ...after }],
    ],
  };
}

function fillTrack(target: string, start: number, end: number): Track {
  return {
    target,
    easing: "linear",
    points: [
      [start, { width: "0%" }],
      [end, { width: "100%" }],
    ],
  };
}

function appearTrack(target: string, seconds: number, from = "translateY(2cqw)"): Track {
  return {
    target,
    easing: "cubic-bezier(.2,.8,.2,1)",
    points: [
      [seconds, { opacity: 0, transform: from }],
      [seconds + 0.45, { opacity: 1, transform: "translate(0, 0)" }],
    ],
  };
}

function flyingZipTrack(stage: HTMLElement, target: string, start: number): Track {
  const landing = offsetFromCenter(stage, '[data-anim="drop-target"]');
  return {
    target,
    easing: "cubic-bezier(.5,0,.2,1)",
    points: [
      [start, { opacity: 0, transform: "translate(-44cqw, 10cqw) rotate(-18deg)" }],
      [start + 0.2, { opacity: 1, transform: "translate(-40cqw, 8cqw) rotate(-12deg)" }],
      [start + 1.1, { opacity: 1, transform: `${landing} rotate(0deg) scale(0.8)` }],
      [start + 1.3, { opacity: 0, transform: `${landing} rotate(0deg) scale(0.5)` }],
    ],
  };
}

function resultTracks(): Track[] {
  return Array.from({ length: 6 }, (_, index) => ({
    target: `[data-anim="result-${index + 1}"]`,
    easing: "cubic-bezier(.2,.8,.2,1)",
    points: [
      [37.6 + index * 0.45, { opacity: 0, transform: "scale(0.85)", filter: "saturate(0) brightness(1.5)" }],
      [38.2 + index * 0.45, { opacity: 1, transform: "scale(1)", filter: "saturate(1) brightness(1)" }],
    ],
  }));
}

export function buildTracks(stage: HTMLElement): Track[] {
  return [
    ...[1, 2, 3, 4, 5, 6].map(sceneTrack),
    ...toggleTracks(1, 1.9),
    ...toggleTracks(2, 3.4),
    {
      target: '[data-anim="range"]',
      points: [
        [4.9, { borderColor: "#d6ccb8", boxShadow: "0 0 0 0 rgba(242,84,45,0)" }],
        [5.1, { borderColor: SNAP_ORANGE, boxShadow: "0 0 0 0.5cqw rgba(242,84,45,.25)" }],
      ],
    },
    pressTrack('[data-anim="submit"]', 6.3, { backgroundColor: SNAP_ORANGE, color: "#1b1a17" }),
    {
      target: '[data-anim="mail"]',
      easing: "cubic-bezier(.2,.8,.2,1)",
      points: [
        [9, { opacity: 0, transform: "translateY(-14cqw)" }],
        [9.6, { opacity: 1, transform: "translateY(0)" }],
        [11, { opacity: 1, transform: "scale(1)" }],
        [11.25, { opacity: 1, transform: "scale(1.05)" }],
        [11.5, { opacity: 1, transform: "scale(1)" }],
      ],
    },
    fillTrack('[data-anim="download-1"]', 15, 18),
    fillTrack('[data-anim="download-2"]', 15.6, 19.4),
    flyingZipTrack(stage, '[data-anim="zip-a"]', 21.8),
    flyingZipTrack(stage, '[data-anim="zip-b"]', 22.2),
    {
      target: '[data-anim="drop-target"]',
      points: [
        [22.7, { borderColor: "#8a8377", backgroundColor: "rgba(251,228,217,0)" }],
        [23.1, { borderColor: SNAP_ORANGE, backgroundColor: "rgba(251,228,217,1)" }],
      ],
    },
    appearTrack('[data-anim="stats"]', 23.5),
    pressTrack('[data-anim="choose"]', 30.4),
    appearTrack('[data-anim="picker"]', 31),
    appearTrack('[data-anim="save"]', 31.9),
    fillTrack('[data-anim="save-bar"]', 32.2, 36.2),
    ...resultTracks(),
    cursorTrack(stage),
  ];
}

export function buildCounters(language: string): Counter[] {
  const number = new Intl.NumberFormat(language);
  const whole = (value: number): string => number.format(Math.round(value));
  return [
    { target: '[data-counter="photos"]', from: 0, to: 1248, start: 24, end: 26.6, format: whole },
    { target: '[data-counter="videos"]', from: 0, to: 312, start: 24.2, end: 26.8, format: whole },
    {
      target: '[data-counter="percent"]',
      from: 0,
      to: 100,
      start: 32.2,
      end: 36.2,
      linear: true,
      format: (value) => `${Math.round(value)}%`,
    },
  ];
}
