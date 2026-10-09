import { h } from "../engine/dom.ts";
import type { Timeline } from "../engine/timeline.ts";

export interface CaptionStep {
  start: number;
  title: string;
  text: string;
}

export function createCaption(
  timeline: Timeline,
  stepWord: string,
  steps: CaptionStep[],
  end: number,
): HTMLElement {
  const badge = h("span", "f-caption-badge");
  const title = h("strong", "f-caption-title");
  const text = h("span", "f-caption-text");
  const progress = h("span", "f-caption-progress", ...steps.map(() => h("i", "")));
  const root = h("div", "f-caption", badge, h("div", "f-caption-copy", title, text), progress);
  const current = (seconds: number): number => {
    let index = 0;
    steps.forEach((step, position) => {
      if (seconds >= step.start - 0.15) index = position;
    });
    return index;
  };
  timeline
    .text(badge, (seconds) => `${stepWord} ${current(seconds) + 1}/${steps.length}`)
    .text(title, (seconds) => steps[current(seconds)]?.title ?? "")
    .text(text, (seconds) => steps[current(seconds)]?.text ?? "");
  steps.forEach((step, index) => {
    const dot = progress.children[index];
    if (dot) timeline.toggleClass(step.start - 0.15, dot, "on");
  });

  const first = steps[0]?.start ?? 0;
  const frames: Array<[number, Keyframe]> = [
    [first - 0.5, { opacity: 0, transform: "translateY(40px)" }],
    [first, { opacity: 1, transform: "translateY(0)" }],
  ];
  for (const step of steps.slice(1)) {
    frames.push(
      [step.start - 0.3, { opacity: 1, transform: "translateY(0)" }],
      [step.start - 0.15, { opacity: 0, transform: "translateY(10px)" }],
      [step.start + 0.2, { opacity: 1, transform: "translateY(0)" }],
    );
    timeline.sound(step.start, "pop", 0.45);
  }
  frames.push(
    [end - 0.4, { opacity: 1, transform: "translateY(0)" }],
    [end, { opacity: 0, transform: "translateY(40px)" }],
  );
  timeline.animate(root, frames).sound(first, "pop", 0.45);
  return root;
}
