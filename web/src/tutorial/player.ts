import { byId } from "../ui/dom";
import { t } from "../ui/i18n";
import {
  type Box,
  chapterAt,
  formatClock,
  type Point,
  parseBox,
  SECONDS_PER_STEP,
  toKeyframes,
  zoomFor,
} from "./timeline";

const EASING = "cubic-bezier(.4,0,.2,1)";
const FADE = 0.35;
const RING_BORDER_PX = 3;
const AUTOPLAY_VISIBILITY = 0.5;

interface Slide {
  element: HTMLElement;
  shot: HTMLElement;
  ring: HTMLElement;
  ripple: HTMLElement;
  box: Box | null;
}

function readSlides(stage: HTMLElement): Slide[] {
  return Array.from(stage.querySelectorAll<HTMLElement>("[data-slide]")).flatMap((element) => {
    const shot = element.querySelector<HTMLElement>(".shot");
    const ring = element.querySelector<HTMLElement>(".ring");
    const ripple = element.querySelector<HTMLElement>(".ripple");
    if (!shot || !ring || !ripple) return [];
    return [{ element, shot, ring, ripple, box: parseBox(element.dataset["box"] ?? "") }];
  });
}

function placeHighlight(slide: Slide, scale: number): void {
  const { box, ring, ripple } = slide;
  if (!box) return;
  ring.style.left = `${box.left}%`;
  ring.style.top = `${box.top}%`;
  ring.style.width = `${box.right - box.left}%`;
  ring.style.height = `${box.bottom - box.top}%`;
  ring.style.borderWidth = `${RING_BORDER_PX / scale}px`;
  ripple.style.left = `${(box.left + box.right) / 2}%`;
  ripple.style.top = `${(box.top + box.bottom) / 2}%`;
}

function slideAnimations(slide: Slide, index: number, count: number, duration: number): Animation[] {
  const start = index * SECONDS_PER_STEP;
  const end = start + SECONDS_PER_STEP;
  const zoom = slide.box ? zoomFor(slide.box) : { scale: 1, transform: "none" };
  placeHighlight(slide, zoom.scale);
  const tracks: Array<[HTMLElement, Point[]]> = [
    [
      slide.element,
      [
        [start, { opacity: index === 0 ? 1 : 0 }],
        [start + FADE, { opacity: 1 }],
        [end - FADE, { opacity: 1 }],
        [end, { opacity: index === count - 1 ? 1 : 0 }],
      ],
    ],
    [
      slide.shot,
      [
        [start + 0.7, { transform: "translate(0%, 0%) scale(1)" }],
        [start + 1.7, { transform: zoom.transform }],
      ],
    ],
  ];
  if (slide.box) {
    tracks.push(
      [
        slide.ring,
        [
          [start + 1.6, { opacity: 0 }],
          [start + 2, { opacity: 1 }],
        ],
      ],
      [
        slide.ripple,
        [
          [start + 2.2, { opacity: 0, transform: "translate(-50%, -50%) scale(0.2)" }],
          [start + 2.3, { opacity: 0.8, transform: "translate(-50%, -50%) scale(0.4)" }],
          [start + 3, { opacity: 0, transform: "translate(-50%, -50%) scale(2.4)" }],
        ],
      ],
    );
  }
  return tracks.map(([element, points]) => {
    const animation = element.animate(toKeyframes(points, duration, EASING), {
      duration: duration * 1000,
      fill: "both",
    });
    animation.pause();
    return animation;
  });
}

export function setupTutorial(): void {
  const player = byId("player");
  const stage = byId("stage");
  const caption = byId("caption");
  const stepLabel = byId("step-label");
  const playButton = byId<HTMLButtonElement>("play-button");
  const scrubber = byId<HTMLInputElement>("scrubber");
  const clock = byId("time");
  const snapLink = byId("snap-link");
  const chapterButtons = Array.from(player.querySelectorAll<HTMLButtonElement>("[data-chapter]"));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const slides = readSlides(stage);
  const duration = slides.length * SECONDS_PER_STEP;
  const animations = slides.flatMap((slide, index) => slideAnimations(slide, index, slides.length, duration));
  scrubber.max = String(duration);

  let time = 0;
  let playing = false;
  let pausedByUser = false;
  let lastFrame = 0;
  let shownChapter = -1;

  const seek = (seconds: number): void => {
    time = Math.min(Math.max(seconds, 0), duration);
    for (const animation of animations) animation.currentTime = time * 1000;
    scrubber.value = time.toFixed(1);
    clock.textContent = `${formatClock(time)} / ${formatClock(duration)}`;
    const chapter = chapterAt(time, slides.length);
    if (chapter === shownChapter) return;
    shownChapter = chapter;
    const button = chapterButtons[chapter];
    const title = button?.textContent?.replace(/^\d+/, "") ?? "";
    stepLabel.textContent = `${t("stepLabel", { n: chapter + 1, total: slides.length })} · ${title}`;
    caption.textContent = button?.dataset["caption"] ?? "";
    snapLink.hidden = !slides[chapter]?.element.hasAttribute("data-snapchat");
    chapterButtons.forEach((item, index) => {
      item.classList.toggle("active", index === chapter);
      item.setAttribute("aria-current", index === chapter ? "step" : "false");
    });
  };

  const tick = (now: number): void => {
    if (!playing) return;
    const next = time + (now - lastFrame) / 1000;
    lastFrame = now;
    seek(next >= duration ? 0 : next);
    requestAnimationFrame(tick);
  };

  const setPlaying = (next: boolean): void => {
    playing = next;
    player.classList.toggle("playing", next);
    playButton.setAttribute("aria-label", playButton.dataset[next ? "labelPause" : "labelPlay"] ?? "");
    if (!next) return;
    lastFrame = performance.now();
    requestAnimationFrame(tick);
  };

  playButton.addEventListener("click", () => {
    pausedByUser = playing;
    setPlaying(!playing);
  });
  scrubber.addEventListener("input", () => seek(Number(scrubber.value)));
  chapterButtons.forEach((button, index) => {
    button.addEventListener("click", () => {
      seek(index * SECONDS_PER_STEP);
      pausedByUser = false;
      if (!playing) setPlaying(true);
    });
  });

  const observer = new IntersectionObserver(
    ([entry]) => {
      if (!entry) return;
      const mostlyVisible = entry.intersectionRatio >= AUTOPLAY_VISIBILITY;
      if (mostlyVisible && !playing && !pausedByUser && !reducedMotion) setPlaying(true);
      if (!entry.isIntersecting && playing) setPlaying(false);
    },
    { threshold: [0, AUTOPLAY_VISIBILITY] },
  );
  seek(0);
  observer.observe(stage);
}
