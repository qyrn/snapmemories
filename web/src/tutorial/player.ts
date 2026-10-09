import { byId } from "../ui/dom";
import { language } from "../ui/i18n";
import { buildCounters, buildTracks } from "./script";
import {
  CHAPTER_STARTS,
  chapterAt,
  chapterRestingTime,
  counterValue,
  DURATION_SECONDS,
  formatClock,
  toKeyframes,
} from "./timeline";

const REDUCED_MOTION_CHAPTER_MS = 6500;
const DEFAULT_EASING = "cubic-bezier(.4,0,.2,1)";

export function setupTutorial(): void {
  const player = byId("player");
  const stage = byId("stage");
  const caption = byId("caption");
  const playButton = byId<HTMLButtonElement>("play-button");
  const scrubber = byId<HTMLInputElement>("scrubber");
  const clock = byId("time");
  const chapterButtons = Array.from(player.querySelectorAll<HTMLButtonElement>("[data-chapter]"));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const counters = buildCounters(language()).flatMap((counter) => {
    const node = stage.querySelector(counter.target);
    return node ? [{ counter, node }] : [];
  });

  let animations: Animation[] = [];
  let time = 0;
  let playing = false;
  let pausedByUser = false;
  let lastFrame = 0;
  let stepTimer = 0;
  let shownChapter = -1;

  const seek = (seconds: number): void => {
    time = Math.min(Math.max(seconds, 0), DURATION_SECONDS);
    for (const animation of animations) animation.currentTime = time * 1000;
    for (const { counter, node } of counters) node.textContent = counter.format(counterValue(counter, time));
    scrubber.value = time.toFixed(1);
    clock.textContent = `${formatClock(time)} / ${formatClock(DURATION_SECONDS)}`;
    const chapter = chapterAt(time);
    if (chapter !== shownChapter) {
      shownChapter = chapter;
      caption.textContent = chapterButtons[chapter]?.dataset["caption"] ?? "";
      chapterButtons.forEach((button, index) => {
        button.classList.toggle("active", index === chapter);
        button.setAttribute("aria-current", index === chapter ? "step" : "false");
      });
    }
  };

  const tick = (now: number): void => {
    if (!playing) return;
    const elapsed = (now - lastFrame) / 1000;
    lastFrame = now;
    seek(time + elapsed >= DURATION_SECONDS ? 0 : time + elapsed);
    requestAnimationFrame(tick);
  };

  const setPlaying = (next: boolean): void => {
    playing = next;
    player.classList.toggle("playing", next);
    playButton.setAttribute("aria-label", playButton.dataset[next ? "labelPause" : "labelPlay"] ?? "");
    window.clearInterval(stepTimer);
    if (!next) return;
    if (reducedMotion) {
      stepTimer = window.setInterval(() => {
        seek(chapterRestingTime((chapterAt(time) + 1) % CHAPTER_STARTS.length));
      }, REDUCED_MOTION_CHAPTER_MS);
      return;
    }
    lastFrame = performance.now();
    requestAnimationFrame(tick);
  };

  const build = (): void => {
    animations = buildTracks(stage).flatMap((track) => {
      const node = stage.querySelector(track.target);
      if (!node) return [];
      const animation = node.animate(toKeyframes(track.points, track.easing ?? DEFAULT_EASING), {
        duration: DURATION_SECONDS * 1000,
        fill: "both",
      });
      animation.pause();
      return [animation];
    });
    seek(reducedMotion ? chapterRestingTime(0) : 0);
  };

  playButton.addEventListener("click", () => {
    pausedByUser = playing;
    setPlaying(!playing);
  });
  scrubber.addEventListener("input", () => seek(Number(scrubber.value)));
  chapterButtons.forEach((button, index) => {
    button.addEventListener("click", () => {
      seek(reducedMotion ? chapterRestingTime(index) : (CHAPTER_STARTS[index] ?? 0));
      if (!playing && !reducedMotion) {
        pausedByUser = false;
        setPlaying(true);
      }
    });
  });

  const observer = new IntersectionObserver(
    ([entry]) => {
      if (!entry) return;
      if (entry.isIntersecting && !playing && !pausedByUser && !reducedMotion) setPlaying(true);
      if (!entry.isIntersecting && playing) setPlaying(false);
    },
    { threshold: 0.5 },
  );

  document.fonts.ready.then(() => {
    build();
    observer.observe(stage);
  });
}
