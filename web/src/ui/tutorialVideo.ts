import { byId } from "./dom";

export function setupTutorialVideo(): void {
  const video = byId<HTMLVideoElement>("tutorial-video");
  const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>("#chapters [data-time]"));
  const starts = buttons.map((button) => Number(button.dataset["time"] ?? 0));

  const mark = (current: number): void => {
    buttons.forEach((button, index) => {
      button.classList.toggle("active", index === current);
      button.setAttribute("aria-current", index === current ? "step" : "false");
    });
  };

  const chapterAt = (seconds: number): number => {
    let current = -1;
    starts.forEach((start, index) => {
      if (seconds >= start - 0.2) current = index;
    });
    return current;
  };

  buttons.forEach((button, index) => {
    button.addEventListener("click", () => {
      video.currentTime = starts[index] ?? 0;
      mark(index);
      void video.play().catch(() => undefined);
    });
  });
  video.addEventListener("timeupdate", () => mark(chapterAt(video.currentTime)));
}
