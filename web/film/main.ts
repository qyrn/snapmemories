import "../src/styles.css";
import "./film.css";

import page from "../src/page.html?raw";
import { h } from "./engine/dom.ts";
import { buildFilm, type Film } from "./script.ts";

declare global {
  interface Window {
    film?: Film;
  }
}

const params = new URLSearchParams(window.location.search);
const language = params.get("lang") === "en" ? "en" : "fr";
document.documentElement.lang = language;

const sprite = /<svg class="sprite"[\s\S]*?<\/svg>/.exec(page)?.[0] ?? "";
const holder = h("div");
holder.innerHTML = sprite;
const stage = h("div", "f-stage");
document.body.append(holder, stage);

await document.fonts.ready;
const film = buildFilm(stage, language);
window.film = film;
film.seek(Number(params.get("t") ?? 0));

if (params.has("play")) {
  const startedAt = performance.now() - Number(params.get("t") ?? 0) * 1000;
  const loop = (now: number): void => {
    film.seek(((now - startedAt) / 1000) % film.duration);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
