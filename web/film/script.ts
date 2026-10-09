import { type Language, STRINGS } from "../src/ui/strings.ts";
import { FILM_TEXT } from "./content/film-text.ts";
import { SNAPCHAT_TEXT } from "./content/snapchat-text.ts";
import { createCursor, placeCursor, type Waypoint } from "./engine/cursor.ts";
import { h, icon, svgUse } from "./engine/dom.ts";
import { Timeline } from "./engine/timeline.ts";
import { createBrowser, ZIP_ICON } from "./scenes/browser.ts";
import { createCaption } from "./scenes/caption.ts";
import { createExplorer, createFolderDialog, createMail } from "./scenes/desktop.ts";
import { createSitePage } from "./scenes/site-page.ts";
import { createSnapchatPage } from "./scenes/snapchat-page.ts";

const SNAP_URL = "accounts.snapchat.com/v2/download-my-data";
const SITE_URL = "memories.qyrn.dev";
const TYPE_SPEED = 0.045;
const SWITCH_OFF = "rgb(83, 87, 91)";
const SWITCH_ON = "rgb(32, 160, 120)";
const END = 51;
const STAGE_WIDTH = 1920;
const STAGE_HEIGHT = 1080;

function focus(x: number, y: number, zoom: number): Keyframe {
  const clamp = (value: number, size: number): number => Math.min(0, Math.max(size - size * zoom, value));
  const left = clamp(STAGE_WIDTH / 2 - x * zoom, STAGE_WIDTH);
  const top = clamp(STAGE_HEIGHT / 2 - y * zoom, STAGE_HEIGHT);
  return { transform: `translate(${left.toFixed(1)}px, ${top.toFixed(1)}px) scale(${zoom})` };
}

export interface Film {
  duration: number;
  seek: (seconds: number) => void;
  sounds: Timeline["sounds"];
}

function typed(value: string, start: number, seconds: number): string {
  const count = Math.max(0, Math.min(value.length, Math.floor((seconds - start) / TYPE_SPEED)));
  return value.slice(0, count);
}

function typeSounds(timeline: Timeline, value: string, start: number): void {
  for (let index = 0; index < value.length; index += 1) {
    timeline.sound(start + index * TYPE_SPEED, "type", 0.3 + (index % 4) * 0.06);
  }
}

function fade(
  from: number,
  to: number,
  enter = "translateY(30px) scale(0.97)",
  exit = "translateY(-20px) scale(0.98)",
): Array<[number, Keyframe]> {
  return [
    [from, { opacity: 0, transform: enter }],
    [from + 0.45, { opacity: 1, transform: "none" }],
    [to - 0.4, { opacity: 1, transform: "none" }],
    [to, { opacity: 0, transform: exit }],
  ];
}

export function buildFilm(stage: HTMLElement, language: Language): Film {
  const site = STRINGS[language];
  const film = FILM_TEXT[language];
  const snap = SNAPCHAT_TEXT[language];
  const scene = new Timeline();
  const cursorTimeline = new Timeline();

  const intro = h(
    "div",
    "f-card",
    svgUse("logo-mark"),
    h("h1", "", film.introTitle),
    h("p", "", film.introSubtitle),
  );
  const outro = h(
    "div",
    "f-card",
    svgUse("logo-mark"),
    h("span", "f-big-url", film.outroTitle),
    h("p", "", film.outroSubtitle),
  );
  const watermark = h(
    "div",
    "f-watermark",
    svgUse("logo-mark"),
    h("span", "", "Snap", h("em", "", "Memories")),
  );
  const browser = createBrowser(film.newTab, film.siteTab, {
    recent: film.recentDownloads,
    done: film.downloadDone,
  });
  const snapPage = createSnapchatPage(snap);
  const sitePage = createSitePage(
    site,
    language === "fr" ? { file: "21,7 Mo", total: "21,8 Mo" } : { file: "21.7 MB", total: "21.8 MB" },
  );
  browser.content.append(snapPage.root, sitePage.root);
  const callout = h("div", "f-callout", film.requestSent);
  const mail = createMail(film);
  const dialog = createFolderDialog(film);
  const explorer = createExplorer(film);
  const caption = createCaption(
    scene,
    film.stepWord,
    [
      { start: 3.6, title: site.step1, text: site.step1Caption },
      { start: 8.6, title: site.step2, text: site.step2Caption },
      { start: 11.4, title: site.step3, text: site.step3Caption },
      { start: 15.2, title: site.step4, text: site.step4Caption },
      { start: 19.4, title: site.step5, text: site.step5Caption },
      { start: 26.8, title: site.step6, text: site.step6Caption },
      { start: 31.6, title: site.step7, text: site.step7Caption },
      { start: 40.6, title: site.step8, text: site.step8Caption },
    ],
    46.9,
  );
  const cursor = createCursor();
  cursor.carry.append(icon(ZIP_ICON), "mydata~1791510669777.zip");
  const camera = h("div", "f-camera", browser.root, callout, dialog.root, cursor.root);
  stage.append(watermark, intro, camera, mail.root, explorer.root, outro, caption);

  const whole = focus(STAGE_WIDTH / 2, STAGE_HEIGHT / 2, 1);
  scene.animate(camera, [
    [3, whole],
    [3.9, whole],
    [4.4, focus(700, 110, 1.45)],
    [6.1, focus(700, 110, 1.45)],
    [6.8, focus(820, 300, 1.3)],
    [7.7, focus(820, 300, 1.3)],
    [8.7, focus(1180, 420, 1.5)],
    [13.9, focus(1180, 420, 1.5)],
    [14.7, whole],
    [19.3, whole],
    [19.9, focus(1250, 360, 1.45)],
    [24.5, focus(1250, 360, 1.45)],
    [25.1, focus(700, 110, 1.4)],
    [26.6, focus(700, 110, 1.4)],
    [27.3, focus(1030, 430, 1.15)],
    [30.4, focus(1030, 430, 1.15)],
    [31, focus(800, 500, 1.3)],
    [40.3, focus(800, 500, 1.3)],
    [40.7, whole],
  ]);

  scene
    .animate(intro, fade(0, 3.2, "scale(0.94)", "scale(1.04)"))
    .animate(outro, [
      [47, { opacity: 0, transform: "scale(0.94)" }],
      [47.45, { opacity: 1, transform: "none" }],
    ])
    .animate(watermark, [
      [3.2, { opacity: 0 }],
      [3.6, { opacity: 1 }],
      [46.8, { opacity: 1 }],
      [47.1, { opacity: 0 }],
    ])
    .animate(browser.root, [
      [3, { opacity: 0, transform: "translateY(60px) scale(0.95)" }],
      [3.6, { opacity: 1, transform: "none" }],
      [15, { opacity: 1, transform: "none" }],
      [15.5, { opacity: 0, transform: "scale(0.92)" }],
      [18.8, { opacity: 0, transform: "scale(0.92)" }],
      [19.3, { opacity: 1, transform: "none" }],
      [40.4, { opacity: 1, transform: "none" }],
      [40.9, { opacity: 0, transform: "translateX(-120px) scale(0.94)" }],
    ])
    .sound(0.15, "whoosh", 0.7)
    .sound(0.6, "chime", 0.5)
    .sound(3, "whoosh", 0.6)
    .sound(15.05, "whoosh", 0.5)
    .sound(18.8, "whoosh", 0.5)
    .sound(40.45, "whoosh", 0.6)
    .sound(46.85, "whoosh", 0.6)
    .sound(47.4, "chime", 0.7);

  scene
    .text(browser.url, (seconds) =>
      seconds < 25 ? typed(SNAP_URL, 4, seconds) : typed(SITE_URL, 25.4, seconds),
    )
    .text(browser.snapTab.title, (seconds) => (seconds < 6.3 ? film.newTab : snap.pageTitle))
    .text(browser.siteTab.title, (seconds) => (seconds < 26.7 ? film.newTab : film.siteTab))
    .animate(browser.caret, [
      [3.9, { opacity: 0 }],
      [4, { opacity: 1 }],
      [6.1, { opacity: 1 }],
      [6.15, { opacity: 0 }],
      [25.2, { opacity: 0 }],
      [25.3, { opacity: 1 }],
      [26.55, { opacity: 1 }],
      [26.6, { opacity: 0 }],
    ])
    .onSeek((seconds) => {
      const onSite = seconds >= 25;
      browser.siteTab.root.hidden = !onSite;
      browser.snapTab.root.classList.toggle("active", !onSite);
      browser.siteTab.root.classList.toggle("active", onSite);
    })
    .sound(6.1, "click", 0.6)
    .sound(25, "pop", 0.4)
    .sound(26.6, "click", 0.6);
  typeSounds(scene, SNAP_URL, 4);
  typeSounds(scene, SITE_URL, 25.4);

  scene.animate(snapPage.root, [
    [6.2, { opacity: 0 }],
    [6.6, { opacity: 1 }],
    [24.99, { opacity: 1 }],
    [25, { opacity: 0 }],
  ]);

  const selectOffset = snapPage.selectSection.offsetTop - 30;
  snapPage.exportsCard.classList.add("visible");
  const exportsOffset = Math.max(0, snapPage.exportsCard.offsetTop - 170);
  snapPage.exportsCard.classList.remove("visible");
  snapPage.panel.style.height = "auto";
  const panelHeight = snapPage.panel.offsetHeight;
  snapPage.panel.style.height = "";

  scene
    .animate(snapPage.scroller, [
      [7.7, { transform: "translateY(0)" }],
      [8.7, { transform: `translateY(${-selectOffset}px)` }],
      [18.79, { transform: `translateY(${-selectOffset}px)` }],
      [18.8, { transform: "translateY(0)" }],
      [19.5, { transform: "translateY(0)" }],
      [20.2, { transform: `translateY(${-exportsOffset}px)` }],
    ])
    .animate(snapPage.memoriesSwitch, [
      [10, { backgroundColor: SWITCH_OFF }],
      [10.18, { backgroundColor: SWITCH_ON }],
    ])
    .animate(snapPage.memoriesSwitch.querySelector(".sc-switch-knob") ?? snapPage.memoriesSwitch, [
      [10, { transform: "translateX(0)" }],
      [10.18, { transform: "translateX(20px)" }],
    ])
    .animate(snapPage.memoriesSwitch.querySelector(".sc-switch-on") ?? snapPage.memoriesSwitch, [
      [10.05, { opacity: 0 }],
      [10.18, { opacity: 1 }],
    ])
    .animate(snapPage.memoriesSwitch.querySelector(".sc-switch-off") ?? snapPage.memoriesSwitch, [
      [10, { opacity: 1 }],
      [10.1, { opacity: 0 }],
    ])
    .text(snapPage.counter, (seconds) => (seconds >= 10.05 ? snap.selectedOne : snap.selectedNone))
    .animate(snapPage.panel, [
      [10.15, { height: "0px" }],
      [10.75, { height: `${panelHeight}px` }],
    ])
    .toggleClass(12.45, snapPage.requestButton, "hover", 13.3)
    .animate(snapPage.requestButton, [
      [12.88, { transform: "scale(1)" }],
      [12.98, { transform: "scale(0.96)" }],
      [13.12, { transform: "scale(1)" }],
    ])
    .animate(callout, fade(13.1, 14.9, "translateY(16px) scale(0.9)"))
    .sound(10.25, "whoosh", 0.25)
    .sound(13.2, "chime", 0.6);

  scene
    .animate(mail.root, fade(15.2, 18.8, "scale(0.96)", "scale(1.03)"))
    .animate(mail.card, [
      [15.4, { transform: "translateY(-70px)" }],
      [15.9, { transform: "translateY(0)" }],
    ])
    .animate(
      mail.hand,
      [
        [15.6, { transform: "rotate(180deg)" }],
        [18.4, { transform: "rotate(1260deg)" }],
      ],
      "linear",
    )
    .sound(15.75, "notify", 0.8);

  scene
    .onSeek((seconds) => {
      snapPage.exportsCard.classList.toggle("visible", seconds >= 18.8);
      snapPage.exportsList.classList.toggle("visible", seconds >= 21.05);
    })
    .text(snapPage.seeLabel, (seconds) => (seconds >= 21.05 ? snap.hideExports : snap.seeExports))
    .animate(browser.downloadsPanel, [
      [22.75, { opacity: 0, transform: "scale(0.92)" }],
      [23.05, { opacity: 1, transform: "scale(1)" }],
      [28.35, { opacity: 1, transform: "scale(1)" }],
      [28.6, { opacity: 0, transform: "scale(0.96)" }],
    ])
    .animate(
      browser.downloadBar,
      [
        [23, { width: "0%" }],
        [24.2, { width: "100%" }],
      ],
      "linear",
    )
    .animate(browser.downloadsRing, [
      [22.8, { opacity: 0, clipPath: "inset(0 0 0 100%)" }],
      [22.9, { opacity: 1, clipPath: "inset(0 0 0 100%)" }],
      [24.2, { opacity: 1, clipPath: "inset(0 0 0 0)" }],
      [24.5, { opacity: 0, clipPath: "inset(0 0 0 0)" }],
    ])
    .text(browser.downloadStatus, (seconds) => {
      if (seconds >= 24.2) return film.downloadDone;
      const done = Math.max(0, Math.min(1, (seconds - 23) / 1.2)) * 21.7;
      const value = language === "fr" ? done.toFixed(1).replace(".", ",") : done.toFixed(1);
      return language === "fr" ? `${value} / 21,7 Mo` : `${value} / 21.7 MB`;
    })
    .toggleClass(27.75, browser.downloadRow, "hover", 28.35)
    .sound(24.2, "ding", 0.7);

  const screens = sitePage.screens;
  const progressStart = 35.6;
  const progressEnd = 39.4;
  scene
    .animate(sitePage.root, [
      [26.7, { opacity: 0 }],
      [27, { opacity: 1 }],
    ])
    .onSeek((seconds) => {
      const current =
        seconds >= 39.6 ? "done" : seconds >= 35.5 ? "progress" : seconds >= 31 ? "summary" : "drop";
      screens.drop.classList.toggle("active", current === "drop");
      screens.summary.classList.toggle("active", current === "summary");
      screens.progress.classList.toggle("active", current === "progress");
      screens.done.classList.toggle("active", current === "done");
      sitePage.dropzone.classList.toggle("dragover", seconds >= 29.3 && seconds < 30);
      sitePage.reading.classList.toggle("visible", seconds >= 30.05 && seconds < 31);
      cursor.carry.classList.toggle("visible", seconds >= 28.1 && seconds < 30);
      dialog.target.classList.toggle("selected", seconds >= 34.2);
      dialog.selectButton.classList.toggle("hover", seconds >= 34.9 && seconds < 35.3);
      sitePage.folderButton.classList.toggle("hover", seconds >= 32.6 && seconds < 33.1);
      explorer.thumbs[0]?.classList.toggle("selected", seconds >= 43.1);
    })
    .animate(
      sitePage.progressBar,
      [
        [progressStart, { width: "0%" }],
        [progressEnd, { width: "100%" }],
      ],
      "linear",
    )
    .counter(sitePage.progressPercent, 0, 100, progressStart, progressEnd, (value) => `${Math.round(value)}%`)
    .counter(sitePage.progressSaved, 0, 16, progressStart, progressEnd, (value) => String(Math.round(value)))
    .counter(sitePage.progressLabel, 0, 16, progressStart, progressEnd, (value) =>
      site.progressLabel.replace("{done}", String(Math.round(value))).replace("{total}", "16"),
    )
    .animate(dialog.root, fade(33.2, 35.55, "translateY(24px) scale(0.97)", "scale(0.98)"))
    .sound(30, "drop", 0.9)
    .sound(31, "pop", 0.4)
    .sound(33.25, "whoosh", 0.35)
    .sound(39.6, "chime", 0.8);
  for (let time = progressStart; time < progressEnd; time += 0.24) scene.sound(time, "tick", 0.18);

  scene.animate(explorer.root, fade(40.6, 47, "translateY(30px) scale(0.97)", "scale(0.98)"));
  explorer.thumbs.forEach((thumb, index) => {
    const time = 41.1 + index * 0.18;
    scene
      .animate(thumb, [
        [
          time,
          { opacity: 0, transform: "translateY(18px) scale(0.9)", filter: "saturate(0) brightness(1.4)" },
        ],
        [time + 0.35, { opacity: 1, transform: "none", filter: "saturate(1) brightness(1)" }],
      ])
      .sound(time + 0.05, "pop", 0.3);
  });
  scene.animate(explorer.info, fade(43.25, 46.9, "translateY(16px) scale(0.95)")).sound(43.3, "pop", 0.5);

  scene.duration = END;
  scene.build();
  scene.seek(12.9);
  const button = snapPage.requestButton.getBoundingClientRect();
  const frame = camera.getBoundingClientRect();
  const zoom = frame.width / STAGE_WIDTH;
  callout.style.left = `${(button.right - frame.left) / zoom + 24}px`;
  callout.style.top = `${(button.top - frame.top) / zoom - 6}px`;

  const waypoints: Waypoint[] = [
    { start: 6.4, at: 6.4, target: [980, 760] },
    { start: 8.8, at: 9.8, target: snapPage.memoriesSwitch, hand: true, click: 10 },
    { start: 11.6, at: 12.5, target: snapPage.requestButton, hand: true, click: 12.9 },
    { start: 13.4, at: 14.2, target: [1260, 820] },
    { start: 19.3, at: 19.3, target: [1180, 800] },
    { start: 20.2, at: 20.8, target: snapPage.seeButton, hand: true, click: 21 },
    { start: 21.5, at: 22.3, target: snapPage.downloadButton, hand: true, click: 22.6 },
    { start: 23.1, at: 23.9, target: [1320, 720] },
    { start: 27.1, at: 27.1, target: [1100, 720] },
    { start: 27.2, at: 27.9, target: browser.downloadRow, anchor: [0.3, 0.5], click: 28 },
    { start: 28.3, at: 29.7, target: sitePage.dropzone },
    { start: 31.8, at: 32.75, target: sitePage.folderButton, hand: true, click: 33 },
    { start: 33.6, at: 34.05, target: dialog.target, click: 34.2 },
    { start: 34.4, at: 35, target: dialog.selectButton, hand: true, click: 35.2 },
    { start: 35.3, at: 35.6, target: [1500, 860] },
    { start: 41.8, at: 41.8, target: [1240, 820] },
    { start: 42.2, at: 42.95, target: explorer.thumbs[0] ?? [500, 300], click: 43.1 },
  ];
  placeCursor(camera, scene, cursorTimeline, cursor, waypoints, [
    [6.4, 14.8],
    [19.3, 24.9],
    [27.1, 35.6],
    [41.8, 46.7],
  ]);
  cursorTimeline.duration = END;
  cursorTimeline.build();

  return {
    duration: END,
    seek: (seconds) => {
      scene.seek(seconds);
      cursorTimeline.seek(seconds);
    },
    sounds: [...scene.sounds, ...cursorTimeline.sounds].sort((a, b) => a.time - b.time),
  };
}
