import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { chromium, type Page } from "@playwright/test";

const ASPECT = 1.6;
const OUTPUT_WIDTH = 1280;
const PADDING = 48;
const PAGES = { en: "/", fr: "/fr/" } as const;

const webRoot = path.join(import.meta.dirname, "..");
const boxesFile = path.join(webRoot, "src", "tutorial", "boxes.json");
const baseUrl = process.env.SITE_URL ?? "http://localhost:4173";
const exportFile = process.env.SNAP_EXPORT ?? path.join(webRoot, "e2e", "fixtures", "sample-export.zip");

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

async function rectOf(page: Page, selector: string): Promise<Rect> {
  const box = await page.locator(selector).boundingBox();
  if (!box) throw new Error(`${selector} is not visible`);
  const scroll = await page.evaluate(() => ({ x: window.scrollX, y: window.scrollY }));
  return { x: box.x + scroll.x, y: box.y + scroll.y, width: box.width, height: box.height };
}

function frameAround(panel: Rect): Rect {
  let width = panel.width + PADDING * 2;
  let height = panel.height + PADDING * 2;
  if (width / height < ASPECT) width = height * ASPECT;
  else height = width / ASPECT;
  return {
    x: Math.max(0, panel.x + panel.width / 2 - width / 2),
    y: Math.max(0, panel.y + panel.height / 2 - height / 2),
    width,
    height,
  };
}

function percentBox(frame: Rect, target: Rect): number[] {
  const round = (value: number): number => Math.round(value * 100) / 100;
  return [
    round(((target.x - frame.x) / frame.width) * 100),
    round(((target.y - frame.y) / frame.height) * 100),
    round(((target.x + target.width - frame.x) / frame.width) * 100),
    round(((target.y + target.height - frame.y) / frame.height) * 100),
  ];
}

async function capture(page: Page, file: string, highlight: string): Promise<number[]> {
  await page.locator("#app").scrollIntoViewIfNeeded();
  const frame = frameAround(await rectOf(page, "#app"));
  const box = percentBox(frame, await rectOf(page, highlight));
  await page.screenshot({ path: file, type: "jpeg", quality: 86, clip: frame, fullPage: true });
  return box;
}

const boxes = JSON.parse(readFileSync(boxesFile, "utf8")) as Record<string, Record<string, number[]>>;
const browser = await chromium.launch();
for (const [language, pagePath] of Object.entries(PAGES)) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: OUTPUT_WIDTH / 1000,
    reducedMotion: "reduce",
    bypassCSP: true,
  });
  const page = await context.newPage();
  await page.addInitScript(() => {
    Reflect.set(window, "showDirectoryPicker", () => navigator.storage.getDirectory());
  });
  await page.goto(`${baseUrl}${pagePath}`);
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: "#outil .section-head { visibility: hidden; }" });
  const directory = path.join(webRoot, "public", "tutorial", language);
  const measured = boxes[language] ?? {};

  measured.boxSiteDrop = await capture(page, path.join(directory, "site-drop.jpg"), "#dropzone");
  await page.setInputFiles("#file-input", exportFile);
  await page.locator("#screen-summary.active").waitFor();
  measured.boxSiteSummary = await capture(page, path.join(directory, "site-summary.jpg"), "#btn-folder");
  await page.locator("#btn-folder").click();
  await page.locator("#screen-done.active").waitFor({ timeout: 120_000 });
  measured.boxSiteDone = await capture(page, path.join(directory, "site-done.jpg"), "#screen-done .stats");

  boxes[language] = measured;
  await context.close();
}
await browser.close();
writeFileSync(boxesFile, `${JSON.stringify(boxes, null, 2)}\n`);
