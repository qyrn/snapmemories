import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { chromium, type Page } from "@playwright/test";

import { type Language, STRINGS } from "../src/ui/strings.ts";

const webRoot = path.join(import.meta.dirname, "..");
const publicDirectory = path.join(webRoot, "public");
const desktopStatic = path.join(webRoot, "..", "snapmemories", "static");
const PAPER = "#f3eee3";

function sprite(): string {
  const page = readFileSync(path.join(webRoot, "src", "page.html"), "utf8");
  const match = /<svg class="sprite"[\s\S]*?<\/svg>\n/.exec(page);
  if (!match) throw new Error("Sprite not found in page.html");
  return match[0];
}

const LOGO = path.join(webRoot, "build", "logo.svg");
const FAVICON = path.join(publicDirectory, "favicon.svg");

async function renderIcon(
  page: Page,
  source: string,
  size: number,
  padding: number,
  background: string,
): Promise<Buffer> {
  const mark = readFileSync(source, "utf8");
  await page.setViewportSize({ width: 600, height: 600 });
  await page.setContent(
    `<body style="margin:0;background:transparent"><div style="width:${size}px;height:${size}px;display:grid;place-items:center;background:${background}">` +
      `<div style="width:${size - padding * 2}px;height:${size - padding * 2}px">${mark.replace("<svg ", '<svg width="100%" height="100%" ')}</div></div></body>`,
  );
  return page.screenshot({
    clip: { x: 0, y: 0, width: size, height: size },
    omitBackground: background === "transparent",
  });
}

function icoFile(images: Array<{ size: number; png: Buffer }>): Buffer {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(images.length, 4);
  let offset = 6 + images.length * 16;
  const entries = images.map(({ size, png }) => {
    const entry = Buffer.alloc(16);
    entry.writeUInt8(size >= 256 ? 0 : size, 0);
    entry.writeUInt8(size >= 256 ? 0 : size, 1);
    entry.writeUInt16LE(1, 4);
    entry.writeUInt16LE(32, 6);
    entry.writeUInt32LE(png.length, 8);
    entry.writeUInt32LE(offset, 12);
    offset += png.length;
    return entry;
  });
  return Buffer.concat([header, ...entries, ...images.map(({ png }) => png)]);
}

async function renderShareImage(page: Page, language: Language): Promise<void> {
  await page.setViewportSize({ width: 1200, height: 630 });
  await page.goto(pathToFileURL(path.join(webRoot, "build", "og.html")).href);
  await page.evaluate(
    ({ markup, title, subtitle }) => {
      const holder = document.getElementById("sprite");
      if (holder) holder.innerHTML = markup;
      const heading = document.getElementById("title");
      const line = document.getElementById("subtitle");
      if (heading) heading.textContent = title;
      if (line) line.textContent = subtitle;
    },
    { markup: sprite(), title: STRINGS[language].heroTitle, subtitle: STRINGS[language].eyebrow },
  );
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: path.join(publicDirectory, `og-${language}.png`) });
}

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 1 });

const icoSizes = [16, 32, 48];
const small = [];
for (const size of icoSizes)
  small.push({ size, png: await renderIcon(page, FAVICON, size, 0, "transparent") });
const large = { size: 256, png: await renderIcon(page, LOGO, 256, 0, "transparent") };
writeFileSync(path.join(publicDirectory, "favicon.ico"), icoFile(small));
writeFileSync(path.join(desktopStatic, "favicon.ico"), icoFile([...small, large]));
writeFileSync(path.join(desktopStatic, "favicon.png"), await renderIcon(page, LOGO, 128, 0, "transparent"));
writeFileSync(
  path.join(publicDirectory, "apple-touch-icon.png"),
  await renderIcon(page, LOGO, 180, 18, PAPER),
);
writeFileSync(path.join(publicDirectory, "icon-192.png"), await renderIcon(page, LOGO, 192, 16, PAPER));
writeFileSync(path.join(publicDirectory, "icon-512.png"), await renderIcon(page, LOGO, 512, 40, PAPER));
writeFileSync(
  path.join(publicDirectory, "icon-512-maskable.png"),
  await renderIcon(page, LOGO, 512, 104, PAPER),
);
for (const language of ["en", "fr"] as const) await renderShareImage(page, language);

await browser.close();
