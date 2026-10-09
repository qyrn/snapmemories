import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";

const FRAME_RATE = 30;
const DURATION_SECONDS = 45;
const WIDTH = 1920;
const HEIGHT = 1080;
const PAGES = { en: "/", fr: "/fr/" } as const;

const ffmpeg = process.env.FFMPEG;
const baseUrl = process.env.SITE_URL ?? "http://localhost:4173";
if (!ffmpeg) throw new Error("Set FFMPEG to the path of an ffmpeg binary.");

const outputDirectory = path.join(import.meta.dirname, "video");
mkdirSync(outputDirectory, { recursive: true });

const CAPTURE_STYLE = `
  body > *:not(main), main > section:not(#tuto), #tuto .section-head, .controls, .chapters { display: none !important; }
  body { background: #f3eee3; overflow: hidden; }
  main > section#tuto { padding: 0; border: none; max-width: none; }
  .player { width: ${WIDTH}px; height: ${HEIGHT}px; max-width: none; padding: 44px 0 0; display: flex; flex-direction: column; align-items: center; }
  .stage { width: 1376px; flex: none; box-shadow: 10px 10px 0 #1b1a17; }
  .caption { width: 1376px; max-width: none; margin-top: 30px; font-size: 34px; text-align: center; }
`;

function encode(file: string): { write: (frame: Buffer) => Promise<void>; finish: () => Promise<void> } {
  const encoder = spawn(
    ffmpeg as string,
    [
      "-y", "-loglevel", "error",
      "-f", "image2pipe", "-framerate", String(FRAME_RATE), "-i", "-",
      "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
      file,
    ],
    { stdio: ["pipe", "inherit", "inherit"] },
  );
  const finished = new Promise<void>((resolve, reject) => {
    encoder.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited with ${code}`))));
  });
  return {
    write: (frame) =>
      new Promise((resolve) => {
        if (encoder.stdin.write(frame)) resolve();
        else encoder.stdin.once("drain", () => resolve());
      }),
    finish: () => {
      encoder.stdin.end();
      return finished;
    },
  };
}

const browser = await chromium.launch();
for (const [language, pagePath] of Object.entries(PAGES)) {
  const context = await browser.newContext({
    viewport: { width: WIDTH, height: HEIGHT },
    bypassCSP: true,
    reducedMotion: "no-preference",
  });
  const page = await context.newPage();
  await page.goto(`${baseUrl}${pagePath}`);
  await page.addStyleTag({ content: CAPTURE_STYLE });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(800);
  await page.evaluate(() => {
    const player = document.getElementById("player");
    const button = document.getElementById("play-button");
    if (player?.classList.contains("playing")) button?.click();
  });

  const output = encode(path.join(outputDirectory, `snapmemories-tutorial-${language}.mp4`));
  const frames = DURATION_SECONDS * FRAME_RATE;
  for (let frame = 0; frame < frames; frame += 1) {
    await page.evaluate((seconds) => {
      const scrubber = document.getElementById("scrubber");
      if (!(scrubber instanceof HTMLInputElement)) return;
      scrubber.step = "any";
      scrubber.value = String(seconds);
      scrubber.dispatchEvent(new Event("input"));
    }, frame / FRAME_RATE);
    await output.write(await page.screenshot({ type: "jpeg", quality: 95 }));
  }
  await output.finish();
  await context.close();
  console.log(`${language}: ${frames} frames`);
}
await browser.close();
