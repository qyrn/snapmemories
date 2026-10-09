import { spawn } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { chromium } from "@playwright/test";

import type { SoundEvent } from "../film/engine/timeline.ts";
import { mix, wavFile } from "./audio.ts";

const FRAME_RATE = Number(process.env.FRAME_RATE ?? 60);
const POSTER_SECONDS = 1.6;
const LANGUAGES = (process.env.LANGUAGES ?? "fr,en").split(",");

const ffmpeg = process.env.FFMPEG;
const filmUrl = process.env.FILM_URL ?? "http://localhost:4174";
if (!ffmpeg) throw new Error("Set FFMPEG to the path of an ffmpeg binary.");

const outputDirectory = path.join(import.meta.dirname, "..", "public", "video");
mkdirSync(outputDirectory, { recursive: true });

function run(args: string[], input?: (stdin: NodeJS.WritableStream) => Promise<void>): Promise<void> {
  const child = spawn(ffmpeg as string, ["-y", "-loglevel", "error", ...args], {
    stdio: [input ? "pipe" : "ignore", "inherit", "inherit"],
  });
  const done = new Promise<void>((resolve, reject) => {
    child.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited with ${code}`))));
  });
  if (!input || !child.stdin) return done;
  const stdin = child.stdin;
  return input(stdin).then(() => {
    stdin.end();
    return done;
  });
}

function write(stream: NodeJS.WritableStream, chunk: Buffer): Promise<void> {
  return new Promise((resolve) => {
    if (stream.write(chunk)) resolve();
    else stream.once("drain", () => resolve());
  });
}

const browser = await chromium.launch();
for (const language of LANGUAGES) {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto(`${filmUrl}/?lang=${language}`);
  await page.waitForFunction(() => window.film !== undefined, null, { timeout: 60_000 });
  const { duration, sounds } = await page.evaluate(() => ({
    duration: window.film?.duration ?? 0,
    sounds: window.film?.sounds ?? [],
  }));

  const work = path.join(tmpdir(), `snapmemories-film-${language}`);
  mkdirSync(work, { recursive: true });
  const silentVideo = path.join(work, "video.mp4");
  const audio = path.join(work, "audio.wav");
  writeFileSync(audio, wavFile(mix(duration, sounds as SoundEvent[])));

  const frames = Math.round(duration * FRAME_RATE);
  await run(
    [
      "-f",
      "image2pipe",
      "-framerate",
      String(FRAME_RATE),
      "-i",
      "-",
      "-vf",
      "scale=in_range=full:out_range=tv,format=yuv420p",
      "-c:v",
      "libx264",
      "-preset",
      "slow",
      "-crf",
      "22",
      "-color_range",
      "tv",
      silentVideo,
    ],
    async (stdin) => {
      for (let frame = 0; frame < frames; frame += 1) {
        await page.evaluate((seconds) => window.film?.seek(seconds), frame / FRAME_RATE);
        await write(stdin, await page.screenshot({ type: "jpeg", quality: 92 }));
      }
    },
  );

  const target = path.join(outputDirectory, `tutorial-${language}.mp4`);
  await run([
    "-i",
    silentVideo,
    "-i",
    audio,
    "-c:v",
    "copy",
    "-af",
    "loudnorm=I=-18:TP=-1.5:LRA=11",
    "-c:a",
    "aac",
    "-b:a",
    "160k",
    "-ar",
    "48000",
    "-movflags",
    "+faststart",
    "-shortest",
    target,
  ]);

  await page.evaluate((seconds) => window.film?.seek(seconds), POSTER_SECONDS);
  await page.screenshot({
    path: path.join(outputDirectory, `tutorial-${language}.jpg`),
    type: "jpeg",
    quality: 82,
  });
  await page.close();
  rmSync(work, { recursive: true, force: true });
  console.log(`${language}: ${frames} frames, ${sounds.length} sounds`);
}
await browser.close();
