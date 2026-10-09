import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { expect, type Page, test } from "@playwright/test";

const SAMPLE_EXPORT = path.join(import.meta.dirname, "fixtures", "sample-export.zip");
const EXPORT_PATH = process.env["SNAP_EXPORT"] ?? SAMPLE_EXPORT;
const DUMP_DIRECTORY = process.env["SNAP_DUMP"];

interface SavedFile {
  path: string;
  base64: string;
}

function collectCspViolations(page: Page): string[] {
  const violations: string[] = [];
  page.on("console", (message) => {
    if (message.text().includes("Content Security Policy")) violations.push(message.text());
  });
  return violations;
}

async function useOriginPrivateFolder(page: Page): Promise<void> {
  await page.addInitScript(() => {
    Reflect.set(window, "showDirectoryPicker", () => navigator.storage.getDirectory());
  });
}

async function readOriginPrivateFolder(page: Page): Promise<SavedFile[]> {
  return page.evaluate(async () => {
    const files: Array<{ path: string; base64: string }> = [];
    const walk = async (directory: FileSystemDirectoryHandle, prefix: string): Promise<void> => {
      for await (const [name, handle] of directory as unknown as AsyncIterable<[string, FileSystemHandle]>) {
        const relative = prefix ? `${prefix}/${name}` : name;
        if (handle.kind === "directory") {
          await walk(handle as FileSystemDirectoryHandle, relative);
        } else {
          const bytes = new Uint8Array(
            await (await (handle as FileSystemFileHandle).getFile()).arrayBuffer(),
          );
          let binary = "";
          for (let index = 0; index < bytes.length; index += 0x8000) {
            binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
          }
          files.push({ path: relative, base64: btoa(binary) });
        }
      }
    };
    await walk(await navigator.storage.getDirectory(), "");
    return files;
  });
}

async function importInto(page: Page, button: "#btn-folder" | "#btn-zip"): Promise<void> {
  await page.goto("/");
  await page.setInputFiles("#file-input", EXPORT_PATH);
  await page.locator(button).click();
  await expect(page.locator("#screen-done")).toBeVisible({ timeout: 110_000 });
}

test("saves into a folder, then skips everything on a second run", async ({ page }) => {
  const violations = collectCspViolations(page);
  await useOriginPrivateFolder(page);

  await importInto(page, "#btn-folder");
  const saved = Number(await page.locator("#done-saved").textContent());
  expect(saved).toBeGreaterThan(0);
  await expect(page.locator("#done-failed")).toHaveText("0");

  const files = await readOriginPrivateFolder(page);
  const catalog = Buffer.from(
    files.find((file) => file.path === ".snapmemories/catalog.jsonl")?.base64 ?? "",
    "base64",
  ).toString();
  expect(catalog.trim().split("\n")).toHaveLength(saved);
  const photo = files.find((file) => file.path.endsWith(".jpg"));
  expect(Buffer.from(photo?.base64 ?? "", "base64").includes("Exif\0\0")).toBe(true);

  if (DUMP_DIRECTORY) {
    for (const file of files) {
      const target = path.join(DUMP_DIRECTORY, file.path);
      await mkdir(path.dirname(target), { recursive: true });
      await writeFile(target, Buffer.from(file.base64, "base64"));
    }
  }

  await importInto(page, "#btn-folder");
  await expect(page.locator("#done-saved")).toHaveText("0");
  await expect(page.locator("#done-skipped")).toHaveText(String(saved));
  expect(violations).toEqual([]);
});

test("falls back to ZIP downloads when folders are not supported", async ({ page }) => {
  const violations = collectCspViolations(page);
  await page.addInitScript(() => {
    Reflect.set(window, "showDirectoryPicker", undefined);
  });
  const download = page.waitForEvent("download");

  await importInto(page, "#btn-zip");

  const file = await download;
  expect(file.suggestedFilename()).toBe("Snapchat-Memories-1.zip");
  await expect(page.locator("#parts-list li")).toHaveCount(1);
  expect(violations).toEqual([]);
});

test("rejects files that are not exports", async ({ page }) => {
  await page.goto("/");
  await page.setInputFiles("#file-input", {
    name: "notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("x"),
  });
  await expect(page.locator("#drop-error")).toContainText("notes.txt");
});
