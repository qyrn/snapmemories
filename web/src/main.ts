import "./styles.css";

import { type OpenedExport, openExport } from "./export/archive";
import { type ExportErrorCode, ExportFormatError } from "./export/history";
import { buildPlan, type ImportPlan, planSize } from "./export/plan";
import { ImportJob, type JobProgress } from "./import/job";
import { FolderTarget } from "./output/folderTarget";
import type { OutputTarget, SavedPart } from "./output/target";
import { ZipTarget } from "./output/zipTarget";
import { byId, element, showScreen } from "./ui/dom";
import { canWriteFolders, pickFolder } from "./ui/fileSystemAccess";
import { formatBytes, formatCount, formatDate, formatDuration } from "./ui/format";
import { isMessageKey, language, type MessageKey, setLanguage, t } from "./ui/i18n";

const ACCEPTED_EXTENSIONS = [".zip", ".json"];
const ZIP_BASE_NAME = "Snapchat-Memories";
const ERROR_MESSAGES: Record<ExportErrorCode, MessageKey> = {
  invalidZip: "errorInvalidZip",
  noMemories: "errorNoMemories",
  invalidHistory: "errorInvalidHistory",
};

const dropzone = byId<HTMLLabelElement>("dropzone");
const fileInput = byId<HTMLInputElement>("file-input");
const dropError = byId("drop-error");
const summaryError = byId("summary-error");
const stopButton = byId<HTMLButtonElement>("btn-stop");

let files: File[] = [];
let opened: OpenedExport | null = null;
let plan: ImportPlan | null = null;
let job: ImportJob | null = null;
let lastProgress: JobProgress | null = null;
let partUrls: string[] = [];
let pendingFrame = 0;

function translatePage(): void {
  document.documentElement.lang = language();
  document.title = t("pageTitle");
  document.querySelector('meta[name="description"]')?.setAttribute("content", t("pageDescription"));
  document.querySelectorAll<HTMLElement>("[data-i18n]").forEach((node) => {
    const key = node.dataset["i18n"] ?? "";
    if (isMessageKey(key)) node.textContent = t(key);
  });
}

function showError(target: HTMLElement, message: string): void {
  target.textContent = message;
  target.hidden = false;
}

function errorMessage(error: unknown): string {
  if (error instanceof ExportFormatError) return t(ERROR_MESSAGES[error.code], { name: error.fileName });
  return t("errorUnexpected");
}

function isAccepted(file: File): boolean {
  const name = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension));
}

async function handleFiles(selected: File[]): Promise<void> {
  if (selected.length === 0 || dropzone.classList.contains("busy")) return;
  dropError.hidden = true;
  const rejected = selected.find((file) => !isAccepted(file));
  if (rejected) {
    showError(dropError, t("errorInvalidZip", { name: rejected.name }));
    return;
  }

  dropzone.classList.add("busy");
  byId("reading").hidden = false;
  try {
    await opened?.close();
    files = selected;
    opened = await openExport(selected);
    plan = buildPlan(opened.entries, opened.embedded);
    if (plan.items.length === 0) {
      showError(
        dropError,
        plan.linkOnly > 0
          ? t("noteLinkOnly", { count: formatCount(plan.linkOnly) })
          : t("errorNothingToImport"),
      );
      return;
    }
    renderSummary();
    showScreen("screen-summary");
  } catch (error) {
    showError(dropError, errorMessage(error));
  } finally {
    dropzone.classList.remove("busy");
    byId("reading").hidden = true;
  }
}

function renderSummary(): void {
  if (!plan) return;
  const fileList = byId("file-list");
  fileList.replaceChildren(
    ...files.map((file) => {
      const row = element("li");
      row.append(element("span", "", file.name), element("span", "", formatBytes(file.size)));
      return row;
    }),
  );
  const photos = plan.items.filter((item) => item.kind === "photo").length;
  byId("summary-photos").textContent = formatCount(photos);
  byId("summary-videos").textContent = formatCount(plan.items.length - photos);
  byId("summary-size").textContent = formatBytes(planSize(plan));

  const located = plan.items.filter((item) => item.location !== null).length;
  const notes: Array<[string, boolean]> = [];
  if (located > 0) notes.push([t("noteLocated", { count: formatCount(located) }), false]);
  if (plan.missing > 0) notes.push([t("noteMissing", { count: formatCount(plan.missing) }), true]);
  if (plan.linkOnly > 0) notes.push([t("noteLinkOnly", { count: formatCount(plan.linkOnly) }), true]);
  byId("summary-notes").replaceChildren(
    ...notes.map(([text, warning]) => element("li", warning ? "warning" : "", text)),
  );

  const folders = canWriteFolders();
  byId("btn-folder").hidden = !folders;
  byId("folder-hint").hidden = !folders;
  byId("btn-zip").hidden = folders;
  byId("zip-hint").hidden = folders;
}

function downloadPart(part: SavedPart): void {
  partUrls.push(part.url);
  const link = element("a");
  link.href = part.url;
  link.download = part.name;
  link.click();
}

async function start(target: OutputTarget): Promise<void> {
  if (!plan) return;
  summaryError.hidden = true;
  stopButton.disabled = false;
  stopButton.textContent = t("stop");
  showScreen("screen-progress");
  window.addEventListener("beforeunload", warnBeforeLeaving);
  job = new ImportJob(plan, target, (progress) => {
    lastProgress = progress;
    scheduleProgressRender();
  });
  try {
    renderDone(await job.run());
  } catch {
    showScreen("screen-summary");
    showError(summaryError, t("errorUnexpected"));
  } finally {
    window.removeEventListener("beforeunload", warnBeforeLeaving);
    job = null;
  }
}

function warnBeforeLeaving(event: BeforeUnloadEvent): void {
  event.preventDefault();
  event.returnValue = t("leaveWarning");
}

function scheduleProgressRender(): void {
  if (pendingFrame) return;
  pendingFrame = requestAnimationFrame(() => {
    pendingFrame = 0;
    if (lastProgress?.phase === "running") renderProgress(lastProgress);
  });
}

function renderProgress(progress: JobProgress): void {
  const toProcess = Math.max(progress.total - progress.skipped, 1);
  const done = progress.processed - progress.skipped;
  const percent = Math.min(100, Math.floor((done / toProcess) * 100));
  const elapsedSeconds = (performance.now() - progress.startedAt) / 1000;
  const remaining = done > 0 ? (elapsedSeconds / done) * (toProcess - done) : 0;
  byId("progress-bar").style.width = `${percent}%`;
  byId("progress-percent").textContent = `${percent}%`;
  byId("progress-label").textContent = t("progressLabel", {
    done: formatCount(done),
    total: formatCount(toProcess),
  });
  byId("progress-saved").textContent = formatCount(progress.saved);
  byId("progress-failed").textContent = formatCount(progress.failed);
  byId("progress-eta").textContent = formatDuration(remaining);
  byId("progress-text").textContent =
    progress.skipped > 0 ? t("skippedLabel", { count: formatCount(progress.skipped) }) : t("savingText");
}

function renderDone(progress: JobProgress): void {
  lastProgress = progress;
  const cancelled = progress.phase === "cancelled";
  byId("done-title").textContent = t(
    cancelled ? "stoppedTitle" : progress.failed > 0 ? "doneIncompleteTitle" : "doneTitle",
  );
  byId("done-text").textContent = t(cancelled ? "stoppedText" : "doneText");
  byId("done-saved").textContent = formatCount(progress.saved);
  byId("done-skipped").textContent = formatCount(progress.skipped);
  byId("done-failed").textContent = formatCount(progress.failed);

  byId("parts").hidden = progress.parts.length === 0;
  byId("parts-list").replaceChildren(
    ...progress.parts.map((part) => {
      const link = element("a");
      link.href = part.url;
      link.download = part.name;
      link.append(element("span", "", part.name), element("span", "", formatBytes(part.size)));
      const row = element("li");
      row.append(link);
      return row;
    }),
  );

  byId("done-errors").hidden = progress.errors.length === 0;
  byId("done-error-list").replaceChildren(
    ...progress.errors.map(({ item, message }) =>
      element("li", "", `${formatDate(item.takenAt)}: ${message}`),
    ),
  );
  showScreen("screen-done");
}

async function resetAll(): Promise<void> {
  await opened?.close();
  opened = null;
  plan = null;
  files = [];
  lastProgress = null;
  for (const url of partUrls) URL.revokeObjectURL(url);
  partUrls = [];
  fileInput.value = "";
  dropError.hidden = true;
  summaryError.hidden = true;
  showScreen("screen-drop");
}

function rerender(): void {
  translatePage();
  if (plan) renderSummary();
  if (lastProgress?.phase === "running") renderProgress(lastProgress);
  else if (lastProgress) renderDone(lastProgress);
}

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  void handleFiles(Array.from(event.dataTransfer?.files ?? []));
});
fileInput.addEventListener("change", () => void handleFiles(Array.from(fileInput.files ?? [])));

byId("btn-folder").addEventListener("click", async () => {
  try {
    const folder = await pickFolder();
    if (folder) await start(new FolderTarget(folder));
  } catch {
    showError(summaryError, t("errorFolder"));
  }
});
byId("btn-zip").addEventListener("click", () => void start(new ZipTarget(ZIP_BASE_NAME, downloadPart)));
byId("btn-reset").addEventListener("click", () => void resetAll());
byId("btn-again").addEventListener("click", () => void resetAll());
stopButton.addEventListener("click", () => {
  job?.cancel();
  stopButton.disabled = true;
  stopButton.textContent = t("stopping");
});
byId("language-toggle").addEventListener("click", () => {
  setLanguage(language() === "fr" ? "en" : "fr");
  rerender();
});

translatePage();
