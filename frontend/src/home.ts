import {
  ApiError,
  getJson,
  type ImportProgress,
  type ImportSummary,
  keepServerAlive,
  postJson,
  uploadFile,
} from "./api.js";
import { byId, element } from "./dom.js";
import { formatBytes, formatCount, formatDuration, plural } from "./format.js";

type Screen = "import" | "progress" | "done";

const POLL_INTERVAL_MS = 1000;
const ACCEPTED_EXTENSIONS = [".zip", ".json"];

const dropzone = byId<HTMLLabelElement>("dropzone");
const fileInput = byId<HTMLInputElement>("file-input");
const fileList = byId<HTMLUListElement>("file-list");
const uploadError = byId("upload-error");
const startButton = byId<HTMLButtonElement>("btn-start");
const clearButton = byId<HTMLButtonElement>("btn-clear");
const cancelButton = byId<HTMLButtonElement>("btn-cancel");

let busy = false;
let pollTimer: number | undefined;

function showScreen(screen: Screen, step: number): void {
  const screens: Record<Screen, string> = {
    import: "screen-import",
    progress: "screen-progress",
    done: "screen-done",
  };
  for (const [name, id] of Object.entries(screens)) {
    byId(id).classList.toggle("active", name === screen);
  }
  document.querySelectorAll<HTMLElement>(".step-item").forEach((item) => {
    const position = Number(item.dataset["step"]);
    item.classList.toggle("done", position < step);
    item.classList.toggle("active", position === step);
  });
}

function showError(message: string): void {
  uploadError.textContent = message;
  uploadError.hidden = false;
}

function hideError(): void {
  uploadError.hidden = true;
  uploadError.textContent = "";
}

function setBusy(value: boolean, label = "Start"): void {
  busy = value;
  startButton.textContent = label;
  dropzone.classList.toggle("disabled", value);
  fileInput.disabled = value;
}

function addFileRow(file: File): HTMLSpanElement {
  const row = element("li", "file-row");
  row.append(element("span", "", file.name));
  const status = element("span", "file-row-status", formatBytes(file.size));
  row.append(status);
  fileList.append(row);
  fileList.hidden = false;
  return status;
}

function isAccepted(file: File): boolean {
  const name = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension));
}

async function handleFiles(files: File[]): Promise<void> {
  if (busy || files.length === 0) return;
  hideError();
  const rejected = files.find((file) => !isAccepted(file));
  if (rejected) {
    showError(`${rejected.name} is not a ZIP file from Snapchat.`);
    return;
  }

  showScreen("import", 2);
  startButton.disabled = true;
  byId("summary-section").hidden = true;
  setBusy(true, "Reading files");
  try {
    for (const file of files) {
      const status = addFileRow(file);
      await uploadFile(file, (ratio) => {
        status.textContent = `${Math.round(ratio * 100)}%`;
      });
      status.textContent = "Ready";
      status.classList.add("done");
    }
    setBusy(true, "Analyzing");
    showSummary(await postJson<ImportSummary>("/api/analyze"));
    startButton.disabled = false;
  } catch (error) {
    showError(error instanceof ApiError ? error.message : "Could not read the files.");
  } finally {
    setBusy(false);
    clearButton.hidden = fileList.childElementCount === 0;
  }
}

function showSummary(summary: ImportSummary): void {
  byId("summary-photos").textContent = formatCount(summary.photos);
  byId("summary-videos").textContent = formatCount(summary.videos);
  byId("summary-size").textContent = formatBytes(summary.estimated_bytes);

  const notes = byId<HTMLUListElement>("summary-notes");
  notes.replaceChildren();
  const addNote = (text: string, warning = false): void => {
    notes.append(element("li", warning ? "warning" : "", text));
  };
  if (summary.located > 0) addNote(`${plural(summary.located, "memory", "memories")} with a location.`);
  if (summary.already_saved > 0) {
    addNote(`${plural(summary.already_saved, "memory", "memories")} already saved will be skipped.`);
  }
  if (summary.needs_network) addNote("Some memories will be downloaded from Snapchat: stay online.");
  if (summary.missing > 0) {
    addNote(
      `${plural(summary.missing, "memory", "memories")} not found in these files. ` +
        "If Snapchat sent several ZIP files, drop them all.",
      true,
    );
  }
  notes.hidden = notes.childElementCount === 0;

  const warning = byId("disk-warning");
  warning.textContent = summary.enough_space
    ? ""
    : `Not enough disk space: ${formatBytes(summary.free_bytes)} free, about ${formatBytes(summary.estimated_bytes)} needed.`;
  warning.classList.toggle("visible", !summary.enough_space);
  byId("summary-section").hidden = false;
}

function renderErrors(listId: string, sectionId: string, errors: string[]): void {
  const list = byId(listId);
  list.replaceChildren(...errors.map((message) => element("div", "error-item", message)));
  byId(sectionId).classList.toggle("visible", errors.length > 0);
}

function renderProgress(progress: ImportProgress): void {
  const toProcess = Math.max(progress.total - progress.skipped, 1);
  const done = progress.processed - progress.skipped;
  const percent = Math.min(100, Math.floor((done / toProcess) * 100));
  byId("progress-bar").style.width = `${percent}%`;
  byId("progress-percent").textContent = `${percent}%`;
  byId("progress-label").textContent = `${formatCount(done)} / ${formatCount(toProcess)} memories`;
  byId("stat-saved").textContent = formatCount(progress.saved);
  byId("stat-failed").textContent = formatCount(progress.failed + progress.expired);
  byId("stat-speed").textContent =
    progress.bytes_per_second > 0 ? `${formatBytes(progress.bytes_per_second)}/s` : "-";
  byId("stat-eta").textContent = formatDuration(progress.eta_seconds);
  byId("progress-subtitle").textContent =
    progress.skipped > 0
      ? `${plural(progress.skipped, "memory", "memories")} already saved, skipped.`
      : "Keep this tab open until the end.";
  renderErrors("progress-error-list", "progress-errors", progress.errors.slice(-20));
  byId("progress-expired").classList.toggle("visible", progress.expired > 0);
}

function renderDone(progress: ImportProgress): void {
  const cancelled = progress.phase === "cancelled";
  const incomplete = progress.failed + progress.expired > 0;
  byId("done-title").textContent = cancelled
    ? "Import stopped"
    : incomplete
      ? "Almost everything saved"
      : "Memories saved!";
  byId("done-subtitle").textContent = cancelled
    ? "Drop the same files again later: saved memories are skipped."
    : "Everything is sorted by year and month on your computer.";
  byId("done-saved").textContent = formatCount(progress.saved);
  byId("done-skipped").textContent = formatCount(progress.skipped);
  byId("done-failed").textContent = formatCount(progress.failed + progress.expired);
  byId("done-path").textContent = progress.output_directory;
  renderErrors("done-error-list", "done-errors", progress.errors);
  byId("done-expired").classList.toggle("visible", progress.expired > 0);
  showScreen("done", 4);
}

async function poll(): Promise<void> {
  try {
    const progress = await getJson<ImportProgress | null>("/api/import");
    if (progress === null) return;
    renderProgress(progress);
    if (progress.phase !== "running") {
      window.clearInterval(pollTimer);
      pollTimer = undefined;
      renderDone(progress);
    }
  } catch {
    byId("progress-subtitle").textContent = "The app stopped responding. Restart SnapMemories.";
  }
}

function startPolling(): void {
  showScreen("progress", 3);
  cancelButton.disabled = false;
  cancelButton.textContent = "Stop";
  window.clearInterval(pollTimer);
  pollTimer = window.setInterval(() => void poll(), POLL_INTERVAL_MS);
  void poll();
}

async function resetAll(): Promise<void> {
  try {
    await postJson("/api/reset");
  } catch (error) {
    showError(error instanceof ApiError ? error.message : "Could not reset.");
    return;
  }
  fileInput.value = "";
  fileList.replaceChildren();
  fileList.hidden = true;
  clearButton.hidden = true;
  startButton.disabled = true;
  byId("summary-section").hidden = true;
  byId("disk-warning").classList.remove("visible");
  hideError();
  showScreen("import", 1);
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
fileInput.addEventListener("change", () => {
  void handleFiles(Array.from(fileInput.files ?? []));
  fileInput.value = "";
});

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  try {
    await postJson("/api/import");
    startPolling();
  } catch (error) {
    startButton.disabled = false;
    showError(error instanceof ApiError ? error.message : "Could not start the import.");
  }
});

clearButton.addEventListener("click", () => void resetAll());
byId("btn-restart").addEventListener("click", () => void resetAll());

cancelButton.addEventListener("click", async () => {
  cancelButton.disabled = true;
  cancelButton.textContent = "Stopping";
  await postJson("/api/import/cancel").catch(() => undefined);
});

byId("btn-open-folder").addEventListener("click", () => {
  void postJson("/api/open-folder").catch(() => undefined);
});

async function restoreState(): Promise<void> {
  showScreen("import", 1);
  const progress = await getJson<ImportProgress | null>("/api/import").catch(() => null);
  if (progress === null) return;
  if (progress.phase === "running") startPolling();
  else renderDone(progress);
}

keepServerAlive();
void restoreState();
