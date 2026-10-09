export interface ImportSummary {
  files: string[];
  photos: number;
  videos: number;
  total: number;
  located: number;
  missing: number;
  already_saved: number;
  needs_network: boolean;
  estimated_bytes: number;
  free_bytes: number;
  enough_space: boolean;
  output_directory: string;
}

export type ImportPhase = "running" | "done" | "cancelled";

export interface ImportProgress {
  phase: ImportPhase;
  total: number;
  processed: number;
  saved: number;
  skipped: number;
  failed: number;
  expired: number;
  missing: number;
  bytes_per_second: number;
  elapsed_seconds: number;
  eta_seconds: number;
  errors: string[];
  output_directory: string;
}

export interface LibraryItem {
  id: string;
  kind: "photo" | "video";
  taken_at: string;
  latitude: number | null;
  longitude: number | null;
}

const HEARTBEAT_INTERVAL_MS = 20_000;

export class ApiError extends Error {}

async function readJson<T>(response: Response): Promise<T> {
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload !== null && typeof payload === "object" && "error" in payload
        ? String(payload.error)
        : "Something went wrong. Please try again.";
    throw new ApiError(message);
  }
  return payload as T;
}

export async function getJson<T>(path: string): Promise<T> {
  return readJson<T>(await fetch(path, { headers: { Accept: "application/json" } }));
}

export async function postJson<T>(path: string): Promise<T> {
  return readJson<T>(await fetch(path, { method: "POST" }));
}

export function uploadFile(file: File, onProgress: (ratio: number) => void): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("PUT", `/api/uploads?name=${encodeURIComponent(file.name)}`);
    request.responseType = "json";
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    request.addEventListener("load", () => {
      const payload: unknown = request.response;
      const body = payload !== null && typeof payload === "object" ? payload : {};
      if (request.status === 200 && "files" in body && Array.isArray(body.files)) {
        resolve(body.files.map(String));
      } else {
        reject(new ApiError("error" in body ? String(body.error) : "Upload failed."));
      }
    });
    request.addEventListener("error", () => reject(new ApiError("The app stopped responding.")));
    request.send(file);
  });
}

export function keepServerAlive(): void {
  const beat = (): void => {
    fetch("/api/heartbeat", { method: "POST" }).catch(() => undefined);
  };
  beat();
  window.setInterval(beat, HEARTBEAT_INTERVAL_MS);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") beat();
  });
}
