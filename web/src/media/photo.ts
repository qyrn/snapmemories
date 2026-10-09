import type { GeoPoint } from "../export/models";
import { type MediaFormat, sniffBlob } from "./format";
import { readOrientation, withExif } from "./jpegExif";
import type { LocalTime } from "./localTime";

const JPEG_QUALITY = 0.95;
const NORMAL_ORIENTATION = 1;

export interface RenderedPhoto {
  data: Blob;
  format: MediaFormat;
}

async function drawToJpeg(main: Blob, overlay: Blob | null): Promise<Blob> {
  const base = await createImageBitmap(main);
  const canvas = new OffscreenCanvas(base.width, base.height);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas unavailable");
  context.fillStyle = "#000";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(base, 0, 0);
  base.close();
  if (overlay) {
    const layer = await createImageBitmap(overlay);
    context.drawImage(layer, 0, 0, canvas.width, canvas.height);
    layer.close();
  }
  return canvas.convertToBlob({ type: "image/jpeg", quality: JPEG_QUALITY });
}

async function jpegBase(main: Blob, overlay: Blob | null, format: MediaFormat): Promise<[Blob, number]> {
  if (overlay) {
    try {
      return [await drawToJpeg(main, overlay), NORMAL_ORIENTATION];
    } catch {
      return jpegBase(main, null, format);
    }
  }
  if (format === "jpeg") return [main, await readOrientation(main).catch(() => NORMAL_ORIENTATION)];
  return [await drawToJpeg(main, null), NORMAL_ORIENTATION];
}

export async function renderPhoto(
  main: Blob,
  overlay: Blob | null,
  takenAt: LocalTime,
  location: GeoPoint | null,
): Promise<RenderedPhoto> {
  const format = await sniffBlob(main);
  let base: Blob;
  let orientation: number;
  try {
    [base, orientation] = await jpegBase(main, overlay, format);
  } catch {
    return { data: main, format };
  }
  try {
    return { data: await withExif(base, { takenAt, location, orientation }), format: "jpeg" };
  } catch {
    return { data: base, format: "jpeg" };
  }
}
