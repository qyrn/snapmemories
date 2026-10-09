import type { GeoPoint } from "../export/models";

const MP4_EPOCH_OFFSET_SECONDS = 2_082_844_800;
const CONTAINER_BOXES = new Set(["moov", "trak", "mdia", "minf", "stbl", "edts", "udta"]);
const TIMED_BOXES = new Set(["mvhd", "tkhd", "mdhd"]);
const LOCATION_BOX = "©xyz";
const UNDETERMINED_LANGUAGE = 0x15c7;
const MAX_UINT32 = 0xffffffff;
const MAX_MOOV_BYTES = 64 * 1024 * 1024;

export class Mp4FormatError extends Error {}

interface Box {
  kind: string;
  start: number;
  headerSize: number;
  size: number;
}

function boxType(bytes: Uint8Array, offset: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + 4));
}

function parseHeader(bytes: Uint8Array, start: number, limit: number, base = 0): Box {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const local = start - base;
  let size = view.getUint32(local);
  const kind = boxType(bytes, local + 4);
  let headerSize = 8;
  if (size === 1) {
    size = Number(view.getBigUint64(local + 8));
    headerSize = 16;
  } else if (size === 0) {
    size = limit - start;
  }
  if (size < headerSize || start + size > limit) throw new Mp4FormatError(`Invalid ${kind} box`);
  return { kind, start, headerSize, size };
}

function* children(bytes: Uint8Array, start: number, end: number): Generator<Box> {
  let position = start;
  while (position + 8 <= end) {
    const box = parseHeader(bytes, position, end);
    yield box;
    position = box.start + box.size;
  }
}

function visit(bytes: Uint8Array, start: number, end: number, visitor: (box: Box) => void): void {
  for (const box of children(bytes, start, end)) {
    visitor(box);
    if (CONTAINER_BOXES.has(box.kind))
      visit(bytes, box.start + box.headerSize, box.start + box.size, visitor);
  }
}

async function topLevelBoxes(video: Blob): Promise<Box[]> {
  const boxes: Box[] = [];
  let position = 0;
  while (position + 8 <= video.size) {
    const header = new Uint8Array(await video.slice(position, position + 16).arrayBuffer());
    const box = parseHeader(header, position, video.size, position);
    boxes.push(box);
    position = box.start + box.size;
  }
  return boxes;
}

function setCreationTimes(moov: Uint8Array, mp4Seconds: number): void {
  const view = new DataView(moov.buffer, moov.byteOffset, moov.byteLength);
  visit(moov, 0, moov.length, (box) => {
    if (!TIMED_BOXES.has(box.kind)) return;
    const payload = box.start + box.headerSize;
    if (moov[payload] === 1) {
      view.setBigUint64(payload + 4, BigInt(mp4Seconds));
      view.setBigUint64(payload + 12, BigInt(mp4Seconds));
    } else if (mp4Seconds <= MAX_UINT32) {
      view.setUint32(payload + 4, mp4Seconds);
      view.setUint32(payload + 8, mp4Seconds);
    }
  });
}

function box(kind: string, payload: Uint8Array): Uint8Array<ArrayBuffer> {
  const result = new Uint8Array(payload.length + 8);
  const view = new DataView(result.buffer);
  view.setUint32(0, result.length);
  for (let index = 0; index < 4; index += 1) result[4 + index] = kind.charCodeAt(index);
  result.set(payload, 8);
  return result;
}

function concat(parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  const result = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

export function locationText(location: GeoPoint): string {
  const format = (value: number, width: number): string => {
    const sign = value >= 0 ? "+" : "-";
    return (
      sign +
      Math.abs(value)
        .toFixed(4)
        .padStart(width - 1, "0")
    );
  };
  return `${format(location.latitude, 8)}${format(location.longitude, 9)}/`;
}

function locationBox(location: GeoPoint): Uint8Array<ArrayBuffer> {
  const text = new TextEncoder().encode(locationText(location));
  const payload = new Uint8Array(4 + text.length);
  const view = new DataView(payload.buffer);
  view.setUint16(0, text.length);
  view.setUint16(2, UNDETERMINED_LANGUAGE);
  payload.set(text, 4);
  return box(LOCATION_BOX, payload);
}

function shiftChunkOffsets(moov: Uint8Array, moovOffset: number, growth: number): boolean {
  const view = new DataView(moov.buffer, moov.byteOffset, moov.byteLength);
  const tables: Box[] = [];
  visit(moov, 0, moov.length, (found) => {
    if (found.kind === "stco" || found.kind === "co64") tables.push(found);
  });
  for (const table of tables) {
    const payload = table.start + table.headerSize;
    const count = view.getUint32(payload + 4);
    const wide = table.kind === "co64";
    for (let index = 0; index < count; index += 1) {
      const position = payload + 8 + index * (wide ? 8 : 4);
      const offset = wide ? Number(view.getBigUint64(position)) : view.getUint32(position);
      if (offset < moovOffset) continue;
      const shifted = offset + growth;
      if (wide) view.setBigUint64(position, BigInt(shifted));
      else if (shifted > MAX_UINT32) return false;
      else view.setUint32(position, shifted);
    }
  }
  return true;
}

function withLocation(
  moov: Uint8Array<ArrayBuffer>,
  moovOffset: number,
  location: GeoPoint,
): Uint8Array<ArrayBuffer> | null {
  const root = parseHeader(moov, 0, moov.length);
  const rootPayload = root.start + root.headerSize;
  const userData = [...children(moov, rootPayload, moov.length)].find((child) => child.kind === "udta");
  let payload: Uint8Array;
  if (!userData) {
    payload = concat([moov.subarray(rootPayload), box("udta", locationBox(location))]);
  } else {
    const userDataEnd = userData.start + userData.size;
    const kept = [...children(moov, userData.start + userData.headerSize, userDataEnd)]
      .filter((child) => child.kind !== LOCATION_BOX)
      .map((child) => moov.subarray(child.start, child.start + child.size));
    payload = concat([
      moov.subarray(rootPayload, userData.start),
      box("udta", concat([...kept, locationBox(location)])),
      moov.subarray(userDataEnd),
    ]);
  }
  const rebuilt = box("moov", payload);
  return shiftChunkOffsets(rebuilt, moovOffset, rebuilt.length - moov.length) ? rebuilt : null;
}

export async function withCaptureMetadata(
  video: Blob,
  takenAt: Date,
  location: GeoPoint | null,
): Promise<Blob> {
  const moovBox = (await topLevelBoxes(video)).find((found) => found.kind === "moov");
  if (!moovBox) throw new Mp4FormatError("No moov box");
  if (moovBox.size > MAX_MOOV_BYTES) throw new Mp4FormatError("moov box too large");
  const moovEnd = moovBox.start + moovBox.size;
  const moov = new Uint8Array(await video.slice(moovBox.start, moovEnd).arrayBuffer());

  setCreationTimes(moov, Math.floor(takenAt.getTime() / 1000) + MP4_EPOCH_OFFSET_SECONDS);
  const updated = location ? (withLocation(moov, moovBox.start, location) ?? moov) : moov;
  return new Blob([video.slice(0, moovBox.start), updated, video.slice(moovEnd)], {
    type: video.type || "video/mp4",
  });
}
