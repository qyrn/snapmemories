import type { GeoPoint } from "../export/models";
import { exifDate, formatOffset, type LocalTime } from "./localTime";

const SOI = 0xd8;
const APP0 = 0xe0;
const APP1 = 0xe1;
const SOS = 0xda;
const EOI = 0xd9;
const EXIF_HEADER = [0x45, 0x78, 0x69, 0x66, 0x00, 0x00];
const MAX_SEGMENT_HEAD_BYTES = 1024 * 1024;

const BYTE = 1;
const ASCII = 2;
const SHORT = 3;
const LONG = 4;
const RATIONAL = 5;

const TAG_ORIENTATION = 0x0112;
const TAG_DATE_TIME = 0x0132;
const TAG_EXIF_POINTER = 0x8769;
const TAG_GPS_POINTER = 0x8825;
const TAG_DATE_TIME_ORIGINAL = 0x9003;
const TAG_DATE_TIME_DIGITIZED = 0x9004;
const TAG_OFFSET_TIME = 0x9010;
const TAG_OFFSET_TIME_ORIGINAL = 0x9011;
const TAG_OFFSET_TIME_DIGITIZED = 0x9012;
const SECONDS_PRECISION = 10_000;

interface Field {
  tag: number;
  type: number;
  count: number;
  bytes: Uint8Array;
}

export interface CaptureMetadata {
  takenAt: LocalTime;
  location: GeoPoint | null;
  orientation: number;
}

export class NotJpegError extends Error {}

function asciiField(tag: number, value: string): Field {
  const bytes = new Uint8Array(value.length + 1);
  for (let index = 0; index < value.length; index += 1) bytes[index] = value.charCodeAt(index);
  return { tag, type: ASCII, count: bytes.length, bytes };
}

function shortField(tag: number, value: number): Field {
  const bytes = new Uint8Array(2);
  new DataView(bytes.buffer).setUint16(0, value);
  return { tag, type: SHORT, count: 1, bytes };
}

function longField(tag: number, value: number): Field {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setUint32(0, value);
  return { tag, type: LONG, count: 1, bytes };
}

function rationalField(tag: number, values: Array<[number, number]>): Field {
  const bytes = new Uint8Array(values.length * 8);
  const view = new DataView(bytes.buffer);
  values.forEach(([numerator, denominator], index) => {
    view.setUint32(index * 8, numerator);
    view.setUint32(index * 8 + 4, denominator);
  });
  return { tag, type: RATIONAL, count: values.length, bytes };
}

export function degreesToRationals(value: number): Array<[number, number]> {
  const absolute = Math.abs(value);
  const degrees = Math.floor(absolute);
  const minutesFloat = (absolute - degrees) * 60;
  const minutes = Math.floor(minutesFloat);
  const seconds = Math.round((minutesFloat - minutes) * 60 * SECONDS_PRECISION);
  return [
    [degrees, 1],
    [minutes, 1],
    [seconds, SECONDS_PRECISION],
  ];
}

function ifdLength(fields: Field[]): number {
  const dataBytes = fields.reduce(
    (total, field) => total + (field.bytes.length > 4 ? field.bytes.length + (field.bytes.length % 2) : 0),
    0,
  );
  return 2 + fields.length * 12 + 4 + dataBytes;
}

function bytesOf(view: DataView): Uint8Array {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

function writeIfd(target: DataView, offset: number, fields: Field[]): void {
  const sorted = [...fields].sort((a, b) => a.tag - b.tag);
  target.setUint16(offset, sorted.length);
  let dataOffset = offset + 2 + sorted.length * 12 + 4;
  sorted.forEach((field, index) => {
    const entry = offset + 2 + index * 12;
    target.setUint16(entry, field.tag);
    target.setUint16(entry + 2, field.type);
    target.setUint32(entry + 4, field.count);
    if (field.bytes.length <= 4) {
      bytesOf(target).set(field.bytes, entry + 8);
    } else {
      target.setUint32(entry + 8, dataOffset);
      bytesOf(target).set(field.bytes, dataOffset);
      dataOffset += field.bytes.length + (field.bytes.length % 2);
    }
  });
  target.setUint32(offset + 2 + sorted.length * 12, 0);
}

export function buildExifPayload(metadata: CaptureMetadata): Uint8Array<ArrayBuffer> {
  const stamp = exifDate(metadata.takenAt);
  const offset = formatOffset(metadata.takenAt);
  const exifFields = [
    asciiField(TAG_DATE_TIME_ORIGINAL, stamp),
    asciiField(TAG_DATE_TIME_DIGITIZED, stamp),
    asciiField(TAG_OFFSET_TIME, offset),
    asciiField(TAG_OFFSET_TIME_ORIGINAL, offset),
    asciiField(TAG_OFFSET_TIME_DIGITIZED, offset),
  ];
  const location = metadata.location;
  const gpsFields: Field[] = location
    ? [
        { tag: 0x0000, type: BYTE, count: 4, bytes: new Uint8Array([2, 3, 0, 0]) },
        asciiField(0x0001, location.latitude >= 0 ? "N" : "S"),
        rationalField(0x0002, degreesToRationals(location.latitude)),
        asciiField(0x0003, location.longitude >= 0 ? "E" : "W"),
        rationalField(0x0004, degreesToRationals(location.longitude)),
      ]
    : [];
  const mainFields = [
    shortField(TAG_ORIENTATION, metadata.orientation),
    asciiField(TAG_DATE_TIME, stamp),
    longField(TAG_EXIF_POINTER, 0),
    ...(location ? [longField(TAG_GPS_POINTER, 0)] : []),
  ];

  const mainOffset = 8;
  const exifOffset = mainOffset + ifdLength(mainFields);
  const gpsOffset = exifOffset + ifdLength(exifFields);
  const totalLength = gpsOffset + (location ? ifdLength(gpsFields) : 0);
  for (const field of mainFields) {
    if (field.tag === TAG_EXIF_POINTER) new DataView(field.bytes.buffer).setUint32(0, exifOffset);
    if (field.tag === TAG_GPS_POINTER) new DataView(field.bytes.buffer).setUint32(0, gpsOffset);
  }

  const tiff = new Uint8Array(totalLength);
  const view = new DataView(tiff.buffer);
  view.setUint16(0, 0x4d4d);
  view.setUint16(2, 42);
  view.setUint32(4, mainOffset);
  writeIfd(view, mainOffset, mainFields);
  writeIfd(view, exifOffset, exifFields);
  if (location) writeIfd(view, gpsOffset, gpsFields);

  const payload = new Uint8Array(EXIF_HEADER.length + tiff.length);
  payload.set(EXIF_HEADER, 0);
  payload.set(tiff, EXIF_HEADER.length);
  return payload;
}

interface Segment {
  marker: number;
  start: number;
  end: number;
}

function readSegments(head: Uint8Array<ArrayBuffer>): { segments: Segment[]; scanStart: number } {
  if (head[0] !== 0xff || head[1] !== SOI) throw new NotJpegError();
  const segments: Segment[] = [];
  const view = new DataView(head.buffer, head.byteOffset, head.byteLength);
  let position = 2;
  while (position + 4 <= head.length && head[position] === 0xff) {
    const marker = head[position + 1] ?? 0;
    if (marker === 0xff) {
      position += 1;
      continue;
    }
    if (marker === SOS || marker === EOI) break;
    const end = position + 2 + view.getUint16(position + 2);
    if (end > head.length) throw new NotJpegError();
    segments.push({ marker, start: position, end });
    position = end;
  }
  return { segments, scanStart: position };
}

function isExifSegment(head: Uint8Array, segment: Segment): boolean {
  return (
    segment.marker === APP1 && EXIF_HEADER.every((byte, index) => head[segment.start + 4 + index] === byte)
  );
}

export async function readOrientation(jpeg: Blob): Promise<number> {
  const head = new Uint8Array(await jpeg.slice(0, MAX_SEGMENT_HEAD_BYTES).arrayBuffer());
  const { segments } = readSegments(head);
  const exif = segments.find((segment) => isExifSegment(head, segment));
  if (!exif) return 1;
  const tiffStart = exif.start + 10;
  const view = new DataView(head.buffer, head.byteOffset, head.byteLength);
  const littleEndian = view.getUint16(tiffStart) === 0x4949;
  const ifdStart = tiffStart + view.getUint32(tiffStart + 4, littleEndian);
  if (ifdStart + 2 > exif.end) return 1;
  const count = view.getUint16(ifdStart, littleEndian);
  for (let index = 0; index < count; index += 1) {
    const entry = ifdStart + 2 + index * 12;
    if (entry + 12 > exif.end) break;
    if (view.getUint16(entry, littleEndian) === TAG_ORIENTATION) {
      const value = view.getUint16(entry + 8, littleEndian);
      return value >= 1 && value <= 8 ? value : 1;
    }
  }
  return 1;
}

export async function withExif(jpeg: Blob, metadata: CaptureMetadata): Promise<Blob> {
  const head = new Uint8Array(await jpeg.slice(0, MAX_SEGMENT_HEAD_BYTES).arrayBuffer());
  const { segments, scanStart } = readSegments(head);
  const payload = buildExifPayload(metadata);
  const segmentHeader = new Uint8Array(4);
  segmentHeader.set([0xff, APP1]);
  new DataView(segmentHeader.buffer).setUint16(2, payload.length + 2);

  const leading: Uint8Array<ArrayBuffer>[] = [];
  const kept: Uint8Array<ArrayBuffer>[] = [];
  for (const segment of segments) {
    if (isExifSegment(head, segment)) continue;
    const bytes = head.subarray(segment.start, segment.end);
    if (segment.marker === APP0 && kept.length === 0) leading.push(bytes);
    else kept.push(bytes);
  }
  return new Blob([head.subarray(0, 2), ...leading, segmentHeader, payload, ...kept, jpeg.slice(scanStart)], {
    type: "image/jpeg",
  });
}
