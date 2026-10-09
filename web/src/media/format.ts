export type MediaFormat = "jpeg" | "png" | "webp" | "heic" | "mp4" | "unknown";

const HEIC_BRANDS = new Set(["heic", "heix", "mif1", "msf1", "hevc"]);

export const FORMAT_EXTENSIONS: Record<MediaFormat, string> = {
  jpeg: "jpg",
  png: "png",
  webp: "webp",
  heic: "heic",
  mp4: "mp4",
  unknown: "bin",
};

function ascii(bytes: Uint8Array, start: number, end: number): string {
  return String.fromCharCode(...bytes.subarray(start, end));
}

export function sniffFormat(header: Uint8Array): MediaFormat {
  if (header[0] === 0xff && header[1] === 0xd8 && header[2] === 0xff) return "jpeg";
  if (ascii(header, 1, 4) === "PNG") return "png";
  if (ascii(header, 0, 4) === "RIFF" && ascii(header, 8, 12) === "WEBP") return "webp";
  if (ascii(header, 4, 8) === "ftyp") return HEIC_BRANDS.has(ascii(header, 8, 12)) ? "heic" : "mp4";
  return "unknown";
}

export async function sniffBlob(blob: Blob): Promise<MediaFormat> {
  return sniffFormat(new Uint8Array(await blob.slice(0, 16).arrayBuffer()));
}
