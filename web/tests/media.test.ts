import { describe, expect, it } from "vitest";

import { buildExifPayload, readOrientation, withExif } from "../src/media/jpegExif";
import { fileStem, type LocalTime, monthFolder } from "../src/media/localTime";
import { locationText, withCaptureMetadata } from "../src/media/mp4Metadata";
import { minimalJpeg, mp4Bytes } from "./fixtures";

const LOCAL: LocalTime = {
  year: 2025,
  month: 6,
  day: 1,
  hours: 2,
  minutes: 11,
  seconds: 34,
  offsetMinutes: 120,
};
const TAKEN = new Date("2026-02-14T16:28:15Z");
const MP4_SECONDS = Math.floor(TAKEN.getTime() / 1000) + 2_082_844_800;
const MEDIA = new Uint8Array(64).fill(7);

function latin(bytes: Uint8Array): string {
  return String.fromCharCode(...bytes);
}

function uint32At(bytes: Uint8Array, offset: number): number {
  return new DataView(bytes.buffer, bytes.byteOffset).getUint32(offset);
}

describe("jpeg exif", () => {
  it("writes dates, offsets and GPS in a single EXIF block", async () => {
    const jpeg = new Blob([minimalJpeg()]);
    const result = new Uint8Array(
      await (
        await withExif(jpeg, {
          takenAt: LOCAL,
          location: { latitude: 48.89948, longitude: -6.049386 },
          orientation: 6,
        })
      ).arrayBuffer(),
    );
    const text = latin(result);
    expect(text.split("Exif\0\0")).toHaveLength(2);
    expect(text).toContain("2025:06:01 02:11:34");
    expect(text).toContain("+02:00");
    expect(text).toContain("W\0");
    expect(latin(result.subarray(result.length - 10))).toBe(latin(minimalJpeg().subarray(-10)));
    expect(await readOrientation(new Blob([result]))).toBe(6);
  });

  it("omits the GPS directory without a location", () => {
    const payload = latin(buildExifPayload({ takenAt: LOCAL, location: null, orientation: 1 }));
    expect(payload).not.toContain("N\0");
  });

  it("names files and folders from local time", () => {
    expect(fileStem(LOCAL)).toBe("2025-06-01_02-11-34");
    expect(monthFolder(LOCAL)).toBe("2025/2025-06");
  });
});

describe("mp4 metadata", () => {
  it.each([true, false])(
    "sets times and location keeping media reachable (moov first: %s)",
    async (moovFirst) => {
      const output = new Uint8Array(
        await (
          await withCaptureMetadata(new Blob([mp4Bytes(moovFirst, MEDIA)]), TAKEN, {
            latitude: 45.44657,
            longitude: 4.389548,
          })
        ).arrayBuffer(),
      );
      const text = latin(output);
      for (const kind of ["mvhd", "tkhd", "mdhd"]) {
        expect(uint32At(output, text.indexOf(kind) + 8)).toBe(MP4_SECONDS);
      }
      expect(text).toContain("+45.4466+004.3895/");
      const chunk = uint32At(output, text.indexOf("stco") + 12);
      expect([...output.subarray(chunk, chunk + MEDIA.length)]).toEqual([...MEDIA]);
    },
  );

  it("keeps the file size without a location", async () => {
    const original = mp4Bytes(true, MEDIA);
    const output = await withCaptureMetadata(new Blob([original]), TAKEN, null);
    expect(output.size).toBe(original.length);
  });

  it("formats ISO 6709 coordinates", () => {
    expect(locationText({ latitude: -33.8688, longitude: 151.2093 })).toBe("-33.8688+151.2093/");
  });
});
