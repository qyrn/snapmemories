import { describe, expect, it } from "vitest";

import { dosTimeToNaiveMs } from "../src/export/archive";
import { parseLocation, parseMemoriesHistory } from "../src/export/history";
import { matchEmbedded } from "../src/export/matching";
import type { EmbeddedMemory, MediaKind, MemoryEntry } from "../src/export/models";
import { buildPlan } from "../src/export/plan";

function entry(iso: string, kind: MediaKind = "photo", link = ""): MemoryEntry {
  return {
    takenAt: new Date(iso),
    kind,
    location: { latitude: 1, longitude: 1 },
    mediaDownloadUrl: link,
    downloadLink: "",
  };
}

function embedded(key: string, naiveIso: string, kind: MediaKind = "photo"): EmbeddedMemory {
  const member = { entry: {} as never, name: `memories/${key}-main.jpg`, size: 10 };
  return { key, kind, main: member, overlay: null, archivedAtMs: Date.parse(`${naiveIso}Z`) };
}

describe("memories history", () => {
  it("reads the current Snapchat format", () => {
    const [memory] = parseMemoriesHistory(
      JSON.stringify({
        "Saved Media": [
          {
            Date: "2026-02-14 16:28:15 UTC",
            "Media Type": "Video",
            Location: "Latitude, Longitude: 45.44657, 4.389548",
            "Download Link": "",
            "Media Download Url": "",
          },
        ],
      }),
    );
    expect(memory?.takenAt.toISOString()).toBe("2026-02-14T16:28:15.000Z");
    expect(memory?.kind).toBe("video");
    expect(memory?.location).toEqual({ latitude: 45.44657, longitude: 4.389548 });
  });

  it.each([
    ["Latitude, Longitude: 0.0, 0.0", null],
    ["Latitude, Longitude: -33.86, 151.2", { latitude: -33.86, longitude: 151.2 }],
    ["Latitude, Longitude: 95, 10", null],
    [
      { Latitude: "48.85", Longitude: 2.35 },
      { latitude: 48.85, longitude: 2.35 },
    ],
    [null, null],
  ])("parses location %j", (value, expected) => {
    expect(parseLocation(value)).toEqual(expected);
  });

  it("decodes MS-DOS timestamps", () => {
    expect(new Date(dosTimeToNaiveMs(1523602800)).toISOString()).toBe("2025-06-16T10:43:32.000Z");
  });
});

describe("matching", () => {
  it("tolerates the two second precision of zip timestamps", () => {
    const memory = embedded("a", "2025-06-16T10:43:32");
    const result = matchEmbedded([entry("2025-06-16T10:43:33Z")], [memory]);
    expect(result.pairs[0]?.[1]?.takenAt.toISOString()).toBe("2025-06-16T10:43:33.000Z");
  });

  it("prefers the dominant offset when archive times are local", () => {
    const result = matchEmbedded(
      [entry("2025-01-01T10:00:00Z"), entry("2025-01-01T10:15:00Z"), entry("2025-01-02T08:00:00Z")],
      [
        embedded("first", "2025-01-01T11:00:00"),
        embedded("second", "2025-01-01T11:15:00"),
        embedded("third", "2025-01-02T09:00:00"),
      ],
    );
    const matched = Object.fromEntries(
      result.pairs.map(([memory, found]) => [memory.key, found?.takenAt.toISOString()]),
    );
    expect(matched["first"]).toBe("2025-01-01T10:00:00.000Z");
    expect(matched["second"]).toBe("2025-01-01T10:15:00.000Z");
    expect(result.archiveOffsetMs).toBe(3600_000);
  });

  it("counts link-only and missing memories separately", () => {
    const plan = buildPlan(
      [
        entry("2025-01-01T10:00:00Z"),
        entry("2025-02-01T10:00:00Z", "photo", "https://example.test"),
        entry("2025-03-01T10:00:00Z", "video"),
      ],
      [embedded("kept", "2025-01-01T10:00:00")],
    );
    expect(plan.items.map((item) => item.id)).toEqual(["kept"]);
    expect(plan.linkOnly).toBe(1);
    expect(plan.missing).toBe(1);
  });
});
