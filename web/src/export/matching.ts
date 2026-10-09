import type { EmbeddedMemory, MemoryEntry } from "./models";

const OFFSET_STEP_MS = 15 * 60 * 1000;
const MAX_OFFSET_MS = 14 * 60 * 60 * 1000;
const TOLERANCE_MS = 2000;

export interface MatchResult {
  pairs: Array<[EmbeddedMemory, MemoryEntry | null]>;
  unmatchedEntries: MemoryEntry[];
  archiveOffsetMs: number;
}

export function matchEmbedded(entries: MemoryEntry[], embedded: EmbeddedMemory[]): MatchResult {
  const candidates = new Map<string, Array<[number, number]>>();
  const popularity = new Map<number, number>();
  for (const memory of embedded) {
    const options: Array<[number, number]> = [];
    entries.forEach((entry, index) => {
      if (entry.kind !== memory.kind) return;
      const offset = archiveOffset(memory.archivedAtMs, entry.takenAt.getTime());
      if (offset === null) return;
      options.push([index, offset]);
      popularity.set(offset, (popularity.get(offset) ?? 0) + 1);
    });
    candidates.set(memory.key, options);
  }

  const used = new Set<number>();
  const pairs: Array<[EmbeddedMemory, MemoryEntry | null]> = [];
  const ordered = [...embedded].sort((a, b) => a.archivedAtMs - b.archivedAtMs);
  for (const memory of ordered) {
    const options = (candidates.get(memory.key) ?? []).filter(([index]) => !used.has(index));
    const best = options.reduce<[number, number] | null>((winner, option) => {
      if (!winner) return option;
      const score = popularity.get(option[1]) ?? 0;
      const winnerScore = popularity.get(winner[1]) ?? 0;
      if (score !== winnerScore) return score > winnerScore ? option : winner;
      return Math.abs(option[1]) < Math.abs(winner[1]) ? option : winner;
    }, null);
    if (!best) {
      pairs.push([memory, null]);
      continue;
    }
    used.add(best[0]);
    pairs.push([memory, entries[best[0]] ?? null]);
  }

  let dominantOffset = 0;
  let dominantCount = 0;
  for (const [offset, count] of popularity) {
    if (count > dominantCount) {
      dominantOffset = offset;
      dominantCount = count;
    }
  }
  return {
    pairs,
    unmatchedEntries: entries.filter((_, index) => !used.has(index)),
    archiveOffsetMs: dominantOffset,
  };
}

function archiveOffset(archivedAtMs: number, takenAtMs: number): number | null {
  const delta = archivedAtMs - takenAtMs;
  const rounded = Math.round(delta / OFFSET_STEP_MS) * OFFSET_STEP_MS;
  if (Math.abs(delta - rounded) > TOLERANCE_MS || Math.abs(rounded) > MAX_OFFSET_MS) return null;
  return rounded;
}
