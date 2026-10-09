import { matchEmbedded } from "./matching";
import type { EmbeddedMemory, ImportItem, MemoryEntry } from "./models";

export interface ImportPlan {
  items: ImportItem[];
  missing: number;
  linkOnly: number;
}

export function buildPlan(entries: MemoryEntry[], embedded: EmbeddedMemory[]): ImportPlan {
  const match = matchEmbedded(entries, embedded);
  const items = match.pairs.map(([memory, entry]) => embeddedItem(memory, entry, match.archiveOffsetMs));
  const linkOnly = match.unmatchedEntries.filter(
    (entry) => entry.mediaDownloadUrl || entry.downloadLink,
  ).length;
  items.sort((a, b) => a.takenAt.getTime() - b.takenAt.getTime());
  return { items, missing: match.unmatchedEntries.length - linkOnly, linkOnly };
}

function embeddedItem(
  memory: EmbeddedMemory,
  entry: MemoryEntry | null,
  archiveOffsetMs: number,
): ImportItem {
  return {
    id: memory.key,
    kind: memory.kind,
    takenAt: entry ? entry.takenAt : new Date(memory.archivedAtMs - archiveOffsetMs),
    location: entry ? entry.location : null,
    source: memory,
  };
}

export function planSize(plan: ImportPlan): number {
  return plan.items.reduce(
    (total, item) => total + item.source.main.size + (item.source.overlay?.size ?? 0),
    0,
  );
}
