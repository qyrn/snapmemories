import { BlobReader, configure, type FileEntry, ZipReader } from "@zip.js/zip.js";

import { ExportFormatError, parseMemoriesHistory } from "./history";
import type { ArchiveMember, EmbeddedMemory, MediaKind, MemoryEntry } from "./models";

const HISTORY_FILE_NAME = "memories_history.json";
const MEMBER_PATTERN = /^(.+?)-(main|overlay)$/i;
const PHOTO_EXTENSIONS = new Set(["jpg", "jpeg", "png", "webp", "heic"]);
const VIDEO_EXTENSIONS = new Set(["mp4", "mov"]);

configure({ useWebWorkers: false });

export interface OpenedExport {
  entries: MemoryEntry[];
  embedded: EmbeddedMemory[];
  close(): Promise<void>;
}

interface MemberGroup {
  main?: FileEntry;
  overlay?: FileEntry;
}

export async function openExport(files: File[]): Promise<OpenedExport> {
  const readers: ZipReader<Blob>[] = [];
  let entries: MemoryEntry[] | null = null;
  const embeddedByKey = new Map<string, EmbeddedMemory>();
  const close = async (): Promise<void> => {
    await Promise.all(readers.map((reader) => reader.close()));
  };

  try {
    for (const file of files) {
      if (file.name.toLowerCase().endsWith(".json")) {
        entries = parseMemoriesHistory(await file.text());
        continue;
      }
      const reader = new ZipReader(new BlobReader(file));
      readers.push(reader);
      let members: FileEntry[];
      try {
        members = (await reader.getEntries()).filter((entry): entry is FileEntry => !entry.directory);
      } catch {
        throw new ExportFormatError("invalidZip", file.name);
      }
      const history = members.find((entry) => baseName(entry.filename) === HISTORY_FILE_NAME);
      if (history && entries === null) {
        entries = parseMemoriesHistory(new TextDecoder().decode(await history.arrayBuffer()));
      }
      for (const memory of listEmbeddedMemories(members)) {
        embeddedByKey.set(memory.key, memory);
      }
    }
  } catch (error) {
    await close();
    throw error;
  }

  if (entries === null && embeddedByKey.size === 0) {
    await close();
    throw new ExportFormatError("noMemories");
  }
  return { entries: entries ?? [], embedded: [...embeddedByKey.values()], close };
}

export function listEmbeddedMemories(members: FileEntry[]): EmbeddedMemory[] {
  const groups = new Map<string, MemberGroup>();
  for (const entry of members) {
    const { stem, extension } = splitName(baseName(entry.filename));
    if (!PHOTO_EXTENSIONS.has(extension) && !VIDEO_EXTENSIONS.has(extension)) continue;
    const match = MEMBER_PATTERN.exec(stem);
    const key = match?.[1] ?? stem;
    const group = groups.get(key) ?? {};
    if (match?.[2]?.toLowerCase() === "overlay") group.overlay = entry;
    else group.main = entry;
    groups.set(key, group);
  }

  const memories: EmbeddedMemory[] = [];
  for (const [key, group] of groups) {
    if (!group.main) continue;
    const kind: MediaKind = VIDEO_EXTENSIONS.has(splitName(group.main.filename).extension)
      ? "video"
      : "photo";
    memories.push({
      key,
      kind,
      main: member(group.main),
      overlay: group.overlay ? member(group.overlay) : null,
      archivedAtMs: dosTimeToNaiveMs(Number(group.main.rawLastModDate)),
    });
  }
  return memories;
}

export function dosTimeToNaiveMs(raw: number): number {
  const date = raw >>> 16;
  const time = raw & 0xffff;
  return Date.UTC(
    (date >> 9) + 1980,
    ((date >> 5) & 0x0f) - 1,
    date & 0x1f,
    time >> 11,
    (time >> 5) & 0x3f,
    (time & 0x1f) * 2,
  );
}

function member(entry: FileEntry): ArchiveMember {
  return { entry, name: entry.filename, size: entry.uncompressedSize };
}

function baseName(path: string): string {
  return path.split("/").pop() ?? path;
}

function splitName(name: string): { stem: string; extension: string } {
  const base = baseName(name);
  const dot = base.lastIndexOf(".");
  if (dot <= 0) return { stem: base, extension: "" };
  return { stem: base.slice(0, dot), extension: base.slice(dot + 1).toLowerCase() };
}
