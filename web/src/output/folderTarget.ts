import {
  CATALOG_FILE,
  type CatalogRecord,
  parseCatalogIds,
  STATE_DIRECTORY,
  serializeRecords,
} from "./catalog";
import { type OutputTarget, type SavedPart, uniqueName } from "./target";

const CATALOG_FLUSH_EVERY = 20;

async function exists(directory: FileSystemDirectoryHandle, name: string): Promise<boolean> {
  try {
    await directory.getFileHandle(name);
    return true;
  } catch {
    return false;
  }
}

export class FolderTarget implements OutputTarget {
  private readonly reserved = new Set<string>();
  private readonly directories = new Map<string, Promise<FileSystemDirectoryHandle>>();
  private pendingRecords: CatalogRecord[] = [];
  private catalogQueue: Promise<void> = Promise.resolve();
  private reservations: Promise<unknown> = Promise.resolve();

  constructor(private readonly root: FileSystemDirectoryHandle) {}

  async existingIds(): Promise<Set<string>> {
    try {
      const state = await this.root.getDirectoryHandle(STATE_DIRECTORY);
      const file = await (await state.getFileHandle(CATALOG_FILE)).getFile();
      return parseCatalogIds(await file.text());
    } catch {
      return new Set();
    }
  }

  async write(folder: string, stem: string, extension: string, data: Blob): Promise<string> {
    const directory = await this.directory(folder);
    const name = await this.reserve(folder, directory, stem, extension);
    const path = `${folder}/${name}`;
    const handle = await directory.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    try {
      await writable.write(data);
      await writable.close();
    } catch (error) {
      await writable.abort().catch(() => undefined);
      await directory.removeEntry(name).catch(() => undefined);
      this.reserved.delete(path);
      throw error;
    }
    return path;
  }

  async record(record: CatalogRecord): Promise<void> {
    this.pendingRecords.push(record);
    if (this.pendingRecords.length >= CATALOG_FLUSH_EVERY) await this.flush();
  }

  async finish(): Promise<SavedPart[]> {
    await this.flush();
    return [];
  }

  private reserve(
    folder: string,
    directory: FileSystemDirectoryHandle,
    stem: string,
    extension: string,
  ): Promise<string> {
    const task = this.reservations.then(async () => {
      let attempt = 1;
      let name = uniqueName(stem, extension, attempt);
      while (this.reserved.has(`${folder}/${name}`) || (await exists(directory, name))) {
        attempt += 1;
        name = uniqueName(stem, extension, attempt);
      }
      this.reserved.add(`${folder}/${name}`);
      return name;
    });
    this.reservations = task.catch(() => undefined);
    return task;
  }

  private flush(): Promise<void> {
    const records = this.pendingRecords;
    this.pendingRecords = [];
    this.catalogQueue = this.catalogQueue.then(async () => {
      if (records.length === 0) return;
      const state = await this.root.getDirectoryHandle(STATE_DIRECTORY, { create: true });
      const handle = await state.getFileHandle(CATALOG_FILE, { create: true });
      const size = (await handle.getFile()).size;
      const writable = await handle.createWritable({ keepExistingData: true });
      await writable.seek(size);
      await writable.write(serializeRecords(records));
      await writable.close();
    });
    return this.catalogQueue;
  }

  private directory(path: string): Promise<FileSystemDirectoryHandle> {
    const cached = this.directories.get(path);
    if (cached) return cached;
    const created = path
      .split("/")
      .reduce<Promise<FileSystemDirectoryHandle>>(
        async (parent, part) => (await parent).getDirectoryHandle(part, { create: true }),
        Promise.resolve(this.root),
      );
    this.directories.set(path, created);
    return created;
  }
}
