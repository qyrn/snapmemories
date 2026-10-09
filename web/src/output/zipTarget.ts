import { BlobReader, BlobWriter, TextReader, ZipWriter } from "@zip.js/zip.js";

import { CATALOG_FILE, type CatalogRecord, STATE_DIRECTORY, serializeRecords } from "./catalog";
import { type OutputTarget, type SavedPart, uniqueName } from "./target";

const PART_LIMIT_BYTES = 1024 * 1024 * 1024;

export class ZipTarget implements OutputTarget {
  private writer = this.newWriter();
  private partBytes = 0;
  private readonly parts: SavedPart[] = [];
  private readonly used = new Set<string>();
  private readonly records: CatalogRecord[] = [];
  private queue: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly baseName: string,
    private readonly onPart: (part: SavedPart) => void,
  ) {}

  existingIds(): Promise<Set<string>> {
    return Promise.resolve(new Set());
  }

  write(folder: string, stem: string, extension: string, data: Blob): Promise<string> {
    let attempt = 1;
    while (this.used.has(`${folder}/${uniqueName(stem, extension, attempt)}`)) attempt += 1;
    const path = `${folder}/${uniqueName(stem, extension, attempt)}`;
    this.used.add(path);
    return this.enqueue(async () => {
      if (this.partBytes > 0 && this.partBytes + data.size > PART_LIMIT_BYTES) await this.closePart();
      await this.writer.add(path, new BlobReader(data), { level: 0 });
      this.partBytes += data.size;
      return path;
    });
  }

  record(record: CatalogRecord): Promise<void> {
    this.records.push(record);
    return Promise.resolve();
  }

  finish(): Promise<SavedPart[]> {
    return this.enqueue(async () => {
      await this.writer.add(
        `${STATE_DIRECTORY}/${CATALOG_FILE}`,
        new TextReader(serializeRecords(this.records)),
      );
      await this.closePart();
      return this.parts;
    });
  }

  private newWriter(): ZipWriter<Blob> {
    return new ZipWriter(new BlobWriter("application/zip"), { zip64: true });
  }

  private async closePart(): Promise<void> {
    const blob = await this.writer.close();
    const part: SavedPart = {
      name: `${this.baseName}-${this.parts.length + 1}.zip`,
      url: URL.createObjectURL(blob),
      size: blob.size,
    };
    this.parts.push(part);
    this.onPart(part);
    this.writer = this.newWriter();
    this.partBytes = 0;
  }

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    const result = this.queue.then(task);
    this.queue = result.catch(() => undefined);
    return result;
  }
}
