import type { ImportItem } from "../export/models";
import type { ImportPlan } from "../export/plan";
import type { OutputTarget, SavedPart } from "../output/target";
import { importItem } from "./importer";

const WORKERS = 3;
const MAX_REPORTED_ERRORS = 200;

export type JobPhase = "running" | "done" | "cancelled";

export interface JobProgress {
  phase: JobPhase;
  total: number;
  processed: number;
  saved: number;
  skipped: number;
  failed: number;
  bytes: number;
  startedAt: number;
  errors: Array<{ item: ImportItem; message: string }>;
  parts: SavedPart[];
}

export class ImportJob {
  private cancelled = false;
  private readonly progress: JobProgress;

  constructor(
    private readonly plan: ImportPlan,
    private readonly target: OutputTarget,
    private readonly onUpdate: (progress: JobProgress) => void,
  ) {
    this.progress = {
      phase: "running",
      total: plan.items.length,
      processed: 0,
      saved: 0,
      skipped: 0,
      failed: 0,
      bytes: 0,
      startedAt: performance.now(),
      errors: [],
      parts: [],
    };
  }

  cancel(): void {
    this.cancelled = true;
  }

  async run(): Promise<JobProgress> {
    const already = await this.target.existingIds();
    const pending = this.plan.items.filter((item) => !already.has(item.id));
    this.progress.skipped = this.plan.items.length - pending.length;
    this.progress.processed = this.progress.skipped;
    this.emit();

    let next = 0;
    const worker = async (): Promise<void> => {
      while (!this.cancelled && next < pending.length) {
        const item = pending[next];
        next += 1;
        if (item) await this.importOne(item);
      }
    };
    await Promise.all(Array.from({ length: WORKERS }, worker));

    this.progress.parts = await this.target.finish();
    this.progress.phase = this.cancelled ? "cancelled" : "done";
    this.emit();
    return this.progress;
  }

  private async importOne(item: ImportItem): Promise<void> {
    try {
      this.progress.bytes += await importItem(item, this.target);
      this.progress.saved += 1;
    } catch (error) {
      this.progress.failed += 1;
      if (this.progress.errors.length < MAX_REPORTED_ERRORS) {
        this.progress.errors.push({
          item,
          message: error instanceof Error ? error.message : String(error),
        });
      }
    }
    this.progress.processed += 1;
    this.emit();
  }

  private emit(): void {
    this.onUpdate({ ...this.progress });
  }
}
