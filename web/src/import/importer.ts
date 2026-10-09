import { BlobWriter } from "@zip.js/zip.js";

import type { ArchiveMember, ImportItem } from "../export/models";
import { FORMAT_EXTENSIONS, sniffBlob } from "../media/format";
import { fileStem, isoWithOffset, monthFolder, toLocalTime } from "../media/localTime";
import { withCaptureMetadata } from "../media/mp4Metadata";
import { renderPhoto } from "../media/photo";
import type { OutputTarget } from "../output/target";

function memberExtension(member: ArchiveMember, fallback: string): string {
  const match = /\.([a-z0-9]+)$/i.exec(member.name);
  return match?.[1]?.toLowerCase() ?? fallback;
}

function readMember(member: ArchiveMember): Promise<Blob> {
  return member.entry.getData(new BlobWriter());
}

export async function importItem(item: ImportItem, target: OutputTarget): Promise<number> {
  const local = toLocalTime(item.takenAt);
  const folder = monthFolder(local);
  const stem = fileStem(local);
  const main = await readMember(item.source.main);
  const overlay = item.source.overlay ? await readMember(item.source.overlay) : null;

  let path: string;
  if (item.kind === "photo") {
    const photo = await renderPhoto(main, overlay, local, item.location);
    const extension =
      photo.format === "unknown" ? memberExtension(item.source.main, "jpg") : FORMAT_EXTENSIONS[photo.format];
    path = await target.write(folder, stem, extension, photo.data);
  } else {
    const video = await withCaptureMetadata(main, item.takenAt, item.location).catch(() => main);
    path = await target.write(folder, stem, memberExtension(item.source.main, "mp4"), video);
    if (overlay) {
      const format = await sniffBlob(overlay);
      await target.write(folder, `${stem}_overlay`, FORMAT_EXTENSIONS[format], overlay);
    }
  }

  await target.record({
    item_id: item.id,
    relative_path: path,
    kind: item.kind,
    taken_at: isoWithOffset(local),
    latitude: item.location?.latitude ?? null,
    longitude: item.location?.longitude ?? null,
  });
  return main.size + (overlay?.size ?? 0);
}
