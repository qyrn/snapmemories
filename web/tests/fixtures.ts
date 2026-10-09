export function box(kind: string, payload: Uint8Array): Uint8Array {
  const result = new Uint8Array(payload.length + 8);
  new DataView(result.buffer).setUint32(0, result.length);
  result.set(new TextEncoder().encode(kind), 4);
  result.set(payload, 8);
  return result;
}

function fullBox(kind: string, payload: Uint8Array): Uint8Array {
  return box(kind, new Uint8Array([0, 0, 0, 0, ...payload]));
}

function uint32s(...values: number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (const [index, value] of values.entries()) view.setUint32(index * 4, value);
  return bytes;
}

function join(...parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  const result = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

export function mp4Bytes(moovFirst: boolean, media: Uint8Array): Uint8Array<ArrayBuffer> {
  const ftyp = box("ftyp", new TextEncoder().encode("isom\0\0\x02\0isomiso2mp41"));
  const mvhd = fullBox("mvhd", join(uint32s(1, 1, 1000, 1000), new Uint8Array(80)));
  const tkhd = fullBox("tkhd", join(uint32s(1, 1, 1), new Uint8Array(68)));
  const mdhd = fullBox("mdhd", join(uint32s(1, 1, 1000, 1000), new Uint8Array(4)));
  const moov = (chunkOffset: number): Uint8Array => {
    const stco = fullBox("stco", uint32s(1, chunkOffset));
    const trak = box("trak", join(tkhd, box("mdia", join(mdhd, box("minf", box("stbl", stco))))));
    return box("moov", join(mvhd, trak));
  };
  const mdat = box("mdat", media);
  if (moovFirst) {
    const moovSize = moov(0).length;
    return join(ftyp, moov(ftyp.length + moovSize + 8), mdat);
  }
  return join(ftyp, mdat, moov(ftyp.length + 8));
}

export function minimalJpeg(): Uint8Array<ArrayBuffer> {
  const app0 = [
    0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0x00,
  ];
  const oldExif = [0xff, 0xe1, 0x00, 0x08, 0x45, 0x78, 0x69, 0x66, 0x00, 0x00];
  const scan = [0xff, 0xda, 0x00, 0x04, 0x01, 0x02, 0x03, 0x04, 0xff, 0xd9];
  return new Uint8Array([0xff, 0xd8, ...app0, ...oldExif, ...scan]);
}
