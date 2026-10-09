import type { Sound, SoundEvent } from "../film/engine/timeline.ts";

export const SAMPLE_RATE = 48_000;

type Voice = (time: number) => number;

function seededNoise(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    return state / 0xffffffff - 0.5;
  };
}

function render(duration: number, voice: Voice): Float32Array {
  const samples = new Float32Array(Math.ceil(duration * SAMPLE_RATE));
  for (let index = 0; index < samples.length; index += 1) samples[index] = voice(index / SAMPLE_RATE);
  return samples;
}

function lowpass(samples: Float32Array, cutoff: (time: number) => number): Float32Array {
  const output = new Float32Array(samples.length);
  let previous = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const rc = 1 / (2 * Math.PI * cutoff(index / SAMPLE_RATE));
    const alpha = 1 / SAMPLE_RATE / (rc + 1 / SAMPLE_RATE);
    previous += alpha * ((samples[index] ?? 0) - previous);
    output[index] = previous;
  }
  return output;
}

const tone = (frequency: number, time: number): number => Math.sin(2 * Math.PI * frequency * time);

function bell(frequency: number, time: number, decay: number): number {
  if (time < 0) return 0;
  const envelope = Math.exp(-time / decay) * Math.min(1, time / 0.004);
  return (
    envelope *
    (tone(frequency, time) + 0.35 * tone(frequency * 2.01, time) + 0.12 * tone(frequency * 3.02, time))
  );
}

function buildSounds(): Record<Sound, Float32Array> {
  const noise = seededNoise(7);
  const click = render(
    0.05,
    (t) => 0.55 * Math.exp(-t / 0.004) * noise() * 2 + 0.35 * Math.exp(-t / 0.008) * tone(2600, t),
  );
  const type = lowpass(
    render(0.04, (t) => Math.exp(-t / 0.006) * noise() * 1.6),
    () => 4200,
  );
  const whoosh = lowpass(
    render(0.7, (t) => Math.sin(Math.PI * Math.min(1, t / 0.7)) ** 2 * noise() * 1.4),
    (t) => 300 + 2600 * Math.sin(Math.PI * Math.min(1, t / 0.7)),
  );
  const pop = render(
    0.14,
    (t) => Math.exp(-t / 0.05) * Math.min(1, t / 0.003) * Math.sin(2 * Math.PI * (520 * t + 2200 * t * t)),
  );
  const chime = render(1.6, (t) => 0.55 * bell(659.25, t, 0.45) + 0.5 * bell(987.77, t - 0.11, 0.6));
  const ding = render(1.2, (t) => 0.6 * bell(1318.5, t, 0.35));
  const notify = render(0.6, (t) => 0.45 * bell(880, t, 0.12) + 0.45 * bell(1174.66, t - 0.13, 0.18));
  const drop = render(
    0.3,
    (t) =>
      Math.exp(-t / 0.07) * Math.sin(2 * Math.PI * (110 * t - 60 * t * t)) +
      0.2 * Math.exp(-t / 0.02) * noise(),
  );
  const tick = render(0.02, (t) => Math.exp(-t / 0.003) * tone(1800, t));
  return { click, type, whoosh, pop, chime, ding, drop, notify, tick };
}

const SOUND_LEVELS: Record<Sound, number> = {
  click: 0.55,
  type: 0.22,
  whoosh: 0.32,
  pop: 0.3,
  chime: 0.3,
  ding: 0.32,
  drop: 0.28,
  notify: 0.34,
  tick: 0.14,
};

const CHORDS = [
  [261.63, 329.63, 392, 493.88],
  [220, 261.63, 329.63, 392],
  [174.61, 220, 261.63, 329.63],
  [196, 246.94, 293.66, 329.63],
];
const CHORD_SECONDS = 4;
const ARPEGGIO_STEP = 0.5;

function music(duration: number): Float32Array {
  const raw = render(duration, (t) => {
    const position = Math.floor(t / CHORD_SECONDS);
    const chord = CHORDS[position % CHORDS.length] ?? [];
    const local = t - position * CHORD_SECONDS;
    const swell = Math.min(1, local / 0.8) * Math.min(1, (CHORD_SECONDS - local) / 0.6 + 0.35);
    let pad = 0;
    for (const note of chord) pad += tone(note, t) + 0.6 * tone(note * 1.003, t) + 0.25 * tone(note / 2, t);
    const step = Math.floor(local / ARPEGGIO_STEP);
    const note = (chord[[0, 2, 1, 3, 2, 1, 3, 2][step % 8] ?? 0] ?? 261.63) * 2;
    const since = local - step * ARPEGGIO_STEP;
    const pluck =
      Math.exp(-since / 0.28) * Math.min(1, since / 0.004) * (tone(note, t) + 0.3 * tone(note * 2, t));
    const fadeIn = Math.min(1, t / 1.2);
    const fadeOut = Math.min(1, (duration - t) / 3);
    return (0.018 * swell * pad + 0.05 * pluck) * fadeIn * Math.max(0, fadeOut);
  });
  return lowpass(raw, () => 2600);
}

export function mix(duration: number, events: SoundEvent[]): Float32Array {
  const sounds = buildSounds();
  const output = music(duration);
  for (const { time, sound, volume } of events) {
    const clip = sounds[sound];
    const offset = Math.round(time * SAMPLE_RATE);
    const gain = SOUND_LEVELS[sound] * volume;
    for (let index = 0; index < clip.length && offset + index < output.length; index += 1) {
      output[offset + index] = (output[offset + index] ?? 0) + (clip[index] ?? 0) * gain;
    }
  }
  for (let index = 0; index < output.length; index += 1)
    output[index] = Math.tanh((output[index] ?? 0) * 1.1);
  return output;
}

export function wavFile(samples: Float32Array): Buffer {
  const channels = 2;
  const data = Buffer.alloc(samples.length * channels * 2);
  samples.forEach((sample, index) => {
    const value = Math.max(-1, Math.min(1, sample)) * 32_767;
    data.writeInt16LE(Math.round(value), index * 4);
    data.writeInt16LE(Math.round(value), index * 4 + 2);
  });
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVEfmt ", 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(SAMPLE_RATE, 24);
  header.writeUInt32LE(SAMPLE_RATE * channels * 2, 28);
  header.writeUInt16LE(channels * 2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}
