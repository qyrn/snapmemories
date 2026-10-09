export type Sound = "click" | "type" | "whoosh" | "pop" | "chime" | "ding" | "drop" | "notify" | "tick";

export interface SoundEvent {
  time: number;
  sound: Sound;
  volume: number;
}

type Point = [seconds: number, frame: Keyframe];

interface Track {
  element: Element;
  points: Point[];
  easing: string;
}

interface Switch {
  time: number;
  apply: (active: boolean) => void;
}

interface TextEffect {
  element: Element;
  render: (seconds: number) => string;
}

const DEFAULT_EASING = "cubic-bezier(.45,0,.2,1)";

export class Timeline {
  private readonly tracks: Track[] = [];
  private readonly switches: Switch[] = [];
  private readonly texts: TextEffect[] = [];
  private readonly hooks: Array<(seconds: number) => void> = [];
  private animations: Animation[] = [];
  readonly sounds: SoundEvent[] = [];
  duration = 0;

  animate(element: Element, points: Point[], easing = DEFAULT_EASING): this {
    this.tracks.push({ element, points, easing });
    this.extend(points.at(-1)?.[0] ?? 0);
    return this;
  }

  at(time: number, apply: (active: boolean) => void): this {
    this.switches.push({ time, apply });
    this.extend(time);
    return this;
  }

  toggleClass(time: number, element: Element, className: string, until = Number.POSITIVE_INFINITY): this {
    return this.at(time, (active) => {
      element.classList.toggle(className, active && this.current < until);
    });
  }

  text(element: Element, render: (seconds: number) => string): this {
    this.texts.push({ element, render });
    return this;
  }

  typing(element: Element, value: string, start: number, perCharacter = 0.055): this {
    this.text(element, (seconds) => {
      const count = Math.max(0, Math.min(value.length, Math.floor((seconds - start) / perCharacter)));
      return value.slice(0, count);
    });
    for (let index = 0; index < value.length; index += 1) {
      this.sound(start + index * perCharacter, "type", 0.35 + (index % 3) * 0.08);
    }
    this.extend(start + value.length * perCharacter);
    return this;
  }

  counter(
    element: Element,
    from: number,
    to: number,
    start: number,
    end: number,
    format: (value: number) => string,
  ): this {
    return this.text(element, (seconds) => {
      const progress = Math.min(Math.max((seconds - start) / (end - start), 0), 1);
      const eased = 1 - (1 - progress) ** 2;
      return format(from + (to - from) * eased);
    });
  }

  onSeek(hook: (seconds: number) => void): this {
    this.hooks.push(hook);
    return this;
  }

  sound(time: number, sound: Sound, volume = 1): this {
    this.sounds.push({ time, sound, volume });
    return this;
  }

  private current = 0;

  build(): void {
    const duration = this.duration;
    this.animations = this.tracks.map(({ element, points, easing }) => {
      const animation = element.animate(toKeyframes(points, duration, easing), {
        duration: duration * 1000,
        fill: "both",
      });
      animation.pause();
      return animation;
    });
  }

  seek(seconds: number): void {
    this.current = seconds;
    for (const { time, apply } of [...this.switches].sort((a, b) => a.time - b.time)) apply(seconds >= time);
    for (const animation of this.animations) animation.currentTime = seconds * 1000;
    for (const hook of this.hooks) hook(seconds);
    for (const { element, render } of this.texts) {
      const value = render(seconds);
      if (element.textContent !== value) element.textContent = value;
    }
  }

  private extend(seconds: number): void {
    this.duration = Math.max(this.duration, seconds);
  }
}

function toKeyframes(points: Point[], duration: number, easing: string): Keyframe[] {
  const sorted = [...points].sort((a, b) => a[0] - b[0]);
  const first = sorted[0];
  const last = sorted.at(-1);
  if (!first || !last) return [];
  const padded: Point[] = [
    ...(first[0] > 0 ? [[0, first[1]] as Point] : []),
    ...sorted,
    ...(last[0] < duration ? [[duration, last[1]] as Point] : []),
  ];
  return padded.map(([seconds, frame]) => ({ ...frame, offset: seconds / duration, easing }));
}
