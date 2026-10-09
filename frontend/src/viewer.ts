import { getJson, keepServerAlive, type LibraryItem } from "./api.js";
import { byId, element } from "./dom.js";
import { fullDateLabel, monthLabel, parseLocalMoment, plural } from "./format.js";

const SWIPE_THRESHOLD_PX = 40;
const VIDEO_PRELOAD_MARGIN = "300px";

const gallery = byId("gallery");
const lightbox = byId("lightbox");
const lightboxMedia = byId("lb-media");
const previousButton = byId<HTMLButtonElement>("lb-prev");
const nextButton = byId<HTMLButtonElement>("lb-next");
const mapLink = byId<HTMLAnchorElement>("lb-map");

let items: LibraryItem[] = [];
let currentIndex = -1;
let touchStartX = 0;

function fileUrl(item: LibraryItem): string {
  return `/api/library/${encodeURIComponent(item.id)}/file`;
}

function thumbnailUrl(item: LibraryItem): string {
  return `/api/library/${encodeURIComponent(item.id)}/thumbnail`;
}

function groupByMonth(list: LibraryItem[]): Map<string, number[]> {
  const groups = new Map<string, number[]>();
  list.forEach((item, index) => {
    const label = monthLabel(parseLocalMoment(item.taken_at));
    const group = groups.get(label);
    if (group) group.push(index);
    else groups.set(label, [index]);
  });
  return groups;
}

function thumbnail(item: LibraryItem, index: number): HTMLElement {
  const tile = element("button", "thumb");
  tile.type = "button";
  tile.dataset["index"] = String(index);
  tile.setAttribute(
    "aria-label",
    `${item.kind === "video" ? "Video" : "Photo"}, ${fullDateLabel(parseLocalMoment(item.taken_at))}`,
  );

  if (item.kind === "photo") {
    const image = element("img");
    image.src = thumbnailUrl(item);
    image.loading = "lazy";
    image.decoding = "async";
    image.alt = "";
    tile.append(image);
  } else {
    const video = element("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "none";
    video.dataset["src"] = `${fileUrl(item)}#t=0.1`;
    tile.append(video, element("div", "play-badge", "▶ video"));
  }

  if (item.latitude !== null && item.longitude !== null) {
    tile.append(element("div", "gps-dot"));
  }
  return tile;
}

function render(): void {
  const videoObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const video = entry.target as HTMLVideoElement;
        video.src = video.dataset["src"] ?? "";
        video.preload = "metadata";
        videoObserver.unobserve(video);
      }
    },
    { rootMargin: VIDEO_PRELOAD_MARGIN },
  );

  const fragment = document.createDocumentFragment();
  for (const [label, indexes] of groupByMonth(items)) {
    const group = element("section", "month-group");
    const header = element("div", "month-header");
    header.append(element("h2", "month-name", label), element("span", "month-count", String(indexes.length)));
    const grid = element("div", "thumb-grid");
    for (const index of indexes) {
      const item = items[index];
      if (item) grid.append(thumbnail(item, index));
    }
    group.append(header, grid);
    fragment.append(group);
  }
  gallery.replaceChildren(fragment);
  gallery.hidden = false;
  for (const video of gallery.querySelectorAll("video[data-src]")) videoObserver.observe(video);

  const photos = items.filter((item) => item.kind === "photo").length;
  byId("nav-stats").textContent = `${plural(photos, "photo")} · ${plural(items.length - photos, "video")}`;
}

function stopLightboxVideo(): void {
  const video = lightboxMedia.querySelector("video");
  if (video) {
    video.pause();
    video.removeAttribute("src");
    video.load();
  }
}

function show(index: number): void {
  const item = items[index];
  if (!item) return;
  currentIndex = index;
  stopLightboxVideo();

  if (item.kind === "photo") {
    const image = element("img");
    image.src = fileUrl(item);
    image.alt = "";
    lightboxMedia.replaceChildren(image);
  } else {
    const video = element("video");
    video.src = fileUrl(item);
    video.controls = true;
    video.autoplay = true;
    video.playsInline = true;
    lightboxMedia.replaceChildren(video);
  }

  byId("lb-date").textContent = fullDateLabel(parseLocalMoment(item.taken_at));
  byId("lb-counter").textContent = `${index + 1} / ${items.length}`;
  byId<HTMLAnchorElement>("lb-download").href = fileUrl(item);

  const located = item.latitude !== null && item.longitude !== null;
  mapLink.hidden = !located;
  if (located) {
    mapLink.href = `https://www.openstreetmap.org/?mlat=${item.latitude}&mlon=${item.longitude}#map=15/${item.latitude}/${item.longitude}`;
  }
  previousButton.hidden = index === 0;
  nextButton.hidden = index === items.length - 1;
}

function open(index: number): void {
  show(index);
  lightbox.classList.add("open");
  document.body.style.overflow = "hidden";
  byId("lb-close").focus();
}

function close(): void {
  lightbox.classList.remove("open");
  document.body.style.overflow = "";
  stopLightboxVideo();
  gallery.querySelector<HTMLElement>(`[data-index="${currentIndex}"]`)?.focus();
}

function step(delta: number): void {
  const target = currentIndex + delta;
  if (target >= 0 && target < items.length) show(target);
}

gallery.addEventListener("click", (event) => {
  const tile = (event.target as HTMLElement).closest<HTMLElement>("[data-index]");
  if (tile) open(Number(tile.dataset["index"]));
});

byId("lb-close").addEventListener("click", close);
previousButton.addEventListener("click", () => step(-1));
nextButton.addEventListener("click", () => step(1));
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) close();
});

document.addEventListener("keydown", (event) => {
  if (!lightbox.classList.contains("open")) return;
  if (event.key === "Escape") close();
  if (event.key === "ArrowLeft") step(-1);
  if (event.key === "ArrowRight") step(1);
});

lightboxMedia.addEventListener(
  "touchstart",
  (event) => {
    touchStartX = event.changedTouches[0]?.clientX ?? 0;
  },
  { passive: true },
);
lightboxMedia.addEventListener(
  "touchend",
  (event) => {
    const deltaX = (event.changedTouches[0]?.clientX ?? 0) - touchStartX;
    if (Math.abs(deltaX) >= SWIPE_THRESHOLD_PX) step(deltaX < 0 ? 1 : -1);
  },
  { passive: true },
);

async function load(): Promise<void> {
  try {
    items = await getJson<LibraryItem[]>("/api/library");
  } catch {
    byId("empty-title").textContent = "Could not load your memories";
    byId("empty-text").textContent = "Restart SnapMemories and try again.";
    items = [];
  }
  byId("loading").hidden = true;
  if (items.length === 0) {
    byId("empty").hidden = false;
    return;
  }
  render();
}

keepServerAlive();
void load();
