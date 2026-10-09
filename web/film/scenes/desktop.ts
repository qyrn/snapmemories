import type { FilmText } from "../content/film-text.ts";
import { h, icon, svgUse } from "../engine/dom.ts";

const FOLDER = `<svg viewBox="0 0 24 24"><path d="M2 6.5A1.5 1.5 0 0 1 3.5 5H9l2 2h9.5A1.5 1.5 0 0 1 22 8.5v10a1.5 1.5 0 0 1-1.5 1.5h-17A1.5 1.5 0 0 1 2 18.5z" fill="#f2c57c" stroke="#9c7a3c" stroke-width="1.1"/></svg>`;
const ENVELOPE = `<svg viewBox="0 0 48 48"><rect x="5" y="10" width="38" height="28" rx="4" fill="#fbf8f1" stroke="#1b1a17" stroke-width="2.5"/><path d="m7 13 17 13 17-13" fill="none" stroke="#1b1a17" stroke-width="2.5" stroke-linejoin="round"/></svg>`;
const PHOTOS: Array<[symbol: string, name: string, stamp: string, pin: boolean, sticker: boolean]> = [
  ["photo-sunset", "2025-06-01_02-11-34.jpg", "'25 6 01", true, false],
  ["photo-window", "2025-06-08_18-40-02.jpg", "'25 6 08", false, false],
  ["photo-mountain", "2025-06-16_12-43-33.jpg", "'25 6 16", true, false],
  ["photo-city", "2025-06-21_23-05-48.jpg", "'25 6 21", false, true],
  ["photo-party", "2025-06-23_23-11-50.jpg", "'25 6 23", true, false],
];

export interface FolderDialog {
  root: HTMLElement;
  target: HTMLElement;
  selectButton: HTMLElement;
}

export interface Mail {
  root: HTMLElement;
  card: HTMLElement;
  hand: HTMLElement;
}

export interface Explorer {
  root: HTMLElement;
  thumbs: HTMLElement[];
  info: HTMLElement;
}

export function createFolderDialog(text: FilmText): FolderDialog {
  const target = h("li", "f-dialog-folder f-dialog-target", icon(FOLDER), text.dialogFolders[2] ?? "");
  const selectButton = h("span", "f-dialog-primary", text.dialogSelect);
  const root = h(
    "div",
    "f-dialog",
    h("div", "f-dialog-title", text.dialogTitle, h("span", "f-dialog-close", "×")),
    h(
      "div",
      "f-dialog-body",
      h(
        "ul",
        "f-dialog-places",
        ...text.dialogPlaces.map((place, index) => h("li", index === 4 ? "active" : "", icon(FOLDER), place)),
      ),
      h(
        "ul",
        "f-dialog-list",
        ...text.dialogFolders.slice(0, 2).map((folder) => h("li", "f-dialog-folder", icon(FOLDER), folder)),
        target,
      ),
    ),
    h(
      "div",
      "f-dialog-footer",
      h("span", "", text.dialogFolderLabel),
      h("span", "f-dialog-field", text.dialogFolders[2] ?? ""),
      selectButton,
      h("span", "f-dialog-secondary", text.dialogCancel),
    ),
  );
  return { root, target, selectButton };
}

export function createMail(text: FilmText): Mail {
  const hand = h("span", "f-clock-hand");
  const card = h(
    "div",
    "f-mail-card",
    icon(ENVELOPE, "f-mail-icon"),
    h(
      "div",
      "f-mail-lines",
      h("strong", "", text.mailFrom),
      h("span", "f-mail-blur"),
      h("span", "f-mail-blur short"),
    ),
  );
  const root = h(
    "div",
    "f-mail",
    card,
    h("div", "f-clock", h("span", "f-clock-face", hand), h("span", "", text.mailWait)),
  );
  return { root, card, hand };
}

export function createExplorer(text: FilmText): Explorer {
  const thumbs = PHOTOS.map(([symbol, name, stamp, pin, sticker]) =>
    h(
      "figure",
      "f-thumb",
      h(
        "span",
        "f-thumb-photo",
        svgUse(symbol),
        sticker ? h("span", "f-thumb-sticker", "wow") : null,
        h("span", "f-thumb-stamp", stamp),
        pin ? svgUse("icon-pin", "f-thumb-pin") : null,
      ),
      h("figcaption", "", name),
    ),
  );
  const info = h(
    "div",
    "f-info",
    h("strong", "", text.infoTaken),
    h("span", "", text.infoPlace),
    h("span", "", text.infoStickers),
  );
  const root = h(
    "div",
    "f-explorer",
    h(
      "div",
      "f-explorer-bar",
      h("span", "f-traffic"),
      ...text.explorerPath.flatMap((part, index) => [
        index ? h("span", "f-crumb-sep", "›") : null,
        h("span", "f-crumb", part),
      ]),
    ),
    h("div", "f-explorer-grid", ...thumbs),
    info,
  );
  return { root, thumbs, info };
}
