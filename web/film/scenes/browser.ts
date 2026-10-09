import { h, icon } from "../engine/dom.ts";

const LOCK = `<svg viewBox="0 0 16 16"><path d="M4.5 7V5a3.5 3.5 0 0 1 7 0v2h.5a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1zm1.5 0h4V5a2 2 0 1 0-4 0z" fill="currentColor"/></svg>`;
const ARROW_LEFT = `<svg viewBox="0 0 16 16"><path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const ARROW_RIGHT = `<svg viewBox="0 0 16 16"><path d="m6 3 5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const RELOAD = `<svg viewBox="0 0 16 16"><path d="M13 8a5 5 0 1 1-1.5-3.55M13 3v3h-3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const DOWNLOAD = `<svg viewBox="0 0 16 16"><path d="M8 2.5v8m0 0L4.8 7.3M8 10.5l3.2-3.2M3 13.5h10" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const ZIP = `<svg viewBox="0 0 24 24"><path d="M6 2h9l5 5v15H6z" fill="#fbf8f1" stroke="#1b1a17" stroke-width="1.6" stroke-linejoin="round"/><path d="M11 4h2v2h-2zm0 4h2v2h-2zm0 4h2v2h-2z" fill="#1b1a17"/><rect x="9" y="15" width="6" height="4" rx="1" fill="#f2542d"/></svg>`;

export interface BrowserTab {
  root: HTMLElement;
  title: HTMLElement;
}

export interface Browser {
  root: HTMLElement;
  snapTab: BrowserTab;
  siteTab: BrowserTab;
  url: HTMLElement;
  caret: HTMLElement;
  content: HTMLElement;
  downloadsButton: HTMLElement;
  downloadsRing: HTMLElement;
  downloadsPanel: HTMLElement;
  downloadRow: HTMLElement;
  downloadBar: HTMLElement;
  downloadStatus: HTMLElement;
}

function tab(favicon: HTMLElement, title: string): BrowserTab {
  const titleNode = h("span", "f-tab-title", title);
  return { root: h("div", "f-tab", favicon, titleNode, h("span", "f-tab-close", "×")), title: titleNode };
}

export function createBrowser(
  snapTitle: string,
  siteTitle: string,
  labels: { recent: string; done: string },
): Browser {
  const snapTab = tab(h("span", "f-favicon f-favicon-snap"), snapTitle);
  const siteTab = tab(h("span", "f-favicon f-favicon-site"), siteTitle);
  const url = h("span", "f-url-text");
  const caret = h("span", "f-caret");
  const downloadsRing = h("span", "f-download-ring");
  const downloadsButton = h("span", "f-tool f-tool-download", icon(DOWNLOAD), downloadsRing);
  const downloadBar = h("span", "f-download-bar-fill");
  const downloadStatus = h("span", "f-download-status", labels.done);
  const downloadRow = h(
    "div",
    "f-download-row",
    icon(ZIP, "f-download-icon"),
    h(
      "span",
      "f-download-meta",
      h("strong", "", "mydata~1791510669777.zip"),
      downloadStatus,
      h("span", "f-download-bar", downloadBar),
    ),
  );
  const downloadsPanel = h("div", "f-downloads", h("p", "f-downloads-title", labels.recent), downloadRow);
  const content = h("div", "f-content");
  const root = h(
    "div",
    "f-browser",
    h("div", "f-tabs", h("span", "f-traffic"), snapTab.root, siteTab.root),
    h(
      "div",
      "f-toolbar",
      icon(ARROW_LEFT, "f-tool"),
      icon(ARROW_RIGHT, "f-tool f-tool-muted"),
      icon(RELOAD, "f-tool"),
      h("div", "f-url", icon(LOCK, "f-lock"), url, caret),
      downloadsButton,
      h("span", "f-avatar"),
    ),
    content,
    downloadsPanel,
  );
  return {
    root,
    snapTab,
    siteTab,
    url,
    caret,
    content,
    downloadsButton,
    downloadsRing,
    downloadsPanel,
    downloadRow,
    downloadBar,
    downloadStatus,
  };
}

export const ZIP_ICON = ZIP;
