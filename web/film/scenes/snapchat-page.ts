import type { SnapchatText } from "../content/snapchat-text.ts";
import { h, icon } from "../engine/dom.ts";

const STROKE = `fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"`;
const NAV_ICONS = [
  `<svg viewBox="0 0 24 24"><rect x="4" y="4" width="6.5" height="16" rx="1.5" ${STROKE}/><rect x="13.5" y="4" width="6.5" height="11" rx="1.5" ${STROKE}/></svg>`,
  `<svg viewBox="0 0 24 24"><path d="M7 4.5v15l12-7.5z" ${STROKE}/></svg>`,
  `<svg viewBox="0 0 24 24"><path d="M5 5h14v11H9l-4 3.5z" ${STROKE}/></svg>`,
  `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8" ${STROKE}/><circle cx="12" cy="12" r="3" ${STROKE}/></svg>`,
  `<svg viewBox="0 0 24 24"><path d="M12 4c-3 0-5 2.4-5 5.3v3L5 15c0 1 2.2 1 3.3.6.5 2 2 3.4 3.7 3.4s3.2-1.4 3.7-3.4c1.1.4 3.3.4 3.3-.6l-2-2.7v-3C17 6.4 15 4 12 4z" ${STROKE}/></svg>`,
];
const MEMORIES_ICON = `<svg viewBox="0 0 24 24"><rect x="3.5" y="5" width="11" height="15" rx="2" transform="rotate(-8 9 12.5)" ${STROKE}/><rect x="9" y="4" width="11" height="15" rx="2" ${STROKE}/><path d="m12 14 3-5 2 3" ${STROKE}/></svg>`;
const JSON_ICON = `<svg viewBox="0 0 24 24"><path d="M5 9V6a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v3M5 9v9a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9M5 9h14M12 12v5m0 0-2-2m2 2 2-2" ${STROKE}/></svg>`;
const DATA_ICON = `<svg viewBox="0 0 24 24"><ellipse cx="12" cy="6.5" rx="7" ry="2.5" ${STROKE}/><path d="M5 6.5v11c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-11M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5" ${STROKE}/></svg>`;
const CHECK = `<svg viewBox="0 0 16 16"><path d="m3.5 8.5 3 3 6-7" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const CROSS = `<svg viewBox="0 0 16 16"><path d="m5 5 6 6m0-6-6 6" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>`;
const CHEVRON = `<svg viewBox="0 0 16 16"><path d="m6 4 4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

export interface SnapchatPage {
  root: HTMLElement;
  scroller: HTMLElement;
  exportsCard: HTMLElement;
  seeButton: HTMLElement;
  seeLabel: HTMLElement;
  exportsList: HTMLElement;
  downloadButton: HTMLElement;
  memoriesSwitch: HTMLElement;
  counter: HTMLElement;
  panel: HTMLElement;
  requestButton: HTMLElement;
  selectSection: HTMLElement;
}

function toggle(): HTMLElement {
  return h(
    "span",
    "sc-switch",
    h("span", "sc-switch-knob"),
    icon(CHECK, "sc-switch-on"),
    icon(CROSS, "sc-switch-off"),
  );
}

export function createSnapchatPage(text: SnapchatText): SnapchatPage {
  const header = h(
    "header",
    "sc-header",
    h("span", "sc-logo"),
    h("span", "sc-search", text.search),
    h(
      "nav",
      "sc-nav",
      ...text.nav.map((label, index) =>
        h("span", "sc-nav-item", icon(NAV_ICONS[index] ?? ""), h("small", "", label)),
      ),
    ),
    h("span", "sc-header-right", h("span", "sc-grid"), h("span", "sc-download", text.download)),
  );

  const sidebar = h(
    "aside",
    "sc-sidebar",
    h(
      "div",
      "sc-profile",
      h("span", "sc-snapcode"),
      h("span", "sc-avatar"),
      h("span", "sc-username", "ton_pseudo"),
    ),
    h(
      "ul",
      "sc-menu",
      ...text.menu.map((label, index) =>
        h("li", index === 0 ? "sc-menu-item sc-menu-active" : "sc-menu-item", label, icon(CHEVRON)),
      ),
    ),
  );

  const seeLabel = h("span", "", text.seeExports);
  const seeButton = h("span", "sc-pill", seeLabel);
  const downloadButton = h("span", "sc-pill sc-pill-link", text.downloadButton);
  const exportsList = h("div", "sc-export-row", h("span", "", text.exportFile), downloadButton);
  const exportsCard = h(
    "section",
    "sc-card sc-exports",
    h("div", "sc-card-head", h("span", "sc-card-title", text.exportsTitle)),
    h(
      "div",
      "sc-exports-body",
      h("p", "", text.exportsCreated),
      h("p", "", text.exportsAvailable),
      seeButton,
    ),
    exportsList,
  );

  const counter = h("span", "sc-counter", text.selectedNone);
  const memoriesSwitch = toggle();
  const requestButton = h("span", "sc-request", text.requestButton);
  const panel = h(
    "div",
    "sc-panel",
    h(
      "div",
      "sc-panel-inner",
      icon(MEMORIES_ICON, "sc-panel-icon"),
      h("h3", "", text.panelTitle),
      h("p", "", text.panelText),
      requestButton,
    ),
  );
  const rows = text.rows.map(([title, subtitle], index) => {
    const rowIcon = index === 0 ? MEMORIES_ICON : index === 1 ? JSON_ICON : DATA_ICON;
    return h(
      "div",
      "sc-row",
      icon(rowIcon, "sc-row-icon"),
      h("span", "sc-row-text", h("span", "", title), subtitle ? h("small", "", subtitle) : null),
      index === 0 ? memoriesSwitch : toggle(),
    );
  });
  const firstRow = rows[0];
  const selectSection = h(
    "section",
    "sc-card sc-select",
    h(
      "div",
      "sc-card-head",
      h("span", "sc-step", "1"),
      h("span", "sc-card-title", text.selectTitle),
      counter,
    ),
    ...(firstRow ? [firstRow] : []),
    panel,
    ...rows.slice(1),
  );

  const main = h(
    "div",
    "sc-main",
    h("h2", "sc-heading", text.heading),
    h("div", "sc-banner", text.banner),
    h(
      "div",
      "sc-ready",
      h("h3", "", text.readyTitle),
      h("p", "", text.intro),
      h("p", "", text.paragraph),
      h("p", "sc-limit", text.limit),
    ),
    exportsCard,
    selectSection,
  );

  const scroller = h("div", "sc-scroller", h("div", "sc-layout", sidebar, main));
  const root = h("div", "sc-page", header, scroller);
  return {
    root,
    scroller,
    exportsCard,
    seeButton,
    seeLabel,
    exportsList,
    downloadButton,
    memoriesSwitch,
    counter,
    panel,
    requestButton,
    selectSection,
  };
}
