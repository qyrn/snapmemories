import type { STRINGS } from "../../src/ui/strings.ts";
import { h, svgUse } from "../engine/dom.ts";

type SiteText = (typeof STRINGS)["en" | "fr"];

export interface SitePage {
  root: HTMLElement;
  dropzone: HTMLElement;
  reading: HTMLElement;
  screens: { drop: HTMLElement; summary: HTMLElement; progress: HTMLElement; done: HTMLElement };
  folderButton: HTMLElement;
  progressBar: HTMLElement;
  progressPercent: HTMLElement;
  progressLabel: HTMLElement;
  progressSaved: HTMLElement;
  doneStats: HTMLElement;
}

function stat(value: string, label: string, valueClass = ""): HTMLElement {
  return h("div", "", h("span", `stat-value ${valueClass}`.trim(), value), h("span", "stat-label", label));
}

export function createSitePage(text: SiteText, sizes: { file: string; total: string }): SitePage {
  const reading = h("p", "status-line", h("span", "spinner"), h("span", "", text.reading));
  const dropzone = h(
    "label",
    "dropzone",
    svgUse("icon-zip", "drop-icon"),
    h("span", "drop-title", text.dropTitle),
    h("span", "drop-hint", text.dropHint),
  );
  const drop = h("div", "screen active", dropzone, h("p", "privacy-line", text.dropPrivacy), reading);

  const folderButton = h("button", "button primary wide", text.chooseFolder);
  const summary = h(
    "div",
    "screen",
    h("ul", "file-list", h("li", "", h("span", "", "mydata~1791510669777.zip"), h("span", "", sizes.file))),
    h("div", "stats", stat("12", text.photos), stat("4", text.videos), stat(sizes.total, text.size)),
    h("ul", "notes", h("li", "", text.noteLocated.replace("{count}", "8"))),
    folderButton,
    h("p", "hint", text.chooseFolderHint),
  );

  const progressBar = h("div", "bar-fill");
  const progressPercent = h("span", "percent", "0%");
  const progressLabel = h("span", "muted", "");
  const progressSaved = h("span", "stat-value", "0");
  const progress = h(
    "div",
    "screen",
    h("h3", "", text.savingTitle),
    h("p", "muted", text.savingText),
    h("div", "progress-head", progressLabel, progressPercent),
    h("div", "bar", progressBar),
    h(
      "div",
      "stats",
      h("div", "", progressSaved, h("span", "stat-label", text.saved)),
      stat("0", text.failed),
      stat("-", text.remaining),
    ),
  );

  const doneStats = h(
    "div",
    "stats",
    stat("16", text.saved, "good"),
    stat("0", text.alreadyThere),
    stat("0", text.failed, "bad"),
  );
  const done = h("div", "screen", h("h3", "", text.doneTitle), h("p", "muted", text.doneText), doneStats);

  const panel = h("div", "panel", drop, summary, progress, done);
  const root = h(
    "div",
    "site-page",
    h(
      "header",
      "site-header",
      h(
        "span",
        "brand",
        svgUse("logo-mark", "brand-mark"),
        h("span", "brand-name", "Snap", h("em", "", "Memories")),
      ),
      h("nav", "site-nav", h("span", "", text.navTutorial), h("span", "", text.navFaq)),
    ),
    h(
      "main",
      "",
      h(
        "section",
        "tool",
        h("div", "section-head", h("h2", "", text.toolTitle), h("p", "", text.toolLead)),
        panel,
      ),
    ),
  );
  return {
    root,
    dropzone,
    reading,
    screens: { drop, summary, progress, done },
    folderButton,
    progressBar,
    progressPercent,
    progressLabel,
    progressSaved,
    doneStats,
  };
}
