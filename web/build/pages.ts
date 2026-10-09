import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { FAQ_KEYS, type Language, PAGE_PATHS, STRINGS } from "../src/ui/strings.ts";

export const SITE_URL = "https://memories.qyrn.dev";
const LANGUAGES: Language[] = ["en", "fr"];
const OTHER: Record<Language, Language> = { en: "fr", fr: "en" };
const OG_LOCALES: Record<Language, string> = { en: "en_US", fr: "fr_FR" };

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function structuredData(language: Language): string {
  const strings = STRINGS[language];
  const url = `${SITE_URL}${PAGE_PATHS[language]}`;
  const graph = [
    {
      "@type": "WebApplication",
      name: "SnapMemories",
      url,
      description: strings.metaDescription,
      inLanguage: language,
      applicationCategory: "MultimediaApplication",
      operatingSystem: "Any",
      browserRequirements: "Requires a modern web browser",
      isAccessibleForFree: true,
      offers: { "@type": "Offer", price: "0", priceCurrency: "EUR" },
      author: { "@type": "Person", name: "qyrn", url: "https://github.com/qyrn" },
    },
    {
      "@type": "VideoObject",
      name: strings.tutorialTitle,
      description: strings.videoDescription,
      inLanguage: language,
      thumbnailUrl: `${SITE_URL}/video/tutorial-${language}.jpg`,
      contentUrl: `${SITE_URL}/video/tutorial-${language}.mp4`,
      embedUrl: `${url}#tuto`,
      uploadDate: "2026-10-09T08:26:50+02:00",
      duration: "PT51S",
    },
    {
      "@type": "FAQPage",
      inLanguage: language,
      mainEntity: FAQ_KEYS.map(([question, answer]) => ({
        "@type": "Question",
        name: strings[question],
        acceptedAnswer: { "@type": "Answer", text: strings[answer] },
      })),
    },
  ];
  return JSON.stringify({ "@context": "https://schema.org", "@graph": graph }).replaceAll("<", "\\u003c");
}

export function renderPage(template: string, language: Language): string {
  const strings = STRINGS[language];
  const other = OTHER[language];
  const variables: Record<string, string> = {
    lang: language,
    ogLocale: OG_LOCALES[language],
    canonical: `${SITE_URL}${PAGE_PATHS[language]}`,
    urlEn: `${SITE_URL}${PAGE_PATHS.en}`,
    urlFr: `${SITE_URL}${PAGE_PATHS.fr}`,
    otherPath: PAGE_PATHS[other],
    otherLang: other,
    ogImage: `${SITE_URL}/og-${language}.png`,
  };
  const filled = template.replace(/\{\{(\w+)\}\}/g, (match, key: string) => {
    if (key === "structuredData") return structuredData(language);
    if (key in variables) return escapeHtml(variables[key] ?? "");
    if (key in strings) return escapeHtml(strings[key as keyof typeof strings]);
    throw new Error(`Unknown template key ${match}`);
  });
  return filled;
}

export function writePages(root: string): string[] {
  const template = readFileSync(path.join(root, "src", "page.html"), "utf8");
  return LANGUAGES.map((language) => {
    const directory = language === "en" ? root : path.join(root, language);
    mkdirSync(directory, { recursive: true });
    const file = path.join(directory, "index.html");
    writeFileSync(file, renderPage(template, language));
    return file;
  });
}

export function writeSitemap(outputDirectory: string): void {
  const alternates = LANGUAGES.map(
    (language) =>
      `    <xhtml:link rel="alternate" hreflang="${language}" href="${SITE_URL}${PAGE_PATHS[language]}"/>`,
  ).join("\n");
  const urls = LANGUAGES.map(
    (language) =>
      `  <url>\n    <loc>${SITE_URL}${PAGE_PATHS[language]}</loc>\n${alternates}\n    <xhtml:link rel="alternate" hreflang="x-default" href="${SITE_URL}/"/>\n  </url>`,
  ).join("\n");
  writeFileSync(
    path.join(outputDirectory, "sitemap.xml"),
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">\n${urls}\n</urlset>\n`,
  );
}
