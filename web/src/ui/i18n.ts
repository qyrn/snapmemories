import { type Language, type MessageKey, STRINGS } from "./strings";

const LANGUAGE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

export function language(): Language {
  return document.documentElement.lang === "fr" ? "fr" : "en";
}

export function t(key: MessageKey, values: Record<string, string | number> = {}): string {
  return STRINGS[language()][key].replace(/\{(\w+)\}/g, (_, name: string) => String(values[name] ?? ""));
}

export function rememberLanguageChoice(link: HTMLAnchorElement): void {
  link.addEventListener("click", () => {
    const chosen = link.hreflang === "fr" ? "fr" : "en";
    document.cookie = `lang=${chosen}; path=/; max-age=${LANGUAGE_COOKIE_MAX_AGE}; SameSite=Lax; Secure`;
  });
}
