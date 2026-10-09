import { expect, test } from "@playwright/test";

test("serves a French page with its own metadata", async ({ page }) => {
  await page.goto("/fr/");
  await expect(page.locator("html")).toHaveAttribute("lang", "fr");
  await expect(page.locator("h1")).toHaveText("Tes souvenirs Snapchat, rapatriés chez toi.");
  await expect(page.locator('link[hreflang="en"]')).toHaveAttribute("href", "https://memories.qyrn.dev/");
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
    "href",
    "https://memories.qyrn.dev/fr/",
  );
  const structured = await page.locator('script[type="application/ld+json"]').textContent();
  const graph = JSON.parse(structured ?? "{}")["@graph"] as Array<{ "@type": string }>;
  expect(graph.map((node) => node["@type"])).toEqual(["WebApplication", "FAQPage"]);
});

test("jumps through the tutorial chapters", async ({ page }) => {
  const violations: string[] = [];
  page.on("console", (message) => {
    if (message.text().includes("Content Security Policy")) violations.push(message.text());
  });
  await page.goto("/");
  await page.locator("#tuto").scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: "Drop them here" }).click();
  await expect(page.locator("#caption")).toContainText("drop all the ZIP files at once");
  await page.getByRole("button", { name: "All sorted" }).click();
  await page.locator("#play-button").click();
  await expect(page.locator('[data-counter="photos"]')).toHaveText("1,248");
  expect(violations).toEqual([]);
});
