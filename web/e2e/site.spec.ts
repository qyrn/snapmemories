import { expect, test } from "@playwright/test";

test("serves a French page with its own metadata", async ({ page }) => {
  await page.goto("/fr/");
  await expect(page.locator("html")).toHaveAttribute("lang", "fr");
  await expect(page.locator("h1")).toHaveText("Tes Souvenirs Snapchat, rapatriés chez toi.");
  await expect(page.locator('link[hreflang="en"]')).toHaveAttribute("href", "https://memories.qyrn.dev/");
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
    "href",
    "https://memories.qyrn.dev/fr/",
  );
  const structured = await page.locator('script[type="application/ld+json"]').textContent();
  const graph = JSON.parse(structured ?? "{}")["@graph"] as Array<{ "@type": string }>;
  expect(graph.map((node) => node["@type"])).toEqual(["WebApplication", "FAQPage"]);
});

test("shows every tutorial step with a real screenshot", async ({ page }) => {
  const violations: string[] = [];
  page.on("console", (message) => {
    if (message.text().includes("Content Security Policy")) violations.push(message.text());
  });
  await page.goto("/");
  await page.locator("#tuto").scrollIntoViewIfNeeded();
  const images = page.locator("#stage img");
  await expect(images).toHaveCount(8);
  for (const image of await images.all()) {
    await image.evaluate((node: HTMLImageElement) => {
      node.loading = "eager";
      return node.decode();
    });
    expect(await image.evaluate((node: HTMLImageElement) => node.naturalWidth)).toBeGreaterThan(0);
  }
  await page.getByRole("button", { name: "Drop the ZIP files here" }).click();
  await expect(page.locator("#caption")).toContainText("Drop every ZIP file at once");
  await expect(page.locator("#step-label")).toContainText("Step 6 of 8");
  await expect(page.locator("#snap-link")).toBeHidden();
  await page.getByRole("button", { name: "Click “Request Only Memories”" }).click();
  await expect(page.locator("#snap-link")).toBeVisible();
  expect(violations).toEqual([]);
});

test.describe("with animations turned off on a small screen", () => {
  test.use({ reducedMotion: "reduce", viewport: { width: 390, height: 700 } });

  test("does not autoplay, but plays when asked", async ({ page }) => {
    await page.goto("/");
    await page.locator("#stage").scrollIntoViewIfNeeded();
    await page.waitForTimeout(800);
    expect(Number(await page.locator("#scrubber").inputValue())).toBe(0);
    await page.locator("#play-button").click();
    await expect
      .poll(async () => Number(await page.locator("#scrubber").inputValue()), { timeout: 15_000 })
      .toBeGreaterThan(1);
  });
});
