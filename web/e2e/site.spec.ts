import { expect, test } from "@playwright/test";

test("serves a French page with its own metadata", async ({ page }) => {
  await page.goto("/fr/");
  await expect(page.locator("html")).toHaveAttribute("lang", "fr");
  await expect(page.locator("h1")).toHaveText("Récupère tous tes Souvenirs Snapchat, avec leur vraie date.");
  await expect(page.locator('link[hreflang="en"]')).toHaveAttribute("href", "https://memories.qyrn.dev/");
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
    "href",
    "https://memories.qyrn.dev/fr/",
  );
  const structured = await page.locator('script[type="application/ld+json"]').textContent();
  const graph = JSON.parse(structured ?? "{}")["@graph"] as Array<{ "@type": string }>;
  expect(graph.map((node) => node["@type"])).toEqual(["WebApplication", "VideoObject", "FAQPage"]);
});

test("offers the tutorial video with chapters in each language", async ({ page, request }) => {
  const violations: string[] = [];
  page.on("console", (message) => {
    if (message.text().includes("Content Security Policy")) violations.push(message.text());
  });
  for (const [path, language] of [
    ["/", "en"],
    ["/fr/", "fr"],
  ] as const) {
    await page.goto(path);
    const video = page.locator("#tutorial-video");
    await expect(video).toHaveAttribute("poster", `/video/tutorial-${language}.jpg`);
    const source = await video.locator("source").getAttribute("src");
    expect(source).toBe(`/video/tutorial-${language}.mp4`);
    const response = await request.get(source ?? "");
    expect(response.status()).toBe(200);
    expect(response.headers()["content-type"]).toContain("video/mp4");
    await expect(page.locator("#chapters button")).toHaveCount(8);
  }
  await page.locator("#chapters button").nth(5).click();
  await expect(page.locator("#chapters button").nth(5)).toHaveClass(/active/);
  expect(violations).toEqual([]);
});
