import { defineConfig } from "vitest/config";

import vercel from "./vercel.json" with { type: "json" };

const siteHeaders = Object.fromEntries(
  vercel.headers[0]?.headers.map(({ key, value }) => [key, value]) ?? [],
);

export default defineConfig({
  build: {
    target: "es2023",
    sourcemap: false,
  },
  preview: {
    headers: siteHeaders,
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
