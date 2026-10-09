import path from "node:path";

import { defineConfig, type Plugin } from "vitest/config";

import { writePages, writeSitemap } from "./build/pages.ts";
import vercel from "./vercel.json" with { type: "json" };

const root = import.meta.dirname;
const pages = writePages(root);
const siteHeaders = Object.fromEntries(
  vercel.headers[0]?.headers.map(({ key, value }) => [key, value]) ?? [],
);

function localizedPages(): Plugin {
  return {
    name: "localized-pages",
    configureServer(server) {
      server.watcher.add([path.join(root, "src", "page.html"), path.join(root, "src", "ui", "strings.ts")]);
      server.watcher.on("change", (file) => {
        if (file.endsWith("page.html") || file.endsWith("strings.ts")) {
          writePages(root);
          server.ws.send({ type: "full-reload" });
        }
      });
    },
    writeBundle(options) {
      writeSitemap(options.dir ?? path.join(root, "dist"));
    },
  };
}

export default defineConfig({
  plugins: [localizedPages()],
  build: {
    target: "es2023",
    sourcemap: false,
    rollupOptions: { input: pages },
  },
  preview: {
    headers: siteHeaders,
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
