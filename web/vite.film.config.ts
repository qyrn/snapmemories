import path from "node:path";

import { defineConfig } from "vite";

export default defineConfig({
  root: path.join(import.meta.dirname, "film"),
  publicDir: path.join(import.meta.dirname, "public"),
  server: { port: 4174, strictPort: true },
});
