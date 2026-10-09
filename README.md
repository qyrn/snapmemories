# SnapMemories

**Save all your Snapchat Memories to your computer**, with the original capture date, GPS location and stickers merged in.

[![Open](https://img.shields.io/badge/Open-memories.qyrn.dev-FFFC00?style=for-the-badge&labelColor=000000)](https://memories.qyrn.dev)
![Platform](https://img.shields.io/badge/Platform-Web%20%7C%20Windows%20%7C%20macOS-blue?style=for-the-badge&labelColor=000000)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&labelColor=000000)

---

## Demo

[![Watch the demo](https://img.youtube.com/vi/ccw4OCh8vA8/maxresdefault.jpg)](https://youtu.be/ccw4OCh8vA8)

---

## Use it

**[memories.qyrn.dev](https://memories.qyrn.dev)**: nothing to install. Your export is read inside your browser and never uploaded. Chrome and Edge on a computer save straight into a folder; other browsers produce ZIP files.

Desktop apps, useful for very old exports that only contain download links: [Windows](https://github.com/qyrn/snapmemories/releases/latest/download/SnapMemories.exe) · [macOS](https://github.com/qyrn/snapmemories/releases/latest/download/SnapMemories-macOS.zip)

---

## Features

- Works with current Snapchat exports, where photos and videos come inside the ZIP, and with older exports that only contain download links
- Accepts exports split into several ZIP files: drop them all at once
- Writes the capture date, time zone and GPS position into each photo (EXIF) without recompressing it
- Writes the capture date and GPS position into each video, so Google Photos, Apple Photos and Windows sort them correctly
- Merges text, stickers and drawings onto photos. Video overlays are kept next to the video as a PNG
- Sorts files by year and month: `~/Memories/2026/2026-05/2026-05-08_19-01-16.jpg`
- Resumes where it stopped: memories already saved are skipped, even after a crash or a new export
- Built-in gallery to browse what you saved
- 100% local: no account, no server, no tracking, no external fonts or scripts

---

## How to use

1. Go to [accounts.snapchat.com/accounts/downloadmydata](https://accounts.snapchat.com/accounts/downloadmydata)
2. Turn on **Export your Memories** and **Export JSON Files**, choose **All time**, then submit
3. When the "Your Snapchat data is ready" email arrives, download every ZIP file it links to
4. Open SnapMemories, drop all the ZIP files, click **Start**

> The links in Snapchat's email expire after **7 days**.

On macOS the app is not signed yet: right-click `SnapMemories.app`, then **Open**.

---

## For developers

Requirements: [uv](https://docs.astral.sh/uv/) and [pnpm](https://pnpm.io/).

Website (`web/`, Vite + TypeScript, deployed on Vercel):

```bash
pnpm --dir web install
pnpm --dir web dev
pnpm --dir web typecheck && pnpm --dir web test && pnpm --dir web e2e
web/node_modules/.bin/biome check .
```

Desktop app (`snapmemories/` + `frontend/`):

```bash
pnpm --dir frontend install
pnpm --dir frontend build
uv run python -m snapmemories
```

The app starts on [http://127.0.0.1:7842](http://127.0.0.1:7842).

Checks:

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest
pnpm --dir frontend typecheck
```

Build the executable:

```bash
uv run pyinstaller --noconfirm --clean SnapMemories.spec
```

Pushing a `v*` tag builds Windows and macOS packages on GitHub Actions and publishes the release.

---

## Privacy

SnapMemories runs entirely on your machine. It only contacts Snapchat's servers when your export contains download links instead of the files themselves. Nothing is collected or sent anywhere else.
