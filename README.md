# SnapMemories

**Save all your Snapchat Memories to your computer** — with capture dates, GPS, and overlays merged automatically.

[![Download](https://img.shields.io/badge/Download-v1.0.0-FFFC00?style=for-the-badge&labelColor=000000)](https://github.com/qyrn/snapmemories/releases/latest)
![Platform](https://img.shields.io/badge/Platform-Windows-blue?style=for-the-badge&labelColor=000000)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&labelColor=000000)

---

## Demo

[![Watch the demo](https://img.youtube.com/vi/ccw4OCh8vA8/maxresdefault.jpg)](https://youtu.be/ccw4OCh8vA8)

---

## Download

**→ [Download SnapMemories.exe](https://github.com/qyrn/snapmemories/releases/latest)**

Windows only. No installation required — just double-click and go.

---

## Features

- Downloads all your Snapchat photos and videos in one click
- Embeds original capture date and GPS into each photo (EXIF)
- Merges text overlays and stickers onto images automatically
- Organizes files by year and month (`~/Memories/YYYY/Month YYYY/`)
- Built-in viewer to browse your saved memories
- 10 concurrent downloads — handles thousands of files
- 100% local — no account, no server, no tracking

---

## How to use

**Step 1 — Export your Snapchat data**

Go to [accounts.snapchat.com/accounts/downloadmydata](https://accounts.snapchat.com/accounts/downloadmydata), check *Export your Memories*, select JSON format, and submit. You'll receive an email when the export is ready (a few minutes to a few hours).

**Step 2 — Run SnapMemories**

Double-click `SnapMemories.exe`. The app opens in your browser automatically.

**Step 3 — Drop the ZIP and start**

Drop the ZIP from Snapchat, click Start, and let it run.

> **Note:** Snapchat download links expire after **7 days**. Run SnapMemories as soon as you receive the confirmation email.

---

## For developers

### Run from source

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:7842](http://localhost:7842).

### Build the exe

```bash
build.bat
```

Requires PyInstaller. Outputs `SnapMemories.exe` to the project root.

---

## Privacy

SnapMemories runs entirely on your machine. It only contacts Snapchat's servers to download your own files. No data is collected, transmitted, or stored anywhere other than your computer.
