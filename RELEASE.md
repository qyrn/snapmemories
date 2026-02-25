# SnapMemories v1.0.0

**Save all your Snapchat Memories locally — with dates, GPS, and overlays.**

## Download

👉 **[SnapMemories.exe](https://github.com/qyrn/snapmemories/releases/download/v1.0.0/SnapMemories.exe)** — Windows, no installation required

---

## What's included

- **Photo & video download** — fetches every memory from your Snapchat export in parallel (10 concurrent connections)
- **EXIF metadata** — original capture date and GPS coordinates embedded directly into JPEG files
- **Overlay merging** — text, stickers, and drawing overlays composited onto photos automatically
- **Smart organization** — files saved to `~/Memories/YYYY/Month YYYY/YYYY-MM-DD_HH-MM-SS.ext`
- **Built-in viewer** — browse all your saved memories grouped by month, with GPS map links
- **Expired link detection** — clearly reports links older than 7 days without crashing

---

## How to use

1. Go to [accounts.snapchat.com/accounts/downloadmydata](https://accounts.snapchat.com/accounts/downloadmydata)
2. Check *Export your Memories*, select **JSON** format, and submit
3. Wait for the confirmation email, then download the ZIP
4. Open `SnapMemories.exe`, drop the ZIP, and click **Start**

> Links expire 7 days after the export — run SnapMemories as soon as you get the email.

---

## Requirements

- Windows 10 or later
- Internet connection (to download from Snapchat's servers)
- No Python or any other software needed

---

## Known limitations

- Windows only (macOS / Linux: run from source)
- MP4 metadata stored as JSON sidecar (`_metadata/` folder) — not embedded in the video container
- Overlay merging requires the base image to be available; if not found, the raw ZIP is saved instead
