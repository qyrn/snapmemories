# SnapMemories

Save all your Snapchat Memories to your computer — with capture dates, GPS coordinates, and overlays merged automatically.

---

## Users

Double-click **`SnapMemories.exe`**. The app opens in your browser. Follow the on-screen steps.

> Snapchat download links expire after **7 days**. Run the app as soon as you receive the Snapchat confirmation email.

---

## Developers

### Requirements

- Python 3.9+
- pip

### Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:7842](http://localhost:7842) in your browser.

### Build the .exe

```bash
build.bat
```

The executable `SnapMemories.exe` is output directly to the project root.

---

## What it does

1. You drop the ZIP exported from Snapchat
2. The app reads your list of memories (photos and videos)
3. It downloads each file directly from Snapchat's servers
4. Capture dates and GPS coordinates are embedded into each photo (EXIF)
5. Overlays (text, stickers) are composited onto the images
6. Everything is organized in `~/Memories/` by year and month

**No data ever leaves your computer.** The app only communicates with Snapchat's servers to retrieve your own files.

---

## Project structure

```
├── app.py              # Flask server + all processing logic
├── templates/
│   ├── index.html      # Main interface
│   └── viewer.html     # Memories viewer
├── static/
│   └── favicon.ico / favicon.png
├── requirements.txt
├── build.bat           # PyInstaller build script
└── README.md
```
