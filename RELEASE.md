## What's new in 2.0

- **Current Snapchat exports work.** Photos and videos now come inside the ZIP: SnapMemories reads them directly, no download needed
- **Split exports.** Drop every ZIP file Snapchat sent at once
- **GPS fixed.** Locations were never read before. They are now written into photos and videos
- **No quality loss.** Photos keep their original compression, only the metadata is added
- **Right time zone.** Dates are converted from UTC to your local time
- **Video dates.** Capture date and location are written inside the video file
- **Resume.** Run it again with the same or a newer export: memories already saved are skipped
- **Faster gallery** with cached thumbnails
- **Safer.** Other websites can no longer talk to the app running on your computer
- **macOS build** (unsigned)

## Download

- Windows: `SnapMemories.exe`
- macOS: `SnapMemories-macOS.zip` (right-click the app, then Open)

## Known limitations

- Stickers on videos are saved as a separate PNG next to the video, not burned into it
- HEIC photos are copied as-is, without added metadata
