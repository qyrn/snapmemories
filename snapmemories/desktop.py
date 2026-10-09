import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def open_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["/usr/bin/open", str(folder)])  # noqa: S603
    else:
        subprocess.Popen(["xdg-open", str(folder)])  # noqa: S603, S607


def open_in_browser(url: str) -> None:
    webbrowser.open(url)
