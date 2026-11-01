#!/usr/bin/env python3
"""Launch the GridEval Streamlit dashboard with the active project environment."""

import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_FILE = PROJECT_ROOT / "app.py"
URL = "http://localhost:8501"


def _python_candidates() -> list[Path]:
    """Return platform-specific project virtual environment interpreters."""
    if os.name == "nt":
        return [PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"]
    return [PROJECT_ROOT / ".venv" / "bin" / "python"]


def is_wsl() -> bool:
    """Return whether this launcher is running inside Windows Subsystem for Linux."""
    try:
        return "microsoft" in os.uname().release.casefold()
    except AttributeError:
        return False


def _open_browser(delay: float = 1.5) -> None:
    time.sleep(delay)
    if is_wsl():
        browser_commands = (
            ("cmd.exe", ["cmd.exe", "/c", "start", "", URL]),
            ("powershell.exe", ["powershell.exe", "-Command", f"Start-Process '{URL}'"]),
            ("wslview", ["wslview", URL]),
        )
        for executable, command in browser_commands:
            if shutil.which(executable):
                try:
                    subprocess.Popen(
                        command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except OSError:
                    continue
        print(f"Open the dashboard in your browser: {URL}")
        return

    try:
        webbrowser.open(URL)
    except Exception:
        pass


def main() -> int:
    if not APP_FILE.is_file():
        print(f"Dashboard app not found: {APP_FILE}", file=sys.stderr)
        return 1

    python_exec = sys.executable
    if importlib.util.find_spec("streamlit") is None:
        python_exec = next(
            (str(path) for path in _python_candidates() if path.is_file()), ""
        )
        if not python_exec:
            print(
                "Streamlit is missing from this Python environment. Install the project "
                "dashboard dependencies, then run this launcher again.",
                file=sys.stderr,
            )
            return 1

    print(f"Starting GridEval dashboard at {URL}...")
    threading.Thread(target=_open_browser, daemon=True).start()
    return subprocess.call(
        [
            python_exec,
            "-m",
            "streamlit",
            "run",
            str(APP_FILE),
            "--server.headless=true",
            "--server.port=8501",
        ],
        cwd=PROJECT_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
