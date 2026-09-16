"""
1-Click Streamlit Dashboard Launcher for VS Code.
Open this file in VS Code and click the green 'Run' (Play) button at the top right,
or execute `python run_dashboard.py` in your terminal.
"""

import os
import sys
import time
import shutil
import threading
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_FILE = PROJECT_ROOT / "app.py"
URL = "http://localhost:8501"


def is_wsl() -> bool:
    """Detect if running inside Windows Subsystem for Linux (WSL)."""
    try:
        if os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"):
            return True
        if "microsoft" in os.uname().release.lower():
            return True
    except Exception:
        pass
    return False


def launch_browser(url: str, delay: float = 1.0):
    """Launch default Windows browser without triggering Linux gio errors."""
    time.sleep(delay)
    
    # 1. WSL Environment: Use Windows host commands
    if is_wsl():
        for cmd in ["cmd.exe", "powershell.exe", "wslview"]:
            if shutil.which(cmd):
                try:
                    if cmd == "cmd.exe":
                        subprocess.Popen(
                            ["cmd.exe", "/c", "start", url],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        return
                    elif cmd == "powershell.exe":
                        subprocess.Popen(
                            ["powershell.exe", "-Command", f"Start-Process '{url}'"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        return
                    elif cmd == "wslview":
                        subprocess.Popen(
                            ["wslview", url],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        return
                except Exception:
                    pass

    # 2. Native Windows / macOS / Linux fallback
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def main():
    print("==================================================================")
    print("  LAUNCHING ENERGY LOAD PREDICTOR STREAMLIT DASHBOARD")
    print("==================================================================")
    print(f"URL: {URL}")
    print("Starting Streamlit server and opening browser...\n")
    
    # Start browser opener in background thread
    threading.Thread(target=launch_browser, args=(URL,), daemon=True).start()
    
    # Run Streamlit CLI inside current interpreter
    try:
        from streamlit.web import cli as stcli
        sys.argv = [
            "streamlit",
            "run",
            str(APP_FILE),
            "--server.headless=true",
            "--server.port=8501",
        ]
        sys.exit(stcli.main())
    except ImportError:
        venv_streamlit = PROJECT_ROOT / ".venv" / "bin" / "streamlit"
        if not venv_streamlit.exists():
            venv_streamlit = PROJECT_ROOT / ".venv" / "Scripts" / "streamlit.exe"
            
        if venv_streamlit.exists():
            subprocess.run([str(venv_streamlit), "run", str(APP_FILE), "--server.port=8501"])
        else:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_FILE), "--server.port=8501"])


if __name__ == "__main__":
    main()
