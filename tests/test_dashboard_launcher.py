import sys
import unittest
from unittest.mock import patch

import run_dashboard


class TestDashboardLauncher(unittest.TestCase):
    @patch("run_dashboard.threading.Thread")
    @patch("run_dashboard.subprocess.call", return_value=0)
    @patch("run_dashboard.importlib.util.find_spec", return_value=object())
    def test_uses_active_python_without_recursive_launcher(self, _streamlit, mock_call, mock_thread):
        self.assertEqual(run_dashboard.main(), 0)
        command = mock_call.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "streamlit"])
        self.assertIn(str(run_dashboard.APP_FILE), command)
        self.assertNotIn(str(run_dashboard.PROJECT_ROOT / "run_dashboard.py"), command)
        mock_thread.return_value.start.assert_called_once()

    def test_wsl_opens_windows_browser_without_gio(self):
        with (
            patch("run_dashboard.time.sleep"),
            patch("run_dashboard.is_wsl", return_value=True),
            patch(
                "run_dashboard.shutil.which",
                side_effect=lambda name: "/usr/bin/cmd.exe" if name == "cmd.exe" else None,
            ),
            patch("run_dashboard.subprocess.Popen") as mock_popen,
            patch("run_dashboard.webbrowser.open") as mock_webbrowser,
        ):
            run_dashboard._open_browser(delay=0)

        mock_popen.assert_called_once_with(
            ["cmd.exe", "/c", "start", "", run_dashboard.URL],
            stdout=run_dashboard.subprocess.DEVNULL,
            stderr=run_dashboard.subprocess.DEVNULL,
        )
        mock_webbrowser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
