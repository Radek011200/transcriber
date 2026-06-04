import logging
import os
import shutil
import subprocess
import tkinter as tk

try:
    import pyperclip
except Exception as e:
    logging.warning(f"pyperclip unavailable: {e}")
    pyperclip = None


class ClipboardManager:
    """Copies and verifies plain text clipboard contents across X11/Wayland."""

    def __init__(self, tk_root: tk.Tk | None = None, session_type: str | None = None):
        self.tk_root = tk_root
        self.session_type = (session_type or os.environ.get("XDG_SESSION_TYPE", "unknown")).lower()
        self._xclip_processes: list[subprocess.Popen] = []
        self.last_copy_method = ""
        self.tools = {
            "xdotool": shutil.which("xdotool"),
            "xclip": shutil.which("xclip"),
            "wl-copy": shutil.which("wl-copy"),
        }
        logging.info(
            "Clipboard diagnostics: XDG_SESSION_TYPE=%s, xdotool=%s, xclip=%s, wl-copy=%s",
            self.session_type,
            bool(self.tools["xdotool"]),
            bool(self.tools["xclip"]),
            bool(self.tools["wl-copy"]),
        )

    def copy_text(self, text: str) -> bool:
        """Copies text as text/plain where the platform tool supports it."""
        logging.info("Clipboard copy requested. text_length=%s", len(text))
        self._reap_xclip_processes()
        self._stop_xclip_processes()
        self.last_copy_method = ""
        methods = self._copy_methods()
        for method in methods:
            try:
                if method == "pyperclip" and pyperclip is not None:
                    pyperclip.copy(text)
                    self.last_copy_method = "pyperclip"
                    logging.info("Clipboard copy succeeded via pyperclip.")
                    return True
                if method == "wl-copy" and self.tools["wl-copy"]:
                    subprocess.run(
                        ["wl-copy", "--type", "text/plain"],
                        input=text,
                        text=True,
                        check=True,
                        capture_output=True,
                        timeout=2,
                    )
                    self.last_copy_method = "wl-copy"
                    logging.info("Clipboard copy succeeded via wl-copy text/plain.")
                    return True
                if method == "xclip" and self.tools["xclip"]:
                    process = subprocess.Popen(
                        ["xclip", "-selection", "clipboard", "-t", "text/plain"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        start_new_session=True,
                    )
                    try:
                        process.stdin.write(text)
                        process.stdin.close()
                    except Exception:
                        process.kill()
                        raise
                    self._xclip_processes.append(process)
                    self.last_copy_method = "xclip"
                    logging.info("Clipboard copy handed off to xclip. pid=%s", process.pid)
                    return True
                if method == "tkinter" and self.tk_root is not None:
                    self.tk_root.clipboard_clear()
                    self.tk_root.clipboard_append(text)
                    self.tk_root.update_idletasks()
                    self.last_copy_method = "tkinter"
                    logging.info("Clipboard copy succeeded via tkinter.")
                    return True
            except Exception as e:
                logging.error("Clipboard copy failed via %s: %s", method, e, exc_info=True)

        logging.error("Clipboard copy failed for all methods.")
        return False

    def verify_text(self, expected_text: str) -> bool:
        """Checks that the clipboard currently exposes the expected text."""
        actual = self.read_text()
        verified = actual == expected_text
        logging.info(
            "Clipboard verify result=%s expected_length=%s actual_length=%s",
            verified,
            len(expected_text),
            len(actual or ""),
        )
        if not verified:
            logging.warning("Clipboard verification failed. Clipboard does not match expected transcription text.")
        return verified

    def read_text(self) -> str | None:
        methods = self._read_methods()
        for method in methods:
            try:
                if method == "pyperclip" and pyperclip is not None:
                    return pyperclip.paste()
                if method == "wl-paste" and shutil.which("wl-paste"):
                    result = subprocess.run(
                        ["wl-paste", "--no-newline"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=1,
                    )
                    return result.stdout
                if method == "xclip" and self.tools["xclip"]:
                    result = subprocess.run(
                        ["xclip", "-selection", "clipboard", "-o", "-t", "text/plain"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=1,
                    )
                    return result.stdout
                if method == "tkinter" and self.tk_root is not None:
                    return self.tk_root.clipboard_get()
            except Exception as e:
                logging.warning("Clipboard read failed via %s: %s", method, e, exc_info=True)
        return None

    def get_targets(self) -> list[str]:
        """Returns X11 clipboard TARGETS when xclip can expose them."""
        if self.session_type != "x11" or not self.tools["xclip"]:
            return []
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o", "-t", "TARGETS"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1,
            )
            targets = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            logging.info("X11 clipboard TARGETS: %s", targets)
            return targets
        except Exception as e:
            logging.error("Could not read X11 clipboard TARGETS: %s", e, exc_info=True)
            return []

    def _reap_xclip_processes(self) -> None:
        live_processes = []
        for process in self._xclip_processes:
            if process.poll() is None:
                live_processes.append(process)
                continue
            try:
                process.wait(timeout=0)
            except Exception:
                logging.debug("Could not reap finished xclip process.", exc_info=True)
        self._xclip_processes = live_processes[-3:]

    def release_clipboard_owner(self) -> None:
        self._stop_xclip_processes()

    def _stop_xclip_processes(self) -> None:
        for process in self._xclip_processes:
            if process.poll() is not None:
                continue
            try:
                process.terminate()
                process.wait(timeout=0.2)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    logging.debug("Could not kill stale xclip process.", exc_info=True)
        self._xclip_processes = []

    def _copy_methods(self) -> list[str]:
        if self.session_type == "wayland":
            return ["wl-copy", "pyperclip", "tkinter"]
        if self.session_type == "x11":
            return ["xclip", "tkinter", "pyperclip"]
        return ["wl-copy", "xclip", "pyperclip", "tkinter"]

    def _read_methods(self) -> list[str]:
        if self.session_type == "wayland":
            return ["pyperclip", "wl-paste", "tkinter"]
        if self.session_type == "x11":
            return ["tkinter", "pyperclip", "xclip"]
        return ["pyperclip", "wl-paste", "xclip", "tkinter"]
