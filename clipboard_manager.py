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
        methods = self._copy_methods()
        for method in methods:
            try:
                if method == "pyperclip" and pyperclip is not None:
                    pyperclip.copy(text)
                    logging.info("Clipboard copy succeeded via pyperclip.")
                    return True
                if method == "wl-copy" and self.tools["wl-copy"]:
                    subprocess.run(
                        ["wl-copy", "--type", "text/plain"],
                        input=text,
                        text=True,
                        check=True,
                        capture_output=True,
                    )
                    logging.info("Clipboard copy succeeded via wl-copy text/plain.")
                    return True
                if method == "xclip" and self.tools["xclip"]:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard", "-t", "text/plain"],
                        input=text,
                        text=True,
                        check=True,
                        capture_output=True,
                    )
                    logging.info("Clipboard copy succeeded via xclip text/plain.")
                    return True
                if method == "tkinter" and self.tk_root is not None:
                    self.tk_root.clipboard_clear()
                    self.tk_root.clipboard_append(text)
                    self.tk_root.update_idletasks()
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
                    )
                    return result.stdout
                if method == "xclip" and self.tools["xclip"]:
                    result = subprocess.run(
                        ["xclip", "-selection", "clipboard", "-o", "-t", "text/plain"],
                        check=True,
                        capture_output=True,
                        text=True,
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
            )
            targets = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            logging.info("X11 clipboard TARGETS: %s", targets)
            return targets
        except Exception as e:
            logging.error("Could not read X11 clipboard TARGETS: %s", e, exc_info=True)
            return []

    def _copy_methods(self) -> list[str]:
        if self.session_type == "wayland":
            return ["wl-copy", "pyperclip", "tkinter"]
        if self.session_type == "x11":
            return ["xclip", "pyperclip", "tkinter"]
        return ["wl-copy", "xclip", "pyperclip", "tkinter"]

    def _read_methods(self) -> list[str]:
        if self.session_type == "wayland":
            return ["pyperclip", "wl-paste", "tkinter"]
        if self.session_type == "x11":
            return ["pyperclip", "xclip", "tkinter"]
        return ["pyperclip", "wl-paste", "xclip", "tkinter"]
