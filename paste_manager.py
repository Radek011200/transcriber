from dataclasses import asdict, dataclass
from datetime import datetime
import logging
import os
import shutil
import subprocess
import time

try:
    import pyautogui
except Exception as e:
    logging.warning(f"pyautogui unavailable: {e}")
    pyautogui = None

from clipboard_manager import ClipboardManager


@dataclass
class PasteResult:
    success: bool
    method: str
    error: str | None
    copied_to_clipboard: bool
    clipboard_verified: bool
    restored_window: bool
    target_window_id: str | None
    session_type: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


class PasteManager:
    """Coordinates clipboard copy, target focus restoration, and Ctrl+V."""

    def __init__(
        self,
        clipboard_manager: ClipboardManager,
        session_type: str | None = None,
        paste_delay_ms: int = 300,
        restore_previous_window_before_paste: bool = True,
        paste_method_preference: str = "auto",
        press_enter_after_paste: bool = False,
    ):
        self.clipboard_manager = clipboard_manager
        self.session_type = (session_type or os.environ.get("XDG_SESSION_TYPE", "unknown")).lower()
        self.paste_delay_ms = paste_delay_ms
        self.restore_previous_window_before_paste = restore_previous_window_before_paste
        self.paste_method_preference = paste_method_preference
        self.press_enter_after_paste = press_enter_after_paste
        self.tools = {
            "xdotool": shutil.which("xdotool"),
            "xclip": shutil.which("xclip"),
            "wl-copy": shutil.which("wl-copy"),
        }
        logging.info(
            "Paste diagnostics: XDG_SESSION_TYPE=%s, xdotool=%s, xclip=%s, wl-copy=%s, "
            "paste_delay_ms=%s, restore_window=%s, method_preference=%s, press_enter=%s",
            self.session_type,
            bool(self.tools["xdotool"]),
            bool(self.tools["xclip"]),
            bool(self.tools["wl-copy"]),
            self.paste_delay_ms,
            self.restore_previous_window_before_paste,
            self.paste_method_preference,
            self.press_enter_after_paste,
        )

    def get_active_window_id(self) -> str | None:
        """Returns the active X11 window id using xdotool when available."""
        if self.session_type != "x11" or not self.tools["xdotool"]:
            return None
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                check=True,
                capture_output=True,
                text=True,
            )
            window_id = result.stdout.strip()
            logging.info("Active X11 window before recording: %s", window_id or "none")
            return window_id or None
        except Exception as e:
            logging.error("Could not read active X11 window id: %s", e, exc_info=True)
            return None

    def restore_window(self, window_id: str | None) -> bool:
        """Restores focus to a previous X11 window using xdotool."""
        if not self.restore_previous_window_before_paste:
            logging.info("Window restore skipped by settings.")
            return False
        if self.session_type != "x11" or not window_id or not self.tools["xdotool"]:
            logging.warning("Cannot restore target window. session=%s window_id=%s xdotool=%s", self.session_type, window_id, bool(self.tools["xdotool"]))
            return False
        try:
            subprocess.run(["xdotool", "windowactivate", window_id], check=True, capture_output=True, text=True)
            logging.info("Restored target window: %s", window_id)
            return True
        except Exception as e:
            logging.error("Could not restore target window %s: %s", window_id, e, exc_info=True)
            return False

    def paste_text(self, text: str, target_window_id: str | None = None) -> PasteResult:
        """Copies text, verifies the clipboard, then performs best-effort auto-paste."""
        copied = False
        verified = False
        restored = False
        method = "fallback_manual_clipboard_only"
        error = None
        success = False

        try:
            logging.info(
                "Paste requested. text_length=%s target_window_id=%s session_type=%s",
                len(text),
                target_window_id,
                self.session_type,
            )
            copied = self.clipboard_manager.copy_text(text)
            if copied:
                verified = self.clipboard_manager.verify_text(text)
                if self.session_type == "x11":
                    self.clipboard_manager.get_targets()

            if not copied:
                error = "Could not copy text to clipboard."
                return self._result(success, method, error, copied, verified, restored, target_window_id)

            time.sleep(max(0, self.paste_delay_ms) / 1000.0)

            if self.session_type == "wayland":
                logging.warning("Wayland detected: auto-paste/window activation may be blocked. Leaving text in clipboard if paste fails.")

            if target_window_id:
                restored = self.restore_window(target_window_id)
                if restored:
                    time.sleep(0.12)

            method, success, error = self._send_paste_keystroke()
            return self._result(success, method, error, copied, verified, restored, target_window_id)
        except Exception as e:
            logging.error("Auto-paste failed with unexpected exception: %s", e, exc_info=True)
            return self._result(False, method, str(e), copied, verified, restored, target_window_id)

    def _send_paste_keystroke(self) -> tuple[str, bool, str | None]:
        strategies = self._strategies()
        last_error = None
        for strategy in strategies:
            try:
                if strategy == "xdotool" and self.tools["xdotool"] and self.session_type == "x11":
                    subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], check=True, capture_output=True, text=True)
                    if self.press_enter_after_paste:
                        subprocess.run(["xdotool", "key", "--clearmodifiers", "Return"], check=True, capture_output=True, text=True)
                    logging.info("Paste keystroke sent via xdotool.")
                    return "xclip_text_plain_plus_xdotool_ctrl_v", True, None
                if strategy == "pyautogui" and pyautogui is not None:
                    pyautogui.hotkey("ctrl", "v")
                    if self.press_enter_after_paste:
                        pyautogui.press("enter")
                    logging.info("Paste keystroke sent via pyautogui.")
                    if self.session_type == "wayland":
                        return "wl_copy_plus_pyautogui_ctrl_v_best_effort", True, None
                    return "pyperclip_plus_pyautogui_ctrl_v", True, None
            except Exception as e:
                last_error = str(e)
                logging.error("Paste keystroke failed via %s: %s", strategy, e, exc_info=True)
        return "fallback_manual_clipboard_only", False, last_error or "No paste backend available."

    def _strategies(self) -> list[str]:
        if self.paste_method_preference == "xdotool":
            return ["xdotool", "pyautogui"]
        if self.paste_method_preference == "pyautogui":
            return ["pyautogui", "xdotool"]
        if self.session_type == "x11":
            return ["xdotool", "pyautogui"]
        return ["pyautogui"]

    def _result(
        self,
        success: bool,
        method: str,
        error: str | None,
        copied: bool,
        verified: bool,
        restored: bool,
        target_window_id: str | None,
    ) -> PasteResult:
        result = PasteResult(
            success=success,
            method=method,
            error=error,
            copied_to_clipboard=copied,
            clipboard_verified=verified,
            restored_window=restored,
            target_window_id=target_window_id,
            session_type=self.session_type,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        logging.info("Paste result: %s", result.to_dict())
        return result
