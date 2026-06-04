from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import os
import time
import threading
import queue
import sys
import json
import math
import array
import logging
import logging.handlers
import re
import shutil
import subprocess
from contextlib import contextmanager
from typing import TextIO

try:
    import customtkinter as ctk
except ImportError:
    ctk = None

from clipboard_manager import ClipboardManager
from paste_manager import PasteManager, PasteResult
from prompt_processor import (
    DEFAULT_PROMPT_PROCESSING_SYSTEM_PROMPT,
    LLM_MODEL_OPTIONS,
    LLM_MODEL_OPTIONS_BY_ID,
    PromptProcessor,
    check_llm_connection,
    select_paste_output,
)

# --- Global Configuration ---
APP_TITLE = "Azor Transcriber"
# Set to True to print output to the console (standard output/stderr).
VERBOSE = False
LOG_FILENAME = "transcriber.log"
HISTORY_FILENAME = "output/transcription-history.json"
SETTINGS_FILENAME = "output/settings.json"
DEFAULT_MODEL_NAME = "openai/whisper-large-v3-turbo"
DEFAULT_LANGUAGE = "polish"
DEFAULT_NUM_BEAMS = 5
DEFAULT_GLOBAL_HOTKEY = "<ctrl>+<alt>+n"
DEFAULT_PASTE_DELAY_MS = 300
DEFAULT_MAX_RECORD_DURATION = 120
DEFAULT_PROMPT_PROCESSING_MODEL = "qwen3:1.7b"
MAX_RECORD_DURATION_SECONDS = 1800
LONG_FORM_TRANSCRIPTION_THRESHOLD_SECONDS = 30
RECORD_DURATION_PRESETS = [30, 60, 120, 180, 300, 600, 900, 1800]
RECORDING_OVERLAY_WIDTH = 440
RECORDING_OVERLAY_HEIGHT = 188
RECORDING_OVERLAY_TOP_MARGIN = 28
RECORDING_OVERLAY_TRANSPARENT_COLOR = "#010203"

MODEL_OPTIONS = [
    {
        "id": "openai/whisper-tiny",
        "label": "Whisper Tiny",
        "approx_size": "ok. 75 MB",
        "quality": "niska / testowa",
        "speed": "bardzo szybki",
        "recommended_for": "tylko szybkie testy",
        "cpu_suitability": "bardzo dobry, ale slaba jakosc",
        "gpu_suitability": "nieoplacalny na mocnym GPU",
    },
    {
        "id": "openai/whisper-small",
        "label": "Whisper Small",
        "approx_size": "ok. 466 MB",
        "quality": "dobra",
        "speed": "szybki",
        "recommended_for": "najlepszy kompromis bez GPU",
        "cpu_suitability": "rekomendowany CPU",
        "gpu_suitability": "szybki, ale mozna uzyc lepszego",
    },
    {
        "id": "openai/whisper-medium",
        "label": "Whisper Medium",
        "approx_size": "ok. 1.5 GB",
        "quality": "dobra/bardzo dobra",
        "speed": "sredni",
        "recommended_for": "lepsza jakosc przy akceptowalnym czasie",
        "cpu_suitability": "moze byc wolny",
        "gpu_suitability": "dobry",
    },
    {
        "id": "openai/whisper-large-v3-turbo",
        "label": "Whisper Large v3 Turbo",
        "approx_size": "ok. 1.6 GB",
        "quality": "bardzo dobra",
        "speed": "dobry na GPU",
        "recommended_for": "rekomendowany dla RTX 5070 Ti",
        "cpu_suitability": "moze byc wolny",
        "gpu_suitability": "rekomendowany GPU",
    },
    {
        "id": "bardsai/whisper-medium-pl",
        "label": "Whisper Medium PL",
        "approx_size": "ok. 1.5 GB",
        "quality": "bardzo dobra dla polskiego",
        "speed": "sredni",
        "recommended_for": "polska transkrypcja",
        "cpu_suitability": "dobry, ale wolniejszy",
        "gpu_suitability": "bardzo dobry",
    },
    {
        "id": "bardsai/whisper-large-v2-pl",
        "label": "Whisper Large v2 PL",
        "approx_size": "ok. 3 GB",
        "quality": "bardzo wysoka dla polskiego",
        "speed": "wolniejszy",
        "recommended_for": "najwyzsza jakosc PL",
        "cpu_suitability": "raczej zbyt wolny do szybkiego dyktowania",
        "gpu_suitability": "dobry na mocnym GPU",
    },
]
MODEL_OPTIONS_BY_ID = {model["id"]: model for model in MODEL_OPTIONS}
MODEL_LABEL_TO_ID = {f'{model["label"]} ({model["id"]})': model["id"] for model in MODEL_OPTIONS}
POSTPROCESS_REPLACEMENTS = {
    "data tables": "DataTables",
    "w m s": "WMS",
    "kohana": "Kohana",
    "sql": "SQL",
    "git hab": "GitHub",
    "doker": "Docker",
    "whisper": "Whisper",
    "pi ajdio": "PyAudio",
}

# --- Logging Setup ---
class StreamToLogger(TextIO):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    This captures stdout/stderr, including print() statements.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        # Handle buffer and write line by line
        for line in buf.rstrip().splitlines():
            # Check if the line is not empty (prevents logging empty lines from print())
            if line.strip():
                self.logger.log(self.level, line.strip())

    def flush(self):
        # Required by TextIO interface, but we flush line-by-line in write
        pass

# Configure the global logger BEFORE application startup
def setup_logging():
    """Con gures the logging system to save all output to a le and optionally to console."""
    os.makedirs('output', exist_ok=True)
    
    # 1. Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Capture everything from INFO level up

    # 2. File Handler (Always active)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILENAME, 
        maxBytes=1024*1024*5, # 5 MB per file
        backupCount=5,
        encoding='utf-8'
    )
    # Define a simple formatter for the file
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 3. Console Handler (Only active if VERBOSE is True)
    if VERBOSE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 4. Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(root_logger, logging.INFO)
    sys.stderr = StreamToLogger(root_logger, logging.ERROR)

setup_logging()
logging.info("Application initialization started.")

# --- Whisper Dependencies ---
# Ensure you have installed: pip install torch transformers librosa
# (Librosa might require ffmpeg)
try:
    import torch
    from transformers import pipeline
except ImportError:
    logging.error("ERROR: 'transformers' or 'torch' libraries not found.")
    logging.error("Install them using: pip install torch transformers")
    exit()

try:
    from pynput import keyboard
except Exception as e:
    logging.warning(f"pynput unavailable: {e}")
    keyboard = None

def output_filename()  -> str:
    """Generates output filename for transcription results."""
    os.makedirs('output', exist_ok=True)
    return f"output/recording-{int(time.time())}.wav"

def history_filename() -> str:
    """Returns the path used for persisted transcription history."""
    os.makedirs('output', exist_ok=True)
    return HISTORY_FILENAME

def settings_filename() -> str:
    """Returns the path used for persisted application settings."""
    os.makedirs('output', exist_ok=True)
    return SETTINGS_FILENAME

@contextmanager
def suppress_native_stderr():
    """Suppresses noisy native ALSA/JACK stderr emitted by PortAudio probing."""
    if VERBOSE:
        yield
        return
    saved_stderr_fd = None
    devnull = None
    try:
        saved_stderr_fd = os.dup(2)
        devnull = open(os.devnull, "w", encoding="utf-8")
        os.dup2(devnull.fileno(), 2)
    except OSError:
        saved_stderr_fd = None
        devnull = None
    try:
        yield
    finally:
        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
            except OSError:
                pass
        if devnull is not None:
            devnull.close()

def load_settings() -> dict:
    """Loads user settings from disk or returns sane defaults."""
    defaults = {
        "selected_model": DEFAULT_MODEL_NAME,
        "language": DEFAULT_LANGUAGE,
        "num_beams": DEFAULT_NUM_BEAMS,
        "input_device_index": None,
        "dictation_mode_enabled": True,
        "auto_paste_enabled": True,
        "copy_to_clipboard_enabled": True,
        "global_hotkey_enabled": True,
        "global_hotkey": DEFAULT_GLOBAL_HOTKEY,
        "floating_window_enabled": False,
        "append_space_after_paste": True,
        "append_newline_after_paste": False,
        "press_enter_after_paste": False,
        "trim_text_before_paste": True,
        "capitalize_first_letter": False,
        "paste_delay_ms": DEFAULT_PASTE_DELAY_MS,
        "max_record_duration": DEFAULT_MAX_RECORD_DURATION,
        "max_record_duration_seconds": DEFAULT_MAX_RECORD_DURATION,
        "record_duration_preset": "120 s",
        "custom_record_duration_seconds": DEFAULT_MAX_RECORD_DURATION,
        "auto_stop_recording_enabled": True,
        "auto_transcribe_after_stop": True,
        "auto_paste_after_transcription": True,
        "show_recording_widget": True,
        "show_audio_waveform": True,
        "audio_level_smoothing": 0.35,
        "restore_previous_window_before_paste": True,
        "paste_method_preference": "auto",
        "show_wayland_warning": True,
        "appearance_mode": "dark",
        "ui_scaling": "100%",
        "prompt_processing_enabled": True,
        "prompt_processing_default_for_recordings": False,
        "prompt_processing_backend": "ollama",
        "prompt_processing_model": DEFAULT_PROMPT_PROCESSING_MODEL,
        "prompt_processing_ollama_url": "http://localhost:11434",
        "prompt_processing_gguf_path": "",
        "prompt_processing_system_prompt": DEFAULT_PROMPT_PROCESSING_SYSTEM_PROMPT,
        "prompt_processing_temperature": 0.2,
        "prompt_processing_max_tokens": 1024,
        "paste_output_preference": "original",
        "fallback_to_original_on_llm_error": True,
    }
    try:
        with open(settings_filename(), 'r', encoding='utf-8') as settings_file:
            settings = json.load(settings_file)
    except FileNotFoundError:
        save_settings(defaults)
        return defaults
    except (json.JSONDecodeError, OSError) as e:
        logging.error(f"Could not load settings: {e}", exc_info=True)
        return defaults

    if not isinstance(settings, dict):
        logging.warning("Settings file has invalid format. Using defaults.")
        return defaults

    merged_settings = defaults | settings
    if merged_settings["selected_model"] not in MODEL_OPTIONS_BY_ID:
        logging.warning(f"Unknown configured model: {merged_settings['selected_model']}. Using default.")
        merged_settings["selected_model"] = DEFAULT_MODEL_NAME
    if merged_settings["prompt_processing_model"] not in LLM_MODEL_OPTIONS_BY_ID:
        logging.warning(f"Unknown configured LLM model: {merged_settings['prompt_processing_model']}. Using default.")
        merged_settings["prompt_processing_model"] = DEFAULT_PROMPT_PROCESSING_MODEL
    if merged_settings["prompt_processing_backend"] not in {"ollama", "llama-cpp-python"}:
        merged_settings["prompt_processing_backend"] = "ollama"
    if merged_settings["paste_output_preference"] not in {"original", "processed"}:
        merged_settings["paste_output_preference"] = "original"
    return merged_settings

def save_settings(settings: dict):
    """Persists user settings to disk."""
    try:
        with open(settings_filename(), 'w', encoding='utf-8') as settings_file:
            json.dump(settings, settings_file, ensure_ascii=False, indent=2)
    except OSError as e:
        logging.error(f"Could not save settings: {e}", exc_info=True)

def postprocess_transcription(text: str) -> str:
    """Applies small, easy-to-edit corrections for common technical terms."""
    processed_text = text
    for wrong, correct in POSTPROCESS_REPLACEMENTS.items():
        processed_text = re.sub(rf"\b{re.escape(wrong)}\b", correct, processed_text, flags=re.IGNORECASE)
    return processed_text

def preprocess_audio(input_path: str) -> str:
    """Hook for future normalization, silence trimming, or noise reduction."""
    return input_path

def get_audio_duration_seconds(audio_path: str) -> float:
    """Returns WAV duration in seconds when possible."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            frame_count = wf.getnframes()
            frame_rate = wf.getframerate()
            return round(frame_count / float(frame_rate), 2) if frame_rate else 0.0
    except (wave.Error, OSError) as e:
        logging.warning(f"Could not calculate audio duration for {audio_path}: {e}")
        return 0.0

def build_asr_call_kwargs(duration_seconds: float, generate_kwargs: dict) -> dict:
    call_kwargs = {"generate_kwargs": generate_kwargs}
    if duration_seconds > LONG_FORM_TRANSCRIPTION_THRESHOLD_SECONDS:
        call_kwargs["return_timestamps"] = True
    return call_kwargs

def detect_session_type() -> str:
    """Returns x11, wayland, or unknown based on the Linux session."""
    return os.environ.get("XDG_SESSION_TYPE", "unknown").lower()

def get_primary_monitor_bounds() -> tuple[int, int, int, int] | None:
    """Returns primary monitor bounds as x, y, width, height on X11 when xrandr is available."""
    if shutil.which("xrandr") is None:
        return None
    try:
        result = subprocess.run(
            ["xrandr", "--query"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        )
    except Exception as e:
        logging.debug(f"Could not query monitors with xrandr: {e}")
        return None
    if result.returncode != 0:
        return None
    primary_match = None
    first_connected_match = None
    pattern = re.compile(r"\bconnected\b(?:\s+primary)?\s+(\d+)x(\d+)\+(-?\d+)\+(-?\d+)")
    for line in result.stdout.splitlines():
        if " connected" not in line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        if first_connected_match is None:
            first_connected_match = match
        if " primary " in f" {line} ":
            primary_match = match
            break
    match = primary_match or first_connected_match
    if match is None:
        return None
    width, height, x, y = (int(group) for group in match.groups())
    return x, y, width, height

def is_transcription_error(text: str) -> bool:
    """Returns True for internal transcription error messages that must not be pasted."""
    return text.strip().upper().startswith("ERROR:")

def is_likely_silence_hallucination(text: str) -> bool:
    normalized = re.sub(r"[^\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", " ", text, flags=re.UNICODE).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    silence_phrases = {
        "dziękuję za uwagę",
        "dziekuje za uwage",
        "dziękuję",
        "dziekuje",
    }
    return normalized in silence_phrases

def prepare_text_for_paste(
    text: str,
    trim_text: bool = True,
    capitalize_first_letter: bool = False,
    append_space: bool = True,
    append_newline: bool = False,
) -> str:
    """Formats transcription text before clipboard copy and auto-paste."""
    prepared_text = text.strip() if trim_text else text
    if capitalize_first_letter and prepared_text:
        prepared_text = prepared_text[0].upper() + prepared_text[1:]
    if append_newline and prepared_text:
        prepared_text += "\n"
    elif append_space and prepared_text:
        prepared_text += " "
    return prepared_text

def show_desktop_notification(title: str, message: str) -> None:
    """Shows a Linux desktop notification when notify-send is available."""
    if shutil.which("notify-send") is None:
        return
    try:
        subprocess.run(
            ["notify-send", title, message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logging.warning(f"Could not show desktop notification: {e}")

def format_history_timestamp(timestamp: float) -> str:
    """Formats a unix timestamp for display in the history tab."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes = seconds // 60
    sec = seconds % 60
    return f"{minutes:02d}:{sec:02d}"

# === 2. Recording Configuration ===
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Standard for speech models (Whisper)
MAX_RECORD_DURATION = DEFAULT_MAX_RECORD_DURATION # Maximum recording length in seconds

# === 3. CustomTkinter GUI Application ===
class AudioRecorderApp:
    def __init__(self, master):
        self.master = master
        self.master.title(APP_TITLE)
        self.master.geometry("1180x640")
        self.master.minsize(1100, 580)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        logging.info("CustomTkinter GUI initialization started.")

        try:
            with suppress_native_stderr():
                self.p = pyaudio.PyAudio()
        except Exception as e:
            logging.critical(f"Could not initialize PyAudio: {e}. Destroying GUI.")
            messagebox.showerror("PyAudio Error", f"Could not initialize PyAudio: {e}\nDo you have 'portaudio' installed?")
            master.destroy()
            return

        self.frames = []
        self.stream = None
        self.recording = False
        self.start_time = None
        self.record_timer_id = None
        self.transcription_history = []
        self.selected_history_index = None
        self.history_cards = []
        self.settings = load_settings()

        self.selected_model_name = self.settings["selected_model"]
        self.language = self.settings["language"]
        self.num_beams = int(self.settings["num_beams"])
        self.input_device_index = self.normalize_input_device_index(self.settings.get("input_device_index"))
        self.max_record_duration_seconds = int(self.settings.get("max_record_duration_seconds", DEFAULT_MAX_RECORD_DURATION))
        self.max_record_duration = self.max_record_duration_seconds
        self.record_duration_preset = str(self.settings.get("record_duration_preset", f"{self.max_record_duration_seconds} s"))
        self.custom_record_duration_seconds = int(self.settings.get("custom_record_duration_seconds", self.max_record_duration_seconds))
        self.auto_stop_recording_enabled = bool(self.settings.get("auto_stop_recording_enabled", True))
        self.auto_transcribe_after_stop = bool(self.settings.get("auto_transcribe_after_stop", True))
        self.auto_paste_after_transcription = bool(self.settings.get("auto_paste_after_transcription", True))
        self.show_recording_widget = bool(self.settings.get("show_recording_widget", True))
        self.show_audio_waveform = bool(self.settings.get("show_audio_waveform", True))
        self.audio_level_smoothing = float(self.settings.get("audio_level_smoothing", 0.35))
        self.dictation_mode_enabled = bool(self.settings["dictation_mode_enabled"])
        self.auto_paste_enabled = bool(self.settings["auto_paste_enabled"])
        self.copy_to_clipboard_enabled = bool(self.settings["copy_to_clipboard_enabled"])
        self.global_hotkey_enabled = bool(self.settings["global_hotkey_enabled"])
        self.global_hotkey = str(self.settings["global_hotkey"])
        self.floating_window_enabled = bool(self.settings["floating_window_enabled"])
        self.append_space_after_paste = bool(self.settings["append_space_after_paste"])
        self.append_newline_after_paste = bool(self.settings["append_newline_after_paste"])
        self.press_enter_after_paste = bool(self.settings["press_enter_after_paste"])
        self.trim_text_before_paste = bool(self.settings["trim_text_before_paste"])
        self.capitalize_first_letter = bool(self.settings["capitalize_first_letter"])
        self.paste_delay_ms = int(self.settings["paste_delay_ms"])
        self.restore_previous_window_before_paste = bool(self.settings["restore_previous_window_before_paste"])
        self.paste_method_preference = str(self.settings["paste_method_preference"])
        self.show_wayland_warning = bool(self.settings["show_wayland_warning"])
        self.prompt_processing_enabled = bool(self.settings["prompt_processing_enabled"])
        self.prompt_processing_default_for_recordings = bool(self.settings["prompt_processing_default_for_recordings"])
        self.prompt_processing_backend = str(self.settings["prompt_processing_backend"])
        self.prompt_processing_model = str(self.settings["prompt_processing_model"])
        self.prompt_processing_ollama_url = str(self.settings["prompt_processing_ollama_url"])
        self.prompt_processing_gguf_path = str(self.settings["prompt_processing_gguf_path"])
        self.prompt_processing_system_prompt = str(self.settings["prompt_processing_system_prompt"])
        self.prompt_processing_temperature = float(self.settings["prompt_processing_temperature"])
        self.prompt_processing_max_tokens = int(self.settings["prompt_processing_max_tokens"])
        self.paste_output_preference = str(self.settings["paste_output_preference"])
        self.fallback_to_original_on_llm_error = bool(self.settings["fallback_to_original_on_llm_error"])

        self.asr_pipeline = None
        self.current_model_name = None
        self.model_loading = False
        self.transcribing = False
        self.app_state = "idle"
        self.last_transcription_text = ""
        self.last_processed_text = ""
        self.last_prompt_processing_error = None
        self.preview_mode = "original"
        self.history_view_mode = "original"
        self.current_record_prompt_processing_enabled = self.prompt_processing_default_for_recordings
        self.llm_connection_check_running = False
        self.llm_connection_status = None
        self.llm_connection_summary = "Nie sprawdzono"
        self.last_paste_result = None
        self.recording_timer_ui_id = None
        self.waveform_update_id = None
        self.recording_widget_hide_id = None
        self.audio_levels = []
        self.last_audio_level = 0.0
        self.audio_level_peak = 0.0
        self.audio_level_sum = 0.0
        self.audio_level_count = 0
        self.red_dot_pulse = False
        self.current_record_limit_seconds = self.max_record_duration_seconds
        self.session_type = detect_session_type()
        self.target_window_id = None
        self.current_trigger_source = "main_button"
        self.hotkey_listener = None
        self.hotkey_status_text = "Global hotkey inactive"
        self.floating_window = None
        self.floating_button = None
        self.recording_overlay_window = None
        self.recording_overlay_canvas = None
        self.recording_overlay_timer_item = None
        self.recording_overlay_remaining_item = None
        self.recording_overlay_dot_item = None
        self.recording_overlay_llm_item = None
        self.recording_overlay_title_item = None
        self.recording_overlay_stage_item = None
        self.processing_wave_phase = 0
        self.recording_overlay_drag_offset = (0, 0)
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.device_name = self.detect_device_name()
        self.audio_input_devices = self.get_audio_input_devices()
        self.transcription_queue = queue.Queue()

        self.clipboard_manager = ClipboardManager(self.master, self.session_type)
        self.paste_manager = self.create_paste_manager()

        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Detected transcription device: {self.device_name}")
        logging.info(f"Detected session type: {self.session_type}")
        self.log_audio_input_devices()
        if self.device_id == -1:
            logging.warning("CUDA is unavailable. Transcription will be slower on CPU.")
        if self.session_type == "wayland":
            logging.warning("Wayland detected. Global hotkey and auto-paste may be limited.")

        self.load_history()
        self.build_layout()
        self.show_view("dictation")
        self.refresh_history_list()
        self.update_model_info()
        self.update_dictation_status()
        self.update_diagnostics()
        self.set_app_state("idle")

        self.master.after(100, self.check_transcription_queue)
        self.master.after(500, self.check_llm_connection_async)
        self.load_model_async(self.selected_model_name)
        self.start_hotkey_listener()
        self.update_floating_window()
        logging.info("CustomTkinter GUI initialized successfully.")

    def create_paste_manager(self) -> PasteManager:
        return PasteManager(
            clipboard_manager=self.clipboard_manager,
            session_type=self.session_type,
            paste_delay_ms=self.paste_delay_ms,
            restore_previous_window_before_paste=self.restore_previous_window_before_paste,
            paste_method_preference=self.paste_method_preference,
            press_enter_after_paste=self.press_enter_after_paste,
        )

    def detect_device_name(self) -> str:
        if torch.cuda.is_available():
            try:
                return f"CUDA: {torch.cuda.get_device_name(0)}"
            except Exception as e:
                logging.warning(f"Could not read CUDA device name: {e}", exc_info=True)
                return "CUDA"
        return "CPU"

    def normalize_input_device_index(self, value) -> int | None:
        if value in (None, "", "default", "System default"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            logging.warning("Invalid input_device_index setting: %s. Falling back to system default.", value)
            return None

    def get_audio_input_devices(self) -> list[dict]:
        devices = []
        try:
            with suppress_native_stderr():
                for index in range(self.p.get_device_count()):
                    info = self.p.get_device_info_by_index(index)
                    if int(info.get("maxInputChannels", 0)) <= 0:
                        continue
                    devices.append({
                        "index": int(info.get("index", index)),
                        "name": str(info.get("name", "unknown")),
                        "channels": int(info.get("maxInputChannels", 0)),
                        "rate": int(float(info.get("defaultSampleRate", 0) or 0)),
                    })
        except Exception:
            logging.exception("Could not enumerate PyAudio input devices.")
        return devices

    def input_device_label(self, device: dict | None = None) -> str:
        if device is None:
            return "System default"
        return f"{device['index']}: {device['name']} ({device['channels']} ch, {device['rate']} Hz)"

    def input_device_labels(self) -> list[str]:
        return ["System default"] + [self.input_device_label(device) for device in self.audio_input_devices]

    def selected_input_device_label(self) -> str:
        if self.input_device_index is None:
            return "System default"
        for device in self.audio_input_devices:
            if device["index"] == self.input_device_index:
                return self.input_device_label(device)
        return "System default"

    def parse_input_device_label(self, label: str) -> int | None:
        if not label or label == "System default":
            return None
        try:
            return int(label.split(":", 1)[0])
        except (ValueError, IndexError):
            logging.warning("Could not parse input device label: %s", label)
            return None

    def log_audio_input_devices(self) -> None:
        try:
            with suppress_native_stderr():
                default_info = self.p.get_default_input_device_info()
            logging.info(
                "PyAudio default input device: index=%s name=%s channels=%s rate=%s",
                default_info.get("index"),
                default_info.get("name"),
                default_info.get("maxInputChannels"),
                default_info.get("defaultSampleRate"),
            )
        except Exception:
            logging.warning("PyAudio could not report a default input device.", exc_info=True)
        if not self.audio_input_devices:
            logging.warning("No PyAudio input devices were detected.")
            return
        for device in self.audio_input_devices:
            logging.info(
                "PyAudio input device: index=%s name=%s channels=%s rate=%s",
                device["index"],
                device["name"],
                device["channels"],
                device["rate"],
            )

    def build_layout(self):
        self.master.grid_columnconfigure(0, minsize=220)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, minsize=36)

        self.sidebar = ctk.CTkFrame(self.master, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", rowspan=2)
        self.sidebar.grid_propagate(False)

        self.content_area = ctk.CTkFrame(self.master, corner_radius=0, fg_color="transparent")
        self.content_area.grid(row=0, column=1, sticky="nsew", padx=16, pady=(16, 8))
        self.content_area.grid_rowconfigure(0, weight=1)
        self.content_area.grid_columnconfigure(0, weight=1)

        self.status_bar = ctk.CTkFrame(self.master, height=36, corner_radius=0)
        self.status_bar.grid(row=1, column=1, sticky="ew")
        self.status_bar.grid_columnconfigure(0, weight=1)

        self.views = {}
        self.nav_buttons = {}
        self.build_sidebar()
        self.build_status_bar()
        self.build_dictation_view()
        self.build_history_view()
        self.build_models_view()
        self.build_settings_view()
        self.build_diagnostics_view()

    def build_sidebar(self):
        self.sidebar.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.sidebar, text="Azor Transcriber", font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, padx=18, pady=(24, 0), sticky="w")
        ctk.CTkLabel(self.sidebar, text="Speech2Text PL", text_color="gray70", font=ctk.CTkFont(size=12)).grid(row=1, column=0, padx=18, pady=(2, 24), sticky="w")

        nav = [
            ("dictation", "🎙 Dyktowanie"),
            ("history", "🕘 Historia"),
            ("models", "🧠 Modele"),
            ("settings", "⚙ Ustawienia"),
            ("diagnostics", "🧪 Diagnostyka"),
        ]
        for row, (view_name, label) in enumerate(nav, start=2):
            button = ctk.CTkButton(
                self.sidebar,
                text=label,
                height=38,
                anchor="w",
                command=lambda name=view_name: self.show_view(name),
            )
            button.grid(row=row, column=0, padx=14, pady=5, sticky="ew")
            self.nav_buttons[view_name] = button

        self.sidebar.grid_rowconfigure(20, weight=1)
        self.sidebar_model_label = ctk.CTkLabel(self.sidebar, text="Model: ...", text_color="gray75", anchor="w", justify="left")
        self.sidebar_model_label.grid(row=21, column=0, padx=18, pady=(8, 2), sticky="ew")
        self.sidebar_device_label = ctk.CTkLabel(self.sidebar, text=f"Device: {self.device_name}", text_color="gray75", anchor="w", justify="left")
        self.sidebar_device_label.grid(row=22, column=0, padx=18, pady=2, sticky="ew")
        self.sidebar_session_label = ctk.CTkLabel(self.sidebar, text=f"Session: {self.session_type}", text_color="gray75", anchor="w", justify="left")
        self.sidebar_session_label.grid(row=23, column=0, padx=18, pady=(2, 18), sticky="ew")

    def build_status_bar(self):
        self.status_label = ctk.CTkLabel(self.status_bar, text="Gotowe", anchor="w")
        self.status_label.grid(row=0, column=0, padx=14, sticky="ew")
        self.status_right_label = ctk.CTkLabel(self.status_bar, text="", anchor="e", text_color="gray75")
        self.status_right_label.grid(row=0, column=1, padx=14, sticky="e")

    def create_view(self, name: str) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.content_area, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_remove()
        frame.grid_columnconfigure(0, weight=1)
        self.views[name] = frame
        return frame

    def create_card(self, parent, title: str, value: str = "", row: int = 0, column: int = 0):
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.grid(row=row, column=column, padx=6, pady=6, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        title_label = ctk.CTkLabel(card, text=title, text_color="gray75", anchor="w", justify="left")
        title_label.grid(row=0, column=0, padx=14, pady=(12, 2), sticky="ew")
        value_label = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=17, weight="bold"), anchor="w", justify="left")
        value_label.grid(row=1, column=0, padx=14, pady=(0, 2), sticky="ew")
        sub_label = ctk.CTkLabel(card, text="", text_color="gray70", anchor="w", justify="left")
        sub_label.grid(row=2, column=0, padx=14, pady=(0, 12), sticky="ew")
        card._responsive_labels = (title_label, value_label, sub_label)
        return card, value_label, sub_label

    def build_dictation_view(self):
        frame = self.create_view("dictation")
        frame.grid_rowconfigure(4, weight=1)
        ctk.CTkLabel(frame, text="Dyktowanie", font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.dashboard_cards_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.dashboard_cards_frame.grid(row=1, column=0, sticky="ew")
        status_card, self.status_card_value, self.status_card_subvalue = self.create_card(self.dashboard_cards_frame, "Status", "Gotowe", 0, 0)
        model_card, self.model_card_value, self.model_card_subvalue = self.create_card(self.dashboard_cards_frame, "Model", "Brak", 0, 1)
        autopaste_card, self.autopaste_card_value, self.autopaste_card_subvalue = self.create_card(self.dashboard_cards_frame, "Auto-paste", "ON" if self.auto_paste_enabled else "OFF", 0, 2)
        llm_card, self.llm_card_value, self.llm_card_subvalue = self.create_card(self.dashboard_cards_frame, "Model LLM", "OFF", 0, 3)
        self.dashboard_cards = [status_card, model_card, autopaste_card, llm_card]
        self.dashboard_cards_frame.bind("<Configure>", self.layout_dashboard_cards)
        self.layout_dashboard_cards()

        action = ctk.CTkFrame(frame)
        action.grid(row=2, column=0, sticky="ew", pady=(10, 12))
        action.grid_columnconfigure(0, weight=1)
        self.record_button = ctk.CTkButton(action, text="🎙 Start dyktowania", height=58, font=ctk.CTkFont(size=20, weight="bold"), command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=14, pady=(14, 10), sticky="ew")
        buttons = ctk.CTkFrame(action, fg_color="transparent")
        buttons.grid(row=1, column=0, padx=8, pady=(0, 12), sticky="ew")
        for col in range(5):
            buttons.grid_columnconfigure(col, weight=1)
        ctk.CTkButton(buttons, text="Kopiuj transkrypcję", command=lambda: self.copy_last_text("original")).grid(row=0, column=0, padx=6, sticky="ew")
        ctk.CTkButton(buttons, text="Kopiuj prompt", command=lambda: self.copy_last_text("processed")).grid(row=0, column=1, padx=6, sticky="ew")
        ctk.CTkButton(buttons, text="Generuj prompt", command=self.generate_prompt_from_last_transcription).grid(row=0, column=2, padx=6, sticky="ew")
        ctk.CTkButton(buttons, text="Wklej wybraną wersję", command=self.paste_selected_preview_text).grid(row=0, column=3, padx=6, sticky="ew")
        ctk.CTkButton(buttons, text="Wyczyść podgląd", command=self.clear_output_preview).grid(row=0, column=4, padx=6, sticky="ew")

        self.recording_widget = ctk.CTkFrame(frame, corner_radius=12)
        self.recording_widget.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        self.recording_widget.grid_columnconfigure(1, weight=1)
        self.recording_dot_label = ctk.CTkLabel(self.recording_widget, text="●", text_color="#ef4444", font=ctk.CTkFont(size=28, weight="bold"))
        self.recording_dot_label.grid(row=0, column=0, padx=(16, 8), pady=(12, 0), sticky="w")
        self.recording_status_label = ctk.CTkLabel(self.recording_widget, text="Nagrywanie...", font=ctk.CTkFont(size=18, weight="bold"), anchor="w")
        self.recording_status_label.grid(row=0, column=1, padx=8, pady=(12, 0), sticky="ew")
        self.recording_stop_button = ctk.CTkButton(self.recording_widget, text="Stop", width=100, command=self.stop_recording)
        self.recording_stop_button.grid(row=0, column=2, padx=16, pady=(12, 0), sticky="e")
        self.recording_timer_label = ctk.CTkLabel(self.recording_widget, text="00:00 / 02:00", text_color="gray80", anchor="w")
        self.recording_timer_label.grid(row=1, column=1, padx=8, pady=(2, 0), sticky="w")
        self.recording_remaining_label = ctk.CTkLabel(self.recording_widget, text="Pozostało: 02:00", text_color="gray70", anchor="e")
        self.recording_remaining_label.grid(row=1, column=2, padx=16, pady=(2, 0), sticky="e")
        self.recording_progress_bar = ctk.CTkProgressBar(self.recording_widget)
        self.recording_progress_bar.grid(row=2, column=0, columnspan=3, padx=16, pady=(8, 10), sticky="ew")
        self.recording_progress_bar.set(0)
        self.waveform_canvas = tk.Canvas(self.recording_widget, height=80, bg="#1f2937", highlightthickness=0)
        self.waveform_canvas.grid(row=3, column=0, columnspan=3, padx=16, pady=(0, 14), sticky="ew")
        self.recording_widget.grid_remove()

        preview = ctk.CTkFrame(frame)
        preview.grid(row=4, column=0, sticky="nsew")
        preview.grid_rowconfigure(1, weight=1)
        preview.grid_columnconfigure(0, weight=1)
        preview_header = ctk.CTkFrame(preview, fg_color="transparent")
        preview_header.grid(row=0, column=0, padx=14, pady=(12, 6), sticky="ew")
        preview_header.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(preview_header, text="Ostatni wynik", font=ctk.CTkFont(size=15, weight="bold")).grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.preview_original_button = ctk.CTkButton(preview_header, text="Transkrypcja", width=120, command=lambda: self.set_preview_mode("original"))
        self.preview_original_button.grid(row=0, column=1, padx=4, sticky="w")
        self.preview_processed_button = ctk.CTkButton(preview_header, text="Prompt", width=90, command=lambda: self.set_preview_mode("processed"))
        self.preview_processed_button.grid(row=0, column=2, padx=4, sticky="w")
        self.transcription_display = ctk.CTkTextbox(preview, height=220, wrap="word")
        self.transcription_display.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self.set_transcription_text("Tutaj pojawi się wynik transkrypcji...")

        quick = ctk.CTkFrame(frame)
        quick.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        for col in range(4):
            quick.grid_columnconfigure(col, weight=1)
        self.auto_paste_var = tk.BooleanVar(value=self.auto_paste_enabled)
        self.copy_to_clipboard_var = tk.BooleanVar(value=self.copy_to_clipboard_enabled)
        self.append_space_var = tk.BooleanVar(value=self.append_space_after_paste)
        self.quick_prompt_processing_default_var = tk.BooleanVar(value=self.prompt_processing_default_for_recordings)
        ctk.CTkSwitch(quick, text="Auto-paste do aktywnego inputu", variable=self.auto_paste_var, command=self.on_quick_settings_change).grid(row=0, column=0, padx=14, pady=14, sticky="w")
        ctk.CTkSwitch(quick, text="Kopiuj do schowka", variable=self.copy_to_clipboard_var, command=self.on_quick_settings_change).grid(row=0, column=1, padx=14, pady=14, sticky="w")
        ctk.CTkSwitch(quick, text="Dodaj spację po tekście", variable=self.append_space_var, command=self.on_quick_settings_change).grid(row=0, column=2, padx=14, pady=14, sticky="w")
        ctk.CTkSwitch(quick, text="Uporządkuj LLM", variable=self.quick_prompt_processing_default_var, command=self.on_quick_settings_change).grid(row=0, column=3, padx=14, pady=14, sticky="w")
        self.update_preview_buttons()

    def build_history_view(self):
        frame = self.create_view("history")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=38)
        frame.grid_columnconfigure(1, weight=62)
        ctk.CTkLabel(frame, text="Historia", font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self.history_list_frame = ctk.CTkScrollableFrame(frame, width=320)
        self.history_list_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.history_detail_frame = ctk.CTkFrame(frame)
        self.history_detail_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.history_detail_frame.grid_rowconfigure(2, weight=1)
        self.history_detail_frame.grid_columnconfigure(0, weight=1)
        self.history_metadata_label = ctk.CTkLabel(self.history_detail_frame, text="Wybierz wpis z historii.", justify="left", anchor="w")
        self.history_metadata_label.grid(row=0, column=0, padx=14, pady=(14, 8), sticky="ew")
        history_tabs = ctk.CTkFrame(self.history_detail_frame, fg_color="transparent")
        history_tabs.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="ew")
        self.history_original_button = ctk.CTkButton(history_tabs, text="Transkrypcja", width=120, command=lambda: self.set_history_view_mode("original"))
        self.history_original_button.pack(side="left", padx=(0, 8))
        self.history_processed_button = ctk.CTkButton(history_tabs, text="Prompt", width=90, command=lambda: self.set_history_view_mode("processed"))
        self.history_processed_button.pack(side="left")
        self.history_display = ctk.CTkTextbox(self.history_detail_frame, wrap="word")
        self.history_display.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self.set_textbox_text(self.history_display, "Brak transkrypcji.", disabled=True)

        actions = ctk.CTkFrame(frame, fg_color="transparent")
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ctk.CTkButton(actions, text="Kopiuj transkrypcję", command=lambda: self.copy_selected_history("original")).pack(side="left", padx=(0, 8))
        ctk.CTkButton(actions, text="Kopiuj prompt", command=lambda: self.copy_selected_history("processed")).pack(side="left", padx=8)
        ctk.CTkButton(actions, text="Przetwórz ponownie", command=self.reprocess_selected_history).pack(side="left", padx=8)
        ctk.CTkButton(actions, text="Usuń zaznaczone", command=self.delete_selected_history).pack(side="left", padx=8)
        ctk.CTkButton(actions, text="Wyczyść historię", command=self.clear_history).pack(side="left", padx=8)

    def build_models_view(self):
        frame = self.create_view("models")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(frame, text="Modele", font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))

        content = ctk.CTkScrollableFrame(frame)
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        self.ensure_prompt_processing_vars()

        selector = ctk.CTkFrame(content)
        selector.grid(row=1, column=0, padx=(0, 8), pady=6, sticky="nsew")
        selector.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(selector, text="Model Whisper", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=14, pady=(14, 8), sticky="w")
        self.model_option_menu = ctk.CTkOptionMenu(selector, values=[model["id"] for model in MODEL_OPTIONS], command=self.on_model_selection_change)
        self.model_option_menu.grid(row=1, column=0, padx=14, pady=8, sticky="ew")
        self.model_option_menu.set(self.selected_model_name)
        self.load_model_button = ctk.CTkButton(selector, text="Załaduj model", command=self.load_selected_model)
        self.load_model_button.grid(row=2, column=0, padx=14, pady=8, sticky="ew")
        self.current_model_label = ctk.CTkLabel(selector, text="Aktualnie załadowany model: brak", anchor="w", justify="left")
        self.current_model_label.grid(row=3, column=0, padx=14, pady=(8, 14), sticky="ew")

        info = ctk.CTkFrame(content)
        info.grid(row=1, column=1, padx=(8, 0), pady=6, sticky="nsew")
        info.grid_columnconfigure(0, weight=1)
        self.model_info_label = ctk.CTkLabel(info, text="", justify="left", anchor="w")
        self.model_info_label.grid(row=0, column=0, padx=14, pady=14, sticky="nsew")

        device = ctk.CTkFrame(content)
        device.grid(row=2, column=0, padx=(0, 8), pady=6, sticky="nsew")
        device.grid_columnconfigure(0, weight=1)
        self.device_label = ctk.CTkLabel(device, text="", justify="left", anchor="w")
        self.device_label.grid(row=0, column=0, padx=14, pady=14, sticky="ew")

        recommendation = ctk.CTkFrame(content)
        recommendation.grid(row=2, column=1, padx=(8, 0), pady=6, sticky="nsew")
        recommendation_text = self.model_recommendation_text()
        ctk.CTkLabel(recommendation, text=recommendation_text, justify="left", anchor="w").grid(row=0, column=0, padx=14, pady=14, sticky="ew")

        self.build_prompt_processing_settings_card(content, 3, 0, columnspan=2)

        llm = ctk.CTkFrame(content)
        llm.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky="ew")
        llm.grid_columnconfigure(0, weight=1)
        llm_text = "\n".join(
            f"- {model['id']} ({model['backend']}): {model['recommendation']}"
            for model in LLM_MODEL_OPTIONS
        )
        ctk.CTkLabel(llm, text="Presety modeli LLM", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")
        ctk.CTkLabel(
            llm,
            text=(
                f"Rekomendowany domyślnie: {DEFAULT_PROMPT_PROCESSING_MODEL}\n"
                "Ollama: użyj `ollama pull <model>`. llama-cpp-python wymaga lokalnego pliku .gguf.\n"
                "Wybór aktywnego modelu znajduje się w sekcji Lokalny LLM powyżej.\n\n"
                f"{llm_text}"
            ),
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")

    def build_settings_view(self):
        frame = self.create_view("settings")
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="Ustawienia", font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))
        content = ctk.CTkScrollableFrame(frame)
        content.grid(row=1, column=0, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)

        self.language_var = tk.StringVar(value=self.language)
        self.num_beams_var = tk.StringVar(value=str(self.num_beams))
        record_duration_options = [f"{seconds} s" for seconds in RECORD_DURATION_PRESETS] + ["Custom"]
        self.record_duration_preset_var = tk.StringVar(value=self.record_duration_preset if self.record_duration_preset in record_duration_options else f"{self.max_record_duration_seconds} s")
        if self.record_duration_preset_var.get() not in record_duration_options:
            self.record_duration_preset_var.set("Custom")
        self.custom_record_duration_var = tk.StringVar(value=str(self.custom_record_duration_seconds))
        self.auto_stop_recording_var = tk.BooleanVar(value=self.auto_stop_recording_enabled)
        self.auto_transcribe_after_stop_var = tk.BooleanVar(value=self.auto_transcribe_after_stop)
        self.auto_paste_after_transcription_var = tk.BooleanVar(value=self.auto_paste_after_transcription)
        self.show_recording_widget_var = tk.BooleanVar(value=self.show_recording_widget)
        self.show_audio_waveform_var = tk.BooleanVar(value=self.show_audio_waveform)
        self.audio_level_smoothing_var = tk.StringVar(value=str(self.audio_level_smoothing))
        self.input_device_var = tk.StringVar(value=self.selected_input_device_label())
        self.dictation_mode_var = tk.BooleanVar(value=self.dictation_mode_enabled)
        self.press_enter_var = tk.BooleanVar(value=self.press_enter_after_paste)
        self.append_newline_var = tk.BooleanVar(value=self.append_newline_after_paste)
        self.global_hotkey_var = tk.BooleanVar(value=self.global_hotkey_enabled)
        self.floating_window_var = tk.BooleanVar(value=self.floating_window_enabled)
        self.trim_text_var = tk.BooleanVar(value=self.trim_text_before_paste)
        self.capitalize_first_var = tk.BooleanVar(value=self.capitalize_first_letter)
        self.restore_window_var = tk.BooleanVar(value=self.restore_previous_window_before_paste)
        self.show_wayland_warning_var = tk.BooleanVar(value=self.show_wayland_warning)
        self.paste_delay_var = tk.StringVar(value=str(self.paste_delay_ms))
        self.paste_method_var = tk.StringVar(value=self.paste_method_preference)
        self.hotkey_var = tk.StringVar(value=self.global_hotkey)
        self.appearance_mode_var = tk.StringVar(value=self.settings.get("appearance_mode", "dark"))
        self.ui_scaling_var = tk.StringVar(value=self.settings.get("ui_scaling", "100%"))

        self.build_recording_settings_card(content, 0, 0)
        self.build_transcription_settings_card(content, 0, 1)
        self.build_paste_settings_card(content, 1, 0)
        self.build_hotkey_settings_card(content, 1, 1)
        self.build_appearance_settings_card(content, 2, 0)
        ctk.CTkButton(content, text="Zapisz ustawienia", height=42, command=self.save_settings_from_ui).grid(row=3, column=0, columnspan=2, padx=8, pady=14, sticky="ew")

    def build_recording_settings_card(self, parent, row: int, column: int):
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=column, padx=8, pady=8, sticky="nsew")
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="Nagrywanie", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w")
        ctk.CTkLabel(card, text="Limit nagrywania").grid(row=1, column=0, padx=14, pady=8, sticky="w")
        self.record_duration_preset_menu = ctk.CTkOptionMenu(
            card,
            values=[f"{seconds} s" for seconds in RECORD_DURATION_PRESETS] + ["Custom"],
            variable=self.record_duration_preset_var,
            command=self.on_record_duration_preset_change,
        )
        self.record_duration_preset_menu.grid(row=1, column=1, padx=14, pady=8, sticky="ew")
        self.custom_record_duration_label = ctk.CTkLabel(card, text="Własny limit (s)")
        self.custom_record_duration_label.grid(row=2, column=0, padx=14, pady=8, sticky="w")
        self.custom_record_duration_entry = ctk.CTkEntry(card, textvariable=self.custom_record_duration_var)
        self.custom_record_duration_entry.grid(row=2, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkSwitch(card, text="Automatycznie zatrzymaj po limicie", variable=self.auto_stop_recording_var).grid(row=3, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkSwitch(card, text="Automatycznie transkrybuj po zatrzymaniu", variable=self.auto_transcribe_after_stop_var).grid(row=4, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkSwitch(card, text="Pokazuj widget nagrywania", variable=self.show_recording_widget_var).grid(row=5, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkSwitch(card, text="Pokazuj falę głosu", variable=self.show_audio_waveform_var).grid(row=6, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkLabel(card, text="Wygładzanie poziomu").grid(row=7, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkEntry(card, textvariable=self.audio_level_smoothing_var).grid(row=7, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Mikrofon").grid(row=8, column=0, padx=14, pady=8, sticky="w")
        input_values = self.input_device_labels()
        ctk.CTkOptionMenu(card, values=input_values or ["System default"], variable=self.input_device_var).grid(row=8, column=1, padx=14, pady=(8, 14), sticky="ew")
        self.on_record_duration_preset_change(self.record_duration_preset_var.get())

    def build_transcription_settings_card(self, parent, row: int, column: int):
        self.ensure_prompt_processing_vars()
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=column, padx=8, pady=8, sticky="nsew")
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="Transkrypcja", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w")
        ctk.CTkLabel(card, text="Język").grid(row=1, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkOptionMenu(card, values=["polish", "english", "auto"], variable=self.language_var).grid(row=1, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Num beams").grid(row=2, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkOptionMenu(card, values=["1", "3", "5"], variable=self.num_beams_var).grid(row=2, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkSwitch(card, text="Domyślnie generuj prompt LLM", variable=self.prompt_processing_default_var).grid(row=3, column=0, columnspan=2, padx=14, pady=(8, 14), sticky="w")

    def build_paste_settings_card(self, parent, row: int, column: int):
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=column, padx=8, pady=8, sticky="nsew")
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="Schowek i auto-paste", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w")
        ctk.CTkSwitch(card, text="Auto-paste", variable=self.auto_paste_var).grid(row=1, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkSwitch(card, text="Kopiuj do schowka", variable=self.copy_to_clipboard_var).grid(row=2, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkSwitch(card, text="Auto-paste po transkrypcji", variable=self.auto_paste_after_transcription_var).grid(row=3, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkSwitch(card, text="Dodaj spację", variable=self.append_space_var).grid(row=4, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkSwitch(card, text="Naciśnij Enter po paste", variable=self.press_enter_var).grid(row=5, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkLabel(card, text="Opóźnienie paste (ms)").grid(row=6, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkEntry(card, textvariable=self.paste_delay_var).grid(row=6, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Metoda paste").grid(row=7, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkOptionMenu(card, values=["auto", "xdotool", "pyautogui"], variable=self.paste_method_var).grid(row=7, column=1, padx=14, pady=(8, 14), sticky="ew")

    def build_hotkey_settings_card(self, parent, row: int, column: int):
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=column, padx=8, pady=8, sticky="nsew")
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="Hotkey", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w")
        ctk.CTkSwitch(card, text="Global hotkey", variable=self.global_hotkey_var).grid(row=1, column=0, columnspan=2, padx=14, pady=8, sticky="w")
        ctk.CTkLabel(card, text="Skrót").grid(row=2, column=0, padx=14, pady=8, sticky="w")
        self.hotkey_entry = ctk.CTkEntry(card, textvariable=self.hotkey_var)
        self.hotkey_entry.grid(row=2, column=1, padx=14, pady=8, sticky="ew")
        self.hotkey_status_label = ctk.CTkLabel(card, text=self.hotkey_status_text, text_color="gray75", anchor="w")
        self.hotkey_status_label.grid(row=3, column=0, columnspan=2, padx=14, pady=(8, 14), sticky="ew")

    def build_appearance_settings_card(self, parent, row: int, column: int):
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=column, padx=8, pady=8, sticky="nsew")
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="Wygląd i sesja", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w")
        ctk.CTkLabel(card, text="Motyw").grid(row=1, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkOptionMenu(card, values=["dark", "light", "system"], variable=self.appearance_mode_var).grid(row=1, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Skalowanie UI").grid(row=2, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkOptionMenu(card, values=["90%", "100%", "110%", "125%"], variable=self.ui_scaling_var).grid(row=2, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkSwitch(card, text="Przywróć poprzednie okno przed paste", variable=self.restore_window_var).grid(row=3, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkSwitch(card, text="Pokaż ostrzeżenie Wayland", variable=self.show_wayland_warning_var).grid(row=4, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        self.session_status_label = ctk.CTkLabel(card, text="", text_color="gray75", justify="left", anchor="w")
        self.session_status_label.grid(row=5, column=0, columnspan=2, padx=14, pady=(8, 14), sticky="ew")

    def ensure_prompt_processing_vars(self):
        if hasattr(self, "prompt_processing_model_var"):
            return
        self.prompt_processing_enabled_var = tk.BooleanVar(value=self.prompt_processing_enabled)
        self.prompt_processing_default_var = tk.BooleanVar(value=self.prompt_processing_default_for_recordings)
        self.prompt_processing_backend_var = tk.StringVar(value=self.prompt_processing_backend)
        self.prompt_processing_model_var = tk.StringVar(value=self.prompt_processing_model)
        self.prompt_processing_ollama_url_var = tk.StringVar(value=self.prompt_processing_ollama_url)
        self.prompt_processing_gguf_path_var = tk.StringVar(value=self.prompt_processing_gguf_path)
        self.prompt_processing_temperature_var = tk.StringVar(value=str(self.prompt_processing_temperature))
        self.prompt_processing_max_tokens_var = tk.StringVar(value=str(self.prompt_processing_max_tokens))
        self.paste_output_preference_var = tk.StringVar(value=self.paste_output_preference)
        self.fallback_to_original_on_llm_error_var = tk.BooleanVar(value=self.fallback_to_original_on_llm_error)

    def build_prompt_processing_settings_card(self, parent, row: int, column: int, columnspan: int = 1):
        self.ensure_prompt_processing_vars()
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=column, columnspan=columnspan, padx=8, pady=8, sticky="nsew")
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="Lokalny LLM", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w")
        ctk.CTkSwitch(card, text="Włącz funkcję", variable=self.prompt_processing_enabled_var).grid(row=1, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkSwitch(card, text="Domyślnie porządkuj nagrania", variable=self.prompt_processing_default_var).grid(row=2, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkLabel(card, text="Backend").grid(row=3, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkOptionMenu(card, values=["ollama", "llama-cpp-python"], variable=self.prompt_processing_backend_var).grid(row=3, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Model").grid(row=4, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkOptionMenu(card, values=[model["id"] for model in LLM_MODEL_OPTIONS], variable=self.prompt_processing_model_var).grid(row=4, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Ollama URL").grid(row=5, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkEntry(card, textvariable=self.prompt_processing_ollama_url_var).grid(row=5, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Ścieżka .gguf").grid(row=6, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkEntry(card, textvariable=self.prompt_processing_gguf_path_var).grid(row=6, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Temperatura").grid(row=7, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkEntry(card, textvariable=self.prompt_processing_temperature_var).grid(row=7, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Max tokenów").grid(row=8, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkEntry(card, textvariable=self.prompt_processing_max_tokens_var).grid(row=8, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkLabel(card, text="Wklejaj").grid(row=9, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkOptionMenu(card, values=["original", "processed"], variable=self.paste_output_preference_var).grid(row=9, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkSwitch(card, text="Fallback do oryginału przy błędzie LLM", variable=self.fallback_to_original_on_llm_error_var).grid(row=10, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkLabel(card, text="Prompt systemowy").grid(row=11, column=0, padx=14, pady=8, sticky="w")
        self.prompt_processing_system_prompt_textbox = ctk.CTkTextbox(card, height=132, wrap="word")
        self.prompt_processing_system_prompt_textbox.grid(row=12, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")
        self.prompt_processing_system_prompt_textbox.insert("end", self.prompt_processing_system_prompt)
        llm_buttons = ctk.CTkFrame(card, fg_color="transparent")
        llm_buttons.grid(row=13, column=0, columnspan=2, padx=8, pady=(0, 14), sticky="ew")
        llm_buttons.grid_columnconfigure(0, weight=1)
        llm_buttons.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(llm_buttons, text="Zapisz ustawienia LLM", height=38, command=self.save_prompt_processing_settings_from_ui).grid(row=0, column=0, padx=6, sticky="ew")
        ctk.CTkButton(llm_buttons, text="Sprawdź połączenie LLM", height=38, command=self.check_llm_connection_from_ui).grid(row=0, column=1, padx=6, sticky="ew")

    def build_diagnostics_view(self):
        frame = self.create_view("diagnostics")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(frame, text="Diagnostyka", font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self.environment_label = ctk.CTkLabel(ctk.CTkFrame(frame), text="")
        env_card = self.environment_label.master
        env_card.grid(row=1, column=0, padx=(0, 8), pady=6, sticky="nsew")
        env_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(env_card, text="Środowisko", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=14, pady=(14, 8), sticky="w")
        self.environment_label.configure(justify="left", anchor="w")
        self.environment_label.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")

        self.dependencies_label = ctk.CTkLabel(ctk.CTkFrame(frame), text="")
        dep_card = self.dependencies_label.master
        dep_card.grid(row=1, column=1, padx=(8, 0), pady=6, sticky="nsew")
        dep_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(dep_card, text="Zależności", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=14, pady=(14, 8), sticky="w")
        self.dependencies_label.configure(justify="left", anchor="w")
        self.dependencies_label.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")

        clip = ctk.CTkFrame(frame)
        clip.grid(row=2, column=0, padx=(0, 8), pady=6, sticky="nsew")
        clip.grid_rowconfigure(2, weight=1)
        clip.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(clip, text="Schowek", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=14, pady=(14, 8), sticky="w")
        clip_buttons = ctk.CTkFrame(clip, fg_color="transparent")
        clip_buttons.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="ew")
        ctk.CTkButton(clip_buttons, text="Check clipboard", command=self.check_clipboard).pack(side="left", padx=6)
        ctk.CTkButton(clip_buttons, text="Test copy text/plain", command=self.test_copy_text_plain).pack(side="left", padx=6)
        ctk.CTkButton(clip_buttons, text="Test auto-paste", command=self.test_auto_paste).pack(side="left", padx=6)
        self.diagnostics_output = ctk.CTkTextbox(clip, height=180, wrap="word")
        self.diagnostics_output.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="nsew")

        logs = ctk.CTkFrame(frame)
        logs.grid(row=2, column=1, padx=(8, 0), pady=6, sticky="nsew")
        logs.grid_rowconfigure(2, weight=1)
        logs.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(logs, text="Logi", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=14, pady=(14, 8), sticky="w")
        log_buttons = ctk.CTkFrame(logs, fg_color="transparent")
        log_buttons.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="ew")
        ctk.CTkButton(log_buttons, text="Odśwież ostatnie 100 linii", command=self.refresh_logs).pack(side="left", padx=6)
        self.logs_output = ctk.CTkTextbox(logs, wrap="word")
        self.logs_output.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="nsew")

    def show_view(self, view_name: str):
        for name, frame in self.views.items():
            if name == view_name:
                frame.grid()
            else:
                frame.grid_remove()
        for name, button in self.nav_buttons.items():
            button.configure(fg_color=("#3B8ED0", "#1F6AA5") if name == view_name else "transparent")
        if view_name == "diagnostics":
            self.update_diagnostics()
            self.refresh_logs()

    def set_textbox_text(self, textbox, text: str, disabled: bool = True):
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("end", text)
        if disabled:
            textbox.configure(state="disabled")

    def set_transcription_text(self, text: str):
        self.set_textbox_text(self.transcription_display, text, disabled=True)

    def layout_dashboard_cards(self, _event=None):
        if not hasattr(self, "dashboard_cards_frame") or not hasattr(self, "dashboard_cards"):
            return
        width = max(1, int(self.dashboard_cards_frame.winfo_width() or 0))
        columns = 2 if width < 920 else 4
        for column in range(4):
            self.dashboard_cards_frame.grid_columnconfigure(column, weight=1 if column < columns else 0, uniform="dashboard_cards" if column < columns else "")
        for index, card in enumerate(self.dashboard_cards):
            card.grid(row=index // columns, column=index % columns, padx=6, pady=6, sticky="nsew")
            card_width = max(160, int(card.winfo_width() or (width / columns)) - 28)
            for label in getattr(card, "_responsive_labels", ()):
                label.configure(wraplength=card_width)

    def update_preview_buttons(self):
        if not hasattr(self, "preview_original_button"):
            return
        self.preview_original_button.configure(fg_color=("#3B8ED0", "#1F6AA5") if self.preview_mode == "original" else "transparent")
        self.preview_processed_button.configure(fg_color=("#3B8ED0", "#1F6AA5") if self.preview_mode == "processed" else "transparent")

    def set_preview_mode(self, mode: str):
        self.preview_mode = "processed" if mode == "processed" else "original"
        text = self.last_processed_text if self.preview_mode == "processed" else self.last_transcription_text
        if not text:
            text = "Prompt LLM nie został wygenerowany." if self.preview_mode == "processed" else "Tutaj pojawi się wynik transkrypcji..."
        self.set_transcription_text(text)
        self.update_preview_buttons()

    def current_preview_text(self) -> str:
        return self.last_processed_text if self.preview_mode == "processed" else self.last_transcription_text

    def set_status(self, status: str):
        logging.info(f"Status: {status}")
        if hasattr(self, "status_label"):
            self.status_label.configure(text=status)
        self.update_status_cards(status)
        self.update_floating_button_state()

    def set_app_state(self, state: str):
        self.app_state = state
        status_by_state = {
            "idle": "Gotowe",
            "recording": "Nagrywanie...",
            "saving": "Zapisywanie audio...",
            "transcribing": "Transkrypcja...",
            "processing_llm": "Porządkowanie promptu LLM...",
            "loading_model": "Ładowanie modelu...",
            "error": "Błąd",
        }
        self.set_status(status_by_state.get(state, state))
        self.set_recording_widget_state(state)
        self.update_control_states()

    def show_recording_widget_frame(self) -> None:
        if self.show_recording_widget and hasattr(self, "recording_widget"):
            self.recording_widget.grid()

    def hide_recording_widget(self) -> None:
        if hasattr(self, "recording_widget"):
            self.recording_widget.grid_remove()

    def ensure_recording_overlay(self) -> None:
        if self.recording_overlay_window is not None and self.recording_overlay_window.winfo_exists():
            return
        overlay = ctk.CTkToplevel(self.master)
        overlay.title("Azor nagrywa")
        overlay.geometry(f"{RECORDING_OVERLAY_WIDTH}x{RECORDING_OVERLAY_HEIGHT}")
        overlay.resizable(False, False)
        overlay.attributes("-topmost", True)
        overlay.overrideredirect(True)
        overlay.configure(fg_color=RECORDING_OVERLAY_TRANSPARENT_COLOR)
        try:
            overlay.tk.call(overlay._w, "configure", "-background", RECORDING_OVERLAY_TRANSPARENT_COLOR)
            overlay.attributes("-transparentcolor", RECORDING_OVERLAY_TRANSPARENT_COLOR)
            overlay.wm_attributes("-transparentcolor", RECORDING_OVERLAY_TRANSPARENT_COLOR)
        except tk.TclError:
            logging.debug("Transparent overlay background is not supported by this Tk/window manager.")
        self.recording_overlay_canvas = tk.Canvas(
            overlay,
            width=RECORDING_OVERLAY_WIDTH,
            height=RECORDING_OVERLAY_HEIGHT,
            bg=RECORDING_OVERLAY_TRANSPARENT_COLOR,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self.recording_overlay_canvas.pack(fill="both", expand=True)
        self.recording_overlay_canvas.bind("<ButtonPress-1>", self.start_recording_overlay_drag)
        self.recording_overlay_canvas.bind("<B1-Motion>", self.drag_recording_overlay)
        self.recording_overlay_canvas.tag_bind("stop_button", "<Button-1>", lambda _event: self.stop_recording())
        self.recording_overlay_canvas.tag_bind("stop_button", "<Enter>", lambda _event: self.recording_overlay_canvas.configure(cursor="hand2"))
        self.recording_overlay_canvas.tag_bind("stop_button", "<Leave>", lambda _event: self.recording_overlay_canvas.configure(cursor=""))
        self.recording_overlay_canvas.tag_bind("llm_toggle", "<Button-1>", lambda _event: self.toggle_current_record_prompt_processing())
        self.recording_overlay_canvas.tag_bind("llm_toggle", "<Enter>", lambda _event: self.recording_overlay_canvas.configure(cursor="hand2"))
        self.recording_overlay_canvas.tag_bind("llm_toggle", "<Leave>", lambda _event: self.recording_overlay_canvas.configure(cursor=""))
        self.recording_overlay_window = overlay
        self.draw_recording_overlay_shell()
        self.position_recording_overlay()

    def draw_recording_overlay_shell(self) -> None:
        canvas = self.recording_overlay_canvas
        if canvas is None:
            return
        canvas.delete("overlay_shell")
        self.create_round_rect(canvas, 0, 0, RECORDING_OVERLAY_WIDTH, RECORDING_OVERLAY_HEIGHT, 28, fill="#111827", outline="#2f3b52", width=1, tags=("overlay_shell",))
        self.recording_overlay_dot_item = canvas.create_text(26, 28, text="●", fill="#ef4444", font=("Arial", 22, "bold"), anchor="center", tags=("overlay_shell",))
        self.recording_overlay_title_item = canvas.create_text(52, 28, text="Nagrywanie", fill="#f8fafc", font=("Arial", 16, "bold"), anchor="w", tags=("overlay_shell",))
        self.recording_overlay_timer_item = canvas.create_text(52, 55, text="00:00 / 02:00", fill="#cbd5e1", font=("Arial", 11), anchor="w", tags=("overlay_shell",))
        self.recording_overlay_remaining_item = canvas.create_text(RECORDING_OVERLAY_WIDTH - 96, 55, text="Pozostało: 02:00", fill="#94a3b8", font=("Arial", 11), anchor="center", tags=("overlay_shell",))
        self.recording_overlay_stage_item = canvas.create_text(52, 90, text="", fill="#93c5fd", font=("Arial", 11, "bold"), anchor="w", tags=("overlay_shell",))
        self.create_round_rect(canvas, RECORDING_OVERLAY_WIDTH - 92, 16, RECORDING_OVERLAY_WIDTH - 20, 46, 10, fill="#dc2626", outline="", tags=("overlay_shell", "stop_button"))
        canvas.create_text(RECORDING_OVERLAY_WIDTH - 56, 31, text="Stop", fill="#ffffff", font=("Arial", 11, "bold"), tags=("overlay_shell", "stop_button"))
        self.recording_overlay_llm_item = self.create_round_rect(canvas, RECORDING_OVERLAY_WIDTH - 116, 76, RECORDING_OVERLAY_WIDTH - 20, 104, 10, fill="#2563eb", outline="", tags=("overlay_shell", "llm_toggle"))
        canvas.create_text(RECORDING_OVERLAY_WIDTH - 68, 90, text="LLM ON", fill="#ffffff", font=("Arial", 10, "bold"), tags=("overlay_shell", "llm_toggle", "llm_toggle_text"))
        self.update_recording_overlay_llm_toggle()
        self.configure_recording_overlay_for_state(self.app_state)

    def create_round_rect(self, canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, radius: float, **kwargs) -> int:
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)

    def position_recording_overlay(self) -> None:
        if self.recording_overlay_window is None or not self.recording_overlay_window.winfo_exists():
            return
        self.recording_overlay_window.update_idletasks()
        primary_bounds = get_primary_monitor_bounds()
        if primary_bounds is None:
            primary_bounds = (0, 0, self.recording_overlay_window.winfo_screenwidth(), self.recording_overlay_window.winfo_screenheight())
        screen_x, screen_y, screen_width, _screen_height = primary_bounds
        x = screen_x + max(16, int((screen_width - RECORDING_OVERLAY_WIDTH) / 2))
        y = screen_y + RECORDING_OVERLAY_TOP_MARGIN
        self.recording_overlay_window.geometry(f"{RECORDING_OVERLAY_WIDTH}x{RECORDING_OVERLAY_HEIGHT}+{x}+{y}")

    def start_recording_overlay_drag(self, event) -> None:
        self.recording_overlay_drag_offset = (event.x_root - self.recording_overlay_window.winfo_x(), event.y_root - self.recording_overlay_window.winfo_y())

    def drag_recording_overlay(self, event) -> None:
        if self.recording_overlay_window is None or not self.recording_overlay_window.winfo_exists():
            return
        offset_x, offset_y = self.recording_overlay_drag_offset
        self.recording_overlay_window.geometry(f"+{event.x_root - offset_x}+{event.y_root - offset_y}")

    def show_recording_overlay(self) -> None:
        self.ensure_recording_overlay()
        if self.recording_overlay_window is not None and self.recording_overlay_window.winfo_exists():
            self.recording_overlay_window.deiconify()
            self.recording_overlay_window.lift()
            self.recording_overlay_window.attributes("-topmost", True)

    def hide_recording_overlay(self) -> None:
        if self.recording_overlay_window is not None and self.recording_overlay_window.winfo_exists():
            self.recording_overlay_window.withdraw()
        if self.recording_overlay_canvas is not None:
            self.recording_overlay_canvas.delete("waveform")
        if self.waveform_update_id:
            try:
                self.master.after_cancel(self.waveform_update_id)
            except Exception:
                logging.debug("Processing wave callback was already cleared.", exc_info=True)
            self.waveform_update_id = None

    def set_recording_widget_state(self, state: str) -> None:
        if not hasattr(self, "recording_widget"):
            return
        if state in {"recording", "saving", "transcribing", "processing_llm"}:
            self.show_recording_overlay()
            self.configure_recording_overlay_for_state(state)
        elif state in {"idle", "done", "error"}:
            self.hide_recording_overlay()
        if state in {"recording", "saving", "transcribing", "processing_llm", "done"}:
            self.show_recording_widget_frame()
        elif state == "idle":
            self.hide_recording_widget()
            return
        elif state == "error":
            self.hide_recording_widget()
            return

        labels = {
            "recording": "Nagrywanie...",
            "saving": "Zapisywanie audio...",
            "transcribing": "Transkrypcja...",
            "processing_llm": "Porządkowanie promptu LLM...",
            "done": "Gotowe. Tekst skopiowany/wklejony.",
        }
        self.recording_status_label.configure(text=labels.get(state, state))
        self.recording_dot_label.configure(text="●" if state == "recording" else "")
        self.recording_stop_button.configure(state="normal" if state == "recording" else "disabled")
        if state == "saving":
            self.recording_progress_bar.set(1)
        elif state == "transcribing":
            self.recording_progress_bar.set(1)
        elif state == "processing_llm":
            self.recording_progress_bar.set(1)

    def configure_recording_overlay_for_state(self, state: str) -> None:
        canvas = self.recording_overlay_canvas
        if canvas is None:
            return
        title_by_state = {
            "recording": "Nagrywanie",
            "saving": "Przygotowanie audio",
            "transcribing": "Mowa na tekst",
            "processing_llm": "Porządkowanie LLM",
        }
        subtitle_by_state = {
            "recording": "",
            "saving": "Zapisywanie nagrania...",
            "transcribing": "Przetwarzanie mowy na tekst...",
            "processing_llm": "Tworzenie uporządkowanego promptu...",
        }
        accent = "#ef4444" if state == "recording" else "#38bdf8"
        if self.recording_overlay_title_item is not None:
            canvas.itemconfigure(self.recording_overlay_title_item, text=title_by_state.get(state, state))
        if self.recording_overlay_stage_item is not None:
            canvas.itemconfigure(self.recording_overlay_stage_item, text=subtitle_by_state.get(state, ""), fill="#93c5fd")
        if self.recording_overlay_dot_item is not None:
            canvas.itemconfigure(self.recording_overlay_dot_item, text="●", fill=accent)
        stop_state = "normal" if state == "recording" else "hidden"
        llm_state = "normal" if state == "recording" else "hidden"
        canvas.itemconfigure("stop_button", state=stop_state)
        canvas.itemconfigure("llm_toggle", state=llm_state)
        canvas.itemconfigure("llm_toggle_text", state=llm_state)
        if state == "recording":
            canvas.delete("waveform")
            self.update_recording_overlay_llm_toggle()
        else:
            if self.waveform_update_id:
                try:
                    self.master.after_cancel(self.waveform_update_id)
                except Exception:
                    logging.debug("Processing wave callback was already cleared.", exc_info=True)
                self.waveform_update_id = None
            self.draw_processing_wave(state)

    def draw_processing_wave(self, state: str | None = None) -> None:
        canvas = self.recording_overlay_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        if state is None:
            state = self.app_state
        if self.app_state not in {"saving", "transcribing", "processing_llm"} or state not in {"saving", "transcribing", "processing_llm"}:
            canvas.delete("waveform")
            return
        canvas.delete("waveform")
        left = 28
        right = RECORDING_OVERLAY_WIDTH - 28
        top = 112
        bottom = 162
        center_y = (top + bottom) / 2
        segment_count = 36
        gap = 3
        segment_width = max(4, int((right - left - gap * (segment_count - 1)) / segment_count))
        start_x = left
        phase = self.processing_wave_phase
        for index in range(segment_count):
            wave = (math.sin((index + phase) / 2.2) + 1.0) / 2.0
            pulse = (math.sin((index - phase) / 4.0) + 1.0) / 2.0
            intensity = 0.25 + wave * 0.45 + pulse * 0.30
            bar_height = 8 + intensity * (bottom - top - 8)
            x1 = start_x + index * (segment_width + gap)
            x2 = x1 + segment_width
            y1 = center_y - bar_height / 2
            y2 = center_y + bar_height / 2
            color = "#60a5fa" if state != "processing_llm" else "#22c55e"
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags=("waveform",))
        self.processing_wave_phase = (self.processing_wave_phase + 1) % segment_count
        self.waveform_update_id = self.master.after(120, self.draw_processing_wave)

    def update_recording_overlay_llm_toggle(self) -> None:
        if self.recording_overlay_canvas is None:
            return
        enabled = self.prompt_processing_enabled and self.current_record_prompt_processing_enabled
        fill = "#16a34a" if enabled else "#475569"
        text = "LLM ON" if enabled else "LLM OFF"
        self.recording_overlay_canvas.itemconfigure("llm_toggle", fill=fill)
        self.recording_overlay_canvas.itemconfigure("llm_toggle_text", text=text)
        self.recording_overlay_canvas.itemconfigure("llm_toggle_text", fill="#ffffff")

    def toggle_current_record_prompt_processing(self) -> None:
        if not self.recording:
            return
        self.current_record_prompt_processing_enabled = not self.current_record_prompt_processing_enabled
        self.set_prompt_processing_default_for_recordings(self.current_record_prompt_processing_enabled, persist=True)
        self.update_recording_overlay_llm_toggle()
        self.set_status(f"LLM {'włączony' if self.current_record_prompt_processing_enabled else 'wyłączony'} dla bieżącego nagrania")

    def set_prompt_processing_default_for_recordings(self, enabled: bool, persist: bool = False) -> None:
        self.prompt_processing_default_for_recordings = bool(enabled)
        if self.prompt_processing_default_for_recordings and not self.prompt_processing_enabled:
            self.prompt_processing_enabled = True
            if hasattr(self, "prompt_processing_enabled_var"):
                self.prompt_processing_enabled_var.set(True)
        if hasattr(self, "quick_prompt_processing_default_var"):
            self.quick_prompt_processing_default_var.set(self.prompt_processing_default_for_recordings)
        if hasattr(self, "prompt_processing_default_var"):
            self.prompt_processing_default_var.set(self.prompt_processing_default_for_recordings)
        if persist:
            self.persist_settings()
        self.update_status_cards()

    def can_process_prompt(self) -> tuple[bool, str]:
        if not self.prompt_processing_enabled:
            return False, "Porządkowanie LLM jest wyłączone w ustawieniach."
        if self.prompt_processing_backend == "llama-cpp-python" and not self.prompt_processing_gguf_path.strip():
            return False, "Wskaż plik .gguf dla backendu llama-cpp-python."
        return True, ""

    def update_recording_timer_ui(self) -> None:
        if not self.recording or not self.start_time:
            return
        limit = max(1, self.current_record_limit_seconds)
        elapsed = max(0.0, time.time() - self.start_time)
        remaining = max(0.0, limit - elapsed)
        self.recording_timer_label.configure(text=f"{format_seconds(elapsed)} / {format_seconds(limit)}")
        self.recording_remaining_label.configure(text=f"Pozostało: {format_seconds(remaining)}")
        self.recording_progress_bar.set(min(1.0, elapsed / limit))
        if self.recording_overlay_canvas is not None and self.recording_overlay_timer_item is not None:
            self.recording_overlay_canvas.itemconfigure(self.recording_overlay_timer_item, text=f"{format_seconds(elapsed)} / {format_seconds(limit)}")
        if self.recording_overlay_canvas is not None and self.recording_overlay_remaining_item is not None:
            self.recording_overlay_canvas.itemconfigure(self.recording_overlay_remaining_item, text=f"Pozostało: {format_seconds(remaining)}")
        self.red_dot_pulse = not self.red_dot_pulse
        self.recording_dot_label.configure(text_color="#ef4444" if self.red_dot_pulse else "#7f1d1d")
        if self.recording_overlay_canvas is not None and self.recording_overlay_dot_item is not None:
            self.recording_overlay_canvas.itemconfigure(self.recording_overlay_dot_item, fill="#ef4444" if self.red_dot_pulse else "#7f1d1d")
        self.recording_timer_ui_id = self.master.after(250, self.update_recording_timer_ui)

    def calculate_audio_level(self, data: bytes) -> float:
        try:
            samples = array.array("h")
            samples.frombytes(data)
            if sys.byteorder == "big":
                samples.byteswap()
            if not samples:
                return 0.0
            square_sum = sum(sample * sample for sample in samples)
            rms = math.sqrt(square_sum / len(samples))
            normalized = min(1.0, rms / 3000.0)
            smoothing = max(0.0, min(1.0, float(self.audio_level_smoothing)))
            return self.last_audio_level * (1.0 - smoothing) + normalized * smoothing
        except Exception:
            logging.exception("Could not calculate audio level")
            return 0.0

    def push_audio_level(self, level: float) -> None:
        self.last_audio_level = max(0.0, min(1.0, level))
        self.audio_level_peak = max(self.audio_level_peak, self.last_audio_level)
        self.audio_level_sum += self.last_audio_level
        self.audio_level_count += 1
        self.audio_levels.append(self.last_audio_level)
        if len(self.audio_levels) > 60:
            self.audio_levels = self.audio_levels[-60:]

    def is_probably_silent_recording(self) -> bool:
        if self.audio_level_count <= 0:
            return True
        avg_level = self.audio_level_sum / self.audio_level_count
        logging.info(
            "Recording audio level summary: peak=%.4f avg=%.4f samples=%s",
            self.audio_level_peak,
            avg_level,
            self.audio_level_count,
        )
        return self.audio_level_peak < 0.008 and avg_level < 0.003

    def update_waveform_canvas(self) -> None:
        if not self.recording:
            return
        if not self.show_audio_waveform:
            self.waveform_canvas.delete("all")
            if self.recording_overlay_canvas is not None:
                self.recording_overlay_canvas.delete("waveform")
            return
        self.draw_waveform(self.waveform_canvas, width_fallback=500, height_fallback=80)
        if self.recording_overlay_canvas is not None and self.recording_overlay_canvas.winfo_exists():
            self.draw_waveform(self.recording_overlay_canvas, width_fallback=390, height_fallback=88, compact=True)
        self.waveform_update_id = self.master.after(80, self.update_waveform_canvas)

    def draw_waveform(self, canvas: tk.Canvas, width_fallback: int, height_fallback: int, compact: bool = False) -> None:
        canvas.delete("waveform" if compact else "all")
        width = int(canvas.winfo_width() or width_fallback)
        height = int(canvas.winfo_height() or height_fallback)
        top = 76 if compact else 0
        bottom_padding = 18 if compact else 0
        left_padding = 18 if compact else 0
        right_padding = 18 if compact else 0
        width = max(1, width - left_padding - right_padding)
        drawable_height = height - top - bottom_padding
        center_y = top + drawable_height // 2
        sample_count = 40 if compact else 50
        levels = self.audio_levels[-sample_count:] or [0.0] * sample_count
        if len(levels) < sample_count:
            levels = [0.0] * (sample_count - len(levels)) + levels
        gap = 3 if compact else 3
        bar_count = len(levels)
        bar_width = max(3, (width - gap * (bar_count - 1)) / bar_count)
        for index, level in enumerate(levels):
            x1 = index * (bar_width + gap)
            x2 = x1 + bar_width
            visual_level = min(1.0, level * 1.55) if compact else level
            bar_height = max(8 if compact else 3, visual_level * (drawable_height - 8))
            y1 = center_y - bar_height / 2
            y2 = center_y + bar_height / 2
            if compact:
                color = "#ef4444" if index > bar_count - 8 else "#38bdf8"
                canvas.create_rectangle(x1 + left_padding, y1, x2 + left_padding, y2, fill=color, outline="", width=0, tags=("waveform",))
            else:
                color = "#ef4444" if self.app_state == "recording" and index > bar_count - 8 else "#3b82f6"
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def cancel_recording_ui_timers(self) -> None:
        if self.recording_timer_ui_id:
            try:
                self.master.after_cancel(self.recording_timer_ui_id)
            except Exception:
                logging.debug("Recording timer UI callback was already cleared.", exc_info=True)
            self.recording_timer_ui_id = None
        if self.waveform_update_id:
            try:
                self.master.after_cancel(self.waveform_update_id)
            except Exception:
                logging.debug("Waveform callback was already cleared.", exc_info=True)
            self.waveform_update_id = None
        if self.recording_widget_hide_id:
            try:
                self.master.after_cancel(self.recording_widget_hide_id)
            except Exception:
                logging.debug("Recording widget hide callback was already cleared.", exc_info=True)
            self.recording_widget_hide_id = None

    def update_status_cards(self, status: str | None = None):
        if not hasattr(self, "status_card_value"):
            return
        status = status or getattr(self.status_label, "cget", lambda _key: "Gotowe")("text")
        self.status_card_value.configure(text=status)
        self.status_card_subvalue.configure(text=self.app_state)
        current_model = self.current_model_name or "Nie załadowano"
        self.model_card_value.configure(text=self.short_model_name(current_model))
        self.model_card_subvalue.configure(text=self.device_name)
        self.autopaste_card_value.configure(text="ON" if self.auto_paste_enabled else "OFF")
        last_paste = "brak próby"
        if self.last_paste_result:
            last_paste = f"{self.last_paste_result.method}: {self.last_paste_result.success}"
        self.autopaste_card_subvalue.configure(text=last_paste)
        if hasattr(self, "llm_card_value"):
            llm_mode = "OFF"
            if self.prompt_processing_enabled:
                llm_mode = "ON" if self.prompt_processing_default_for_recordings else "Ręcznie"
            self.llm_card_value.configure(text=llm_mode)
            self.llm_card_subvalue.configure(text=self.llm_dashboard_subvalue())
        self.status_right_label.configure(
            text=(
                f"Hotkey: {self.global_hotkey} | Auto-paste: {'ON' if self.auto_paste_enabled else 'OFF'} | "
                f"Model: {self.short_model_name(current_model)} | LLM: {self.short_model_name(self.prompt_processing_model)}"
            )
        )
        self.sidebar_model_label.configure(text=f"Model: {self.short_model_name(current_model)}")
        self.sidebar_device_label.configure(text=f"Device: {self.device_name}")
        self.sidebar_session_label.configure(text=f"Session: {self.session_type}")

    def llm_dashboard_subvalue(self) -> str:
        if not self.prompt_processing_enabled:
            return f"{self.prompt_processing_backend} | wyłączony"
        if self.llm_connection_check_running:
            connection = "sprawdzanie..."
        elif self.llm_connection_status is None:
            connection = self.llm_connection_summary
        elif self.llm_connection_status.connected and self.llm_connection_status.model_available:
            connection = "połączony"
        elif self.llm_connection_status.connected:
            connection = "backend OK, brak modelu"
        else:
            connection = "brak połączenia"
        mode = "przetwarza nagrania" if self.prompt_processing_default_for_recordings else "dostępny ręcznie"
        return f"{self.prompt_processing_backend} | {self.short_model_name(self.prompt_processing_model)}\n{connection} | {mode}"

    def check_llm_connection_async(self):
        if self.llm_connection_check_running:
            return
        self.llm_connection_check_running = True
        self.llm_connection_summary = "Sprawdzanie..."
        self.update_status_cards()
        threading.Thread(target=self.check_llm_connection_worker, daemon=True).start()

    def check_llm_connection_worker(self):
        status = check_llm_connection(self.prompt_processor_settings())
        self.transcription_queue.put({"type": "llm_connection_status", "status": status})

    def apply_llm_connection_status(self, status):
        self.llm_connection_check_running = False
        self.llm_connection_status = status
        if status.connected and status.model_available:
            self.llm_connection_summary = "Połączony"
        elif status.connected:
            self.llm_connection_summary = "Backend OK, brak modelu"
        else:
            self.llm_connection_summary = "Brak połączenia"
        logging.info(
            "LLM connection status: backend=%s model=%s connected=%s model_available=%s message=%s",
            status.backend,
            status.model,
            status.connected,
            status.model_available,
            status.message,
        )
        self.update_status_cards()
        self.update_diagnostics()

    def short_model_name(self, model_name: str) -> str:
        return model_name.split("/")[-1] if model_name else "brak"

    def clear_output_preview(self):
        self.last_transcription_text = ""
        self.last_processed_text = ""
        self.last_prompt_processing_error = None
        self.preview_mode = "original"
        self.set_transcription_text("Tutaj pojawi się wynik transkrypcji...")
        self.update_preview_buttons()
        self.set_status("Gotowe")

    def copy_last_text(self, mode: str = "original"):
        text = self.last_processed_text if mode == "processed" else self.last_transcription_text
        text = text.strip()
        if not text:
            self.set_status("Brak promptu do skopiowania" if mode == "processed" else "Brak tekstu do skopiowania")
            return
        self.copy_to_clipboard(text)

    def paste_selected_preview_text(self):
        text = self.current_preview_text().strip()
        if not text:
            self.set_status("Brak wybranej wersji do wklejenia")
            return
        self.paste_text_to_active_input(prepare_text_for_paste(
            text,
            trim_text=self.trim_text_before_paste,
            capitalize_first_letter=self.capitalize_first_letter,
            append_space=self.append_space_after_paste,
            append_newline=self.append_newline_after_paste,
        ))

    def prompt_processor_settings(self) -> dict:
        self.settings.update({
            "prompt_processing_backend": self.prompt_processing_backend,
            "prompt_processing_model": self.prompt_processing_model,
            "prompt_processing_ollama_url": self.prompt_processing_ollama_url,
            "prompt_processing_gguf_path": self.prompt_processing_gguf_path,
            "prompt_processing_system_prompt": self.prompt_processing_system_prompt,
            "prompt_processing_temperature": self.prompt_processing_temperature,
            "prompt_processing_max_tokens": self.prompt_processing_max_tokens,
        })
        return dict(self.settings)

    def process_prompt_text(self, text: str):
        processor = PromptProcessor(self.prompt_processor_settings())
        return processor.process(text)

    def reprocess_last_transcription(self):
        text = self.last_transcription_text.strip()
        if not text:
            self.set_status("Brak transkrypcji do przetworzenia")
            return
        can_process, message = self.can_process_prompt()
        if not can_process:
            self.set_status(message)
            return
        self.set_status("Porządkowanie promptu...")
        threading.Thread(target=self.reprocess_last_worker, args=(text,), daemon=True).start()

    def reprocess_last_worker(self, text: str):
        result = self.process_prompt_text(text)
        self.transcription_queue.put({"type": "prompt_reprocessed_last", "result": result})

    def generate_prompt_from_last_transcription(self):
        text = self.last_transcription_text.strip()
        if not text:
            self.set_status("Brak transkrypcji do wygenerowania promptu")
            return
        self.set_status("Sprawdzanie dostępności LLM...")
        threading.Thread(target=self.generate_prompt_from_last_worker, args=(text,), daemon=True).start()

    def generate_prompt_from_last_worker(self, text: str):
        status = check_llm_connection(self.prompt_processor_settings())
        self.transcription_queue.put({"type": "llm_connection_status", "status": status})
        if not status.connected or not status.model_available:
            self.transcription_queue.put({"type": "manual_prompt_error", "message": status.message})
            return
        self.transcription_queue.put({"type": "status", "status": "Porządkowanie promptu..."})
        result = self.process_prompt_text(text)
        self.transcription_queue.put({"type": "prompt_reprocessed_last", "result": result})

    def reprocess_selected_history(self):
        entry = self.selected_history_entry()
        if not entry:
            self.set_status("Wybierz wpis historii")
            return
        can_process, message = self.can_process_prompt()
        if not can_process:
            self.set_status(message)
            return
        entry_index = self.selected_history_index
        self.set_status("Porządkowanie promptu...")
        threading.Thread(target=self.reprocess_history_worker, args=(entry_index, entry.get("text", "")), daemon=True).start()

    def reprocess_history_worker(self, entry_index: int, text: str):
        result = self.process_prompt_text(text)
        self.transcription_queue.put({"type": "prompt_reprocessed_history", "index": entry_index, "result": result})

    def on_quick_settings_change(self):
        self.auto_paste_enabled = self.auto_paste_var.get()
        self.auto_paste_after_transcription = self.auto_paste_enabled
        self.copy_to_clipboard_enabled = self.copy_to_clipboard_var.get()
        self.append_space_after_paste = self.append_space_var.get()
        if hasattr(self, "quick_prompt_processing_default_var"):
            self.set_prompt_processing_default_for_recordings(self.quick_prompt_processing_default_var.get())
        self.persist_settings()
        self.update_dictation_status()

    def on_record_duration_preset_change(self, preset: str | None = None):
        preset = preset or self.record_duration_preset_var.get()
        if not hasattr(self, "custom_record_duration_label"):
            return
        if preset == "Custom":
            self.custom_record_duration_label.grid()
            self.custom_record_duration_entry.grid()
        else:
            self.custom_record_duration_label.grid_remove()
            self.custom_record_duration_entry.grid_remove()

    def get_max_record_duration_seconds(self) -> int:
        preset = getattr(self, "record_duration_preset", "120 s")
        if preset == "Custom":
            seconds = int(getattr(self, "custom_record_duration_seconds", DEFAULT_MAX_RECORD_DURATION))
        else:
            seconds = int(str(preset).replace(" s", ""))
        return max(5, min(MAX_RECORD_DURATION_SECONDS, seconds))

    def set_recording_limit(self, seconds: int) -> None:
        seconds = max(5, min(MAX_RECORD_DURATION_SECONDS, int(seconds)))
        self.max_record_duration_seconds = seconds
        self.max_record_duration = seconds

    def persist_settings(self):
        self.settings.update({
            "selected_model": self.selected_model_name,
            "language": self.language,
            "num_beams": self.num_beams,
            "input_device_index": self.input_device_index,
            "max_record_duration": self.max_record_duration_seconds,
            "max_record_duration_seconds": self.max_record_duration_seconds,
            "record_duration_preset": self.record_duration_preset,
            "custom_record_duration_seconds": self.custom_record_duration_seconds,
            "auto_stop_recording_enabled": self.auto_stop_recording_enabled,
            "auto_transcribe_after_stop": self.auto_transcribe_after_stop,
            "auto_paste_after_transcription": self.auto_paste_after_transcription,
            "show_recording_widget": self.show_recording_widget,
            "show_audio_waveform": self.show_audio_waveform,
            "audio_level_smoothing": self.audio_level_smoothing,
            "dictation_mode_enabled": self.dictation_mode_enabled,
            "auto_paste_enabled": self.auto_paste_enabled,
            "copy_to_clipboard_enabled": self.copy_to_clipboard_enabled,
            "global_hotkey_enabled": self.global_hotkey_enabled,
            "global_hotkey": self.global_hotkey,
            "floating_window_enabled": self.floating_window_enabled,
            "append_space_after_paste": self.append_space_after_paste,
            "append_newline_after_paste": self.append_newline_after_paste,
            "press_enter_after_paste": self.press_enter_after_paste,
            "trim_text_before_paste": self.trim_text_before_paste,
            "capitalize_first_letter": self.capitalize_first_letter,
            "paste_delay_ms": self.paste_delay_ms,
            "restore_previous_window_before_paste": self.restore_previous_window_before_paste,
            "paste_method_preference": self.paste_method_preference,
            "show_wayland_warning": self.show_wayland_warning,
            "appearance_mode": self.appearance_mode_var.get() if hasattr(self, "appearance_mode_var") else self.settings.get("appearance_mode", "dark"),
            "ui_scaling": self.ui_scaling_var.get() if hasattr(self, "ui_scaling_var") else self.settings.get("ui_scaling", "100%"),
            "prompt_processing_enabled": self.prompt_processing_enabled,
            "prompt_processing_default_for_recordings": self.prompt_processing_default_for_recordings,
            "prompt_processing_backend": self.prompt_processing_backend,
            "prompt_processing_model": self.prompt_processing_model,
            "prompt_processing_ollama_url": self.prompt_processing_ollama_url,
            "prompt_processing_gguf_path": self.prompt_processing_gguf_path,
            "prompt_processing_system_prompt": self.prompt_processing_system_prompt,
            "prompt_processing_temperature": self.prompt_processing_temperature,
            "prompt_processing_max_tokens": self.prompt_processing_max_tokens,
            "paste_output_preference": self.paste_output_preference,
            "fallback_to_original_on_llm_error": self.fallback_to_original_on_llm_error,
        })
        save_settings(self.settings)

    def read_prompt_processing_settings_from_ui(self):
        self.ensure_prompt_processing_vars()
        self.prompt_processing_enabled = self.prompt_processing_enabled_var.get()
        self.set_prompt_processing_default_for_recordings(self.prompt_processing_default_var.get())
        self.prompt_processing_backend = self.prompt_processing_backend_var.get()
        self.prompt_processing_model = self.prompt_processing_model_var.get()
        self.prompt_processing_ollama_url = self.prompt_processing_ollama_url_var.get().strip() or "http://localhost:11434"
        self.prompt_processing_gguf_path = self.prompt_processing_gguf_path_var.get().strip()
        if hasattr(self, "prompt_processing_system_prompt_textbox"):
            system_prompt = self.prompt_processing_system_prompt_textbox.get("1.0", "end").strip()
        else:
            system_prompt = self.prompt_processing_system_prompt
        self.prompt_processing_system_prompt = system_prompt or DEFAULT_PROMPT_PROCESSING_SYSTEM_PROMPT
        self.prompt_processing_temperature = max(0.0, min(2.0, float(self.prompt_processing_temperature_var.get())))
        self.prompt_processing_max_tokens = max(64, min(8192, int(self.prompt_processing_max_tokens_var.get())))
        self.paste_output_preference = self.paste_output_preference_var.get()
        self.fallback_to_original_on_llm_error = self.fallback_to_original_on_llm_error_var.get()

    def save_prompt_processing_settings_from_ui(self):
        try:
            self.read_prompt_processing_settings_from_ui()
        except ValueError as e:
            messagebox.showerror("Settings Error", f"Invalid numeric LLM setting: {e}")
            return
        self.persist_settings()
        self.check_llm_connection_async()
        self.update_diagnostics()
        self.set_status("Ustawienia LLM zapisane")

    def check_llm_connection_from_ui(self):
        try:
            self.read_prompt_processing_settings_from_ui()
        except ValueError as e:
            messagebox.showerror("Settings Error", f"Invalid numeric LLM setting: {e}")
            return
        self.persist_settings()
        self.check_llm_connection_async()
        self.set_status("Sprawdzanie połączenia LLM...")

    def save_settings_from_ui(self):
        try:
            self.language = self.language_var.get()
            self.num_beams = int(self.num_beams_var.get())
            self.input_device_index = self.parse_input_device_label(self.input_device_var.get())
            self.record_duration_preset = self.record_duration_preset_var.get()
            self.custom_record_duration_seconds = int(self.custom_record_duration_var.get())
            if self.custom_record_duration_seconds < 5 or self.custom_record_duration_seconds > MAX_RECORD_DURATION_SECONDS:
                raise ValueError(f"Custom recording duration must be between 5 and {MAX_RECORD_DURATION_SECONDS} seconds.")
            if self.record_duration_preset == "Custom":
                self.set_recording_limit(self.custom_record_duration_seconds)
            else:
                self.set_recording_limit(int(self.record_duration_preset.replace(" s", "")))
            self.auto_stop_recording_enabled = self.auto_stop_recording_var.get()
            self.auto_transcribe_after_stop = self.auto_transcribe_after_stop_var.get()
            self.auto_paste_after_transcription = self.auto_paste_after_transcription_var.get()
            self.show_recording_widget = self.show_recording_widget_var.get()
            self.show_audio_waveform = self.show_audio_waveform_var.get()
            self.audio_level_smoothing = max(0.0, min(1.0, float(self.audio_level_smoothing_var.get())))
            self.dictation_mode_enabled = self.dictation_mode_var.get()
            self.auto_paste_enabled = self.auto_paste_var.get()
            self.copy_to_clipboard_enabled = self.copy_to_clipboard_var.get()
            self.append_space_after_paste = self.append_space_var.get()
            self.append_newline_after_paste = self.append_newline_var.get()
            self.press_enter_after_paste = self.press_enter_var.get()
            self.global_hotkey_enabled = self.global_hotkey_var.get()
            self.global_hotkey = self.hotkey_var.get().strip() or DEFAULT_GLOBAL_HOTKEY
            self.floating_window_enabled = self.floating_window_var.get()
            self.trim_text_before_paste = self.trim_text_var.get()
            self.capitalize_first_letter = self.capitalize_first_var.get()
            self.paste_delay_ms = int(self.paste_delay_var.get())
            self.restore_previous_window_before_paste = self.restore_window_var.get()
            self.paste_method_preference = self.paste_method_var.get()
            self.show_wayland_warning = self.show_wayland_warning_var.get()
            self.read_prompt_processing_settings_from_ui()
        except ValueError as e:
            messagebox.showerror("Settings Error", f"Invalid numeric setting: {e}")
            return

        ctk.set_appearance_mode(self.appearance_mode_var.get())
        scaling = int(self.ui_scaling_var.get().replace("%", "")) / 100
        ctk.set_widget_scaling(scaling)
        self.paste_manager = self.create_paste_manager()
        self.persist_settings()
        self.check_llm_connection_async()
        self.restart_hotkey_listener()
        self.update_floating_window()
        self.update_dictation_status()
        self.update_control_states()
        if self.max_record_duration_seconds > 300:
            self.set_status("Długie nagrania mogą transkrybować się wolniej. Do meetingów użyj później trybu transkrypcji pliku.")
            return
        self.set_status("Ustawienia zapisane")

    def on_model_selection_change(self, selected_model: str | None = None):
        self.selected_model_name = selected_model or self.model_option_menu.get()
        self.persist_settings()
        self.update_model_info()
        self.update_control_states()

    def update_model_info(self):
        if not hasattr(self, "model_info_label"):
            return
        model = MODEL_OPTIONS_BY_ID.get(self.selected_model_name, MODEL_OPTIONS_BY_ID[DEFAULT_MODEL_NAME])
        cpu_note = model.get("cpu_suitability", "")
        gpu_note = model.get("gpu_suitability", "")
        if self.device_id == -1 and "wolny" in cpu_note:
            cpu_note += "\nUwaga: na CPU ten model moze wyraznie spowolnic dyktowanie."
        info = (
            f"{model['label']} ({model['id']})\n\n"
            f"Rozmiar: {model['approx_size']}\n"
            f"Jakość: {model['quality']}\n"
            f"Szybkość: {model['speed']}\n"
            f"Rekomendacja: {model['recommended_for']}\n\n"
            f"CPU: {cpu_note}\n"
            f"GPU: {gpu_note}"
        )
        self.model_info_label.configure(text=info)
        current_model = self.current_model_name or "brak"
        self.current_model_label.configure(text=f"Aktualnie załadowany model: {current_model}")
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
        self.device_label.configure(
            text=(
                f"CUDA available: {torch.cuda.is_available()}\n"
                f"CUDA device: {cuda_name}\n"
                f"torch CUDA version: {torch.version.cuda}\n"
                f"Fallback: {'GPU' if self.device_id != -1 else 'CPU'}"
            )
        )
        self.update_status_cards()

    def model_recommendation_text(self) -> str:
        if torch.cuda.is_available():
            first = "CUDA dostępna: rekomendowany openai/whisper-large-v3-turbo."
        else:
            first = "CPU only: rekomendowany openai/whisper-small."
        return (
            "Rekomendacja\n\n"
            f"{first}\n"
            "Dla jakości PL przy dłuższym czasie: bardsai/whisper-medium-pl.\n"
            "Do szybkich testów: openai/whisper-tiny."
        )

    def update_dictation_status(self):
        if hasattr(self, "session_status_label"):
            if self.session_type == "wayland":
                text = "Wayland detected: auto-paste/global hotkey mogą działać niestabilnie. Fallback zostawia tekst w schowku."
            elif self.session_type == "x11":
                text = "X11 detected: xdotool/xclip są preferowane, jeśli są zainstalowane."
            else:
                text = f"Session type: {self.session_type}. Auto-paste support may vary."
            self.session_status_label.configure(text=text)
        if hasattr(self, "hotkey_status_label"):
            self.hotkey_status_label.configure(text=self.hotkey_status_text)
        self.update_status_cards()

    def update_diagnostics(self):
        if not hasattr(self, "environment_label"):
            return
        audio_devices_text = "\n".join(
            f"- {device['index']}: {device['name']} ({device['channels']} ch, {device['rate']} Hz)"
            for device in self.audio_input_devices
        ) or "- brak wykrytych wejść audio"
        self.environment_label.configure(
            text=(
                f"OS: {sys.platform}\n"
                f"XDG_SESSION_TYPE: {self.session_type}\n"
                f"Python: {sys.version.split()[0]}\n"
                f"torch: {getattr(torch, '__version__', 'unknown')}\n"
                f"CUDA available: {torch.cuda.is_available()}\n"
                f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}\n"
                f"Selected input device: {self.input_device_index if self.input_device_index is not None else 'system default'}\n"
                f"Audio input devices:\n{audio_devices_text}"
            )
        )
        self.dependencies_label.configure(
            text=(
                f"xdotool: {bool(shutil.which('xdotool'))}\n"
                f"xclip: {bool(shutil.which('xclip'))}\n"
                f"wl-copy: {bool(shutil.which('wl-copy'))}\n"
                f"pyautogui: {self.module_available('pyautogui')}\n"
                f"pynput: {keyboard is not None}\n"
                f"pyperclip: {self.module_available('pyperclip')}\n"
                f"llama_cpp: {self.module_available('llama_cpp')}\n"
                f"Ollama URL: {self.prompt_processing_ollama_url}\n"
                f"LLM backend: {self.prompt_processing_backend}\n"
                f"LLM model: {self.prompt_processing_model}\n"
                f"LLM connection: {self.llm_connection_summary}"
            )
        )

    def module_available(self, module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except Exception:
            return False

    def append_diagnostic_output(self, text: str):
        if hasattr(self, "diagnostics_output"):
            self.diagnostics_output.configure(state="normal")
            self.diagnostics_output.insert("end", text + "\n")
            self.diagnostics_output.see("end")

    def refresh_logs(self):
        if not hasattr(self, "logs_output"):
            return
        try:
            with open(LOG_FILENAME, "r", encoding="utf-8") as log_file:
                lines = log_file.readlines()[-100:]
            self.set_textbox_text(self.logs_output, "".join(lines) or "Log jest pusty.", disabled=True)
        except OSError as e:
            self.set_textbox_text(self.logs_output, f"Nie można odczytać loga: {e}", disabled=True)

    def load_selected_model(self):
        self.load_model_async(self.selected_model_name)

    def load_model_async(self, model_name: str):
        if self.model_loading:
            return
        self.model_loading = True
        self.set_app_state("loading_model")
        logging.info(f"Starting background model load: {model_name}")
        threading.Thread(target=self.load_model_worker, args=(model_name,), daemon=True).start()

    def load_model_worker(self, model_name: str):
        try:
            loaded_pipeline = self.create_asr_pipeline(model_name)
            self.transcription_queue.put({"type": "model_loaded", "model_name": model_name, "pipeline": loaded_pipeline})
        except Exception as e:
            logging.error(f"Could not load model {model_name}: {e}", exc_info=True)
            self.transcription_queue.put({"type": "model_error", "model_name": model_name, "error": str(e)})

    def create_asr_pipeline(self, model_name: str):
        device = 0 if torch.cuda.is_available() else -1
        logging.info(f"Creating ASR pipeline for {model_name} on device={device}.")
        return pipeline("automatic-speech-recognition", model=model_name, device=device)

    def load_history(self):
        try:
            with open(history_filename(), "r", encoding="utf-8") as history_file:
                history = json.load(history_file)
        except FileNotFoundError:
            self.transcription_history = []
            return
        except (json.JSONDecodeError, OSError) as e:
            logging.error(f"Could not load transcription history: {e}", exc_info=True)
            self.transcription_history = []
            return
        self.transcription_history = [item for item in history if isinstance(item, dict) and isinstance(item.get("text"), str)] if isinstance(history, list) else []

    def save_history(self):
        try:
            with open(history_filename(), "w", encoding="utf-8") as history_file:
                json.dump(self.transcription_history, history_file, ensure_ascii=False, indent=2)
        except OSError as e:
            logging.error(f"Could not save transcription history: {e}", exc_info=True)
            messagebox.showerror("History Error", f"Could not save transcription history: {e}")

    def refresh_history_list(self, selected_index=None):
        if not hasattr(self, "history_list_frame"):
            return
        for child in self.history_list_frame.winfo_children():
            child.destroy()
        self.history_cards = []
        if not self.transcription_history:
            ctk.CTkLabel(self.history_list_frame, text="Brak transkrypcji.", text_color="gray70").pack(padx=12, pady=12, anchor="w")
            self.selected_history_index = None
            self.show_history_placeholder("Brak transkrypcji.")
            return
        if selected_index is None:
            selected_index = 0
        selected_index = max(0, min(selected_index, len(self.transcription_history) - 1))
        for index, entry in enumerate(self.transcription_history):
            timestamp = format_history_timestamp(entry.get("timestamp", time.time()))
            preview_source = entry.get("processed_text") or entry.get("text", "")
            preview = preview_source.replace("\n", " ").strip()
            if len(preview) > 90:
                preview = preview[:87] + "..."
            paste = "paste OK" if entry.get("paste_success") else "paste n/a"
            llm = "LLM OK" if entry.get("processed_text") else ("LLM error" if entry.get("prompt_processing_error") else "LLM n/a")
            card = ctk.CTkFrame(self.history_list_frame, corner_radius=8)
            card.pack(fill="x", padx=8, pady=6)
            card.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(card, text=timestamp, text_color="gray70", anchor="w").grid(row=0, column=0, padx=10, pady=(8, 0), sticky="ew")
            ctk.CTkLabel(card, text=preview or "(puste)", anchor="w", justify="left").grid(row=1, column=0, padx=10, pady=2, sticky="ew")
            ctk.CTkLabel(card, text=f"{entry.get('model_name', 'model n/a')} | {paste} | {llm}", text_color="gray65", anchor="w").grid(row=2, column=0, padx=10, pady=(0, 8), sticky="ew")
            card.bind("<Button-1>", lambda _event, i=index: self.show_history_entry(i))
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda _event, i=index: self.show_history_entry(i))
            self.history_cards.append(card)
        self.show_history_entry(selected_index)

    def show_history_placeholder(self, text: str):
        if hasattr(self, "history_metadata_label"):
            self.history_metadata_label.configure(text=text)
            self.set_textbox_text(self.history_display, text, disabled=True)

    def show_history_entry(self, index: int):
        if index < 0 or index >= len(self.transcription_history):
            self.selected_history_index = None
            self.show_history_placeholder("Wybierz wpis z historii.")
            return
        self.selected_history_index = index
        for card_index, card in enumerate(self.history_cards):
            card.configure(fg_color=("#2b5f87", "#1f4f72") if card_index == index else ("gray86", "gray17"))
        entry = self.transcription_history[index]
        timestamp = format_history_timestamp(entry.get("timestamp", time.time()))
        metadata = (
            f"Data: {timestamp}\n"
            f"Audio: {entry.get('audio_path', 'unknown')}\n"
            f"Model: {entry.get('model_name', 'unknown')}\n"
            f"Device: {entry.get('device', 'unknown')}\n"
            f"Paste success: {entry.get('paste_success', False)}\n"
            f"Paste method: {entry.get('paste_method', 'unknown')}\n"
            f"Paste output: {entry.get('paste_output_used', 'unknown')}\n"
            f"LLM: {entry.get('prompt_processing_backend', 'n/a')} / {entry.get('prompt_processing_model', 'n/a')}\n"
            f"LLM time: {entry.get('prompt_processing_elapsed_seconds', 'n/a')}\n"
            f"LLM error: {entry.get('prompt_processing_error') or 'n/a'}"
        )
        self.history_metadata_label.configure(text=metadata)
        self.update_history_buttons()
        self.set_textbox_text(self.history_display, self.history_entry_text(entry), disabled=True)

    def history_entry_text(self, entry: dict) -> str:
        if self.history_view_mode == "processed":
            return entry.get("processed_text") or "Prompt LLM nie został wygenerowany."
        return entry.get("text", "")

    def update_history_buttons(self):
        if not hasattr(self, "history_original_button"):
            return
        self.history_original_button.configure(fg_color=("#3B8ED0", "#1F6AA5") if self.history_view_mode == "original" else "transparent")
        self.history_processed_button.configure(fg_color=("#3B8ED0", "#1F6AA5") if self.history_view_mode == "processed" else "transparent")

    def set_history_view_mode(self, mode: str):
        self.history_view_mode = "processed" if mode == "processed" else "original"
        entry = self.selected_history_entry()
        self.update_history_buttons()
        if entry:
            self.set_textbox_text(self.history_display, self.history_entry_text(entry), disabled=True)

    def selected_history_entry(self):
        if self.selected_history_index is None:
            return None
        if self.selected_history_index < 0 or self.selected_history_index >= len(self.transcription_history):
            return None
        return self.transcription_history[self.selected_history_index]

    def copy_selected_history(self, mode: str = "original"):
        entry = self.selected_history_entry()
        if not entry:
            self.set_status("Wybierz wpis historii")
            return
        text = entry.get("processed_text", "") if mode == "processed" else entry.get("text", "")
        if mode == "processed" and not text.strip():
            self.set_status("Prompt LLM nie został wygenerowany")
            return
        self.copy_to_clipboard(text)

    def delete_selected_history(self):
        if self.selected_history_index is None:
            self.set_status("Wybierz wpis historii")
            return
        if not messagebox.askyesno("Usuń transkrypcję", "Usunąć zaznaczoną transkrypcję z historii?"):
            return
        entry_index = self.selected_history_index
        del self.transcription_history[entry_index]
        self.save_history()
        self.refresh_history_list(selected_index=entry_index)

    def clear_history(self):
        if not self.transcription_history:
            return
        if not messagebox.askyesno("Wyczyść historię", "Usunąć całą historię transkrypcji?"):
            return
        self.transcription_history = []
        self.save_history()
        self.refresh_history_list()

    def add_history_entry(self, transcription, audio_path, metadata=None):
        history_entry = {"timestamp": time.time(), "audio_path": audio_path, "text": transcription}
        if metadata:
            history_entry.update(metadata)
        self.transcription_history.insert(0, history_entry)
        self.save_history()
        self.refresh_history_list(selected_index=0)

    def copy_to_clipboard(self, text: str):
        copied = self.clipboard_manager.copy_text(text)
        verified = self.clipboard_manager.verify_text(text) if copied else False
        if self.session_type == "x11":
            self.clipboard_manager.get_targets()
        logging.info("Manual clipboard copy result: copied=%s verified=%s text_length=%s", copied, verified, len(text))
        self.set_status("Skopiowano do schowka" if copied else "Nie udało się skopiować do schowka")
        return copied

    def paste_text_to_active_input(self, text: str, target_window_id: str | None = None) -> PasteResult:
        self.set_status("Wklejanie do aktywnego inputu...")
        result = self.paste_manager.paste_text(text, target_window_id)
        self.last_paste_result = result
        self.update_status_cards()
        return result

    def handle_successful_transcription(self, transcription: str, metadata: dict, processed_text: str = "") -> PasteResult:
        effective_paste_preference = self.paste_output_preference
        if metadata.get("prompt_processing_enabled_for_recording"):
            effective_paste_preference = "processed"
        output_text, output_used = select_paste_output(
            transcription,
            processed_text,
            effective_paste_preference,
            metadata.get("prompt_processing_error") or None,
            self.fallback_to_original_on_llm_error,
        )
        metadata["paste_output_preference"] = effective_paste_preference
        metadata["paste_output_used"] = output_used
        if effective_paste_preference == "processed" and output_used == "original_no_processed_prompt":
            self.set_status("Prompt LLM nie został wygenerowany. Wklejam oryginał.")
        prepared_text = prepare_text_for_paste(
            output_text,
            trim_text=self.trim_text_before_paste,
            capitalize_first_letter=self.capitalize_first_letter,
            append_space=self.append_space_after_paste,
            append_newline=self.append_newline_after_paste,
        )
        fallback_result = PasteResult(False, "fallback_manual_clipboard_only", None, False, False, False, metadata.get("target_window_id"), self.session_type, time.strftime("%Y-%m-%dT%H:%M:%S"))
        if not prepared_text.strip() or is_transcription_error(prepared_text):
            fallback_result.error = "Prepared text is empty or an error."
            self.set_status("Gotowe")
            return fallback_result
        if self.dictation_mode_enabled and self.auto_paste_enabled and self.auto_paste_after_transcription:
            if self.session_type == "wayland" and self.show_wayland_warning:
                self.set_status("Wayland detected: auto-paste może działać niestabilnie")
            paste_result = self.paste_text_to_active_input(prepared_text, metadata.get("target_window_id"))
            if paste_result.success:
                self.set_status("Wklejono do aktywnego inputu")
                show_desktop_notification(APP_TITLE, "Pasted transcription into active input.")
            else:
                self.set_status("Auto-paste nieudany, tekst jest w schowku")
                show_desktop_notification(APP_TITLE, "Could not auto-paste. Text copied to clipboard.")
            return paste_result
        if self.copy_to_clipboard_enabled:
            copied = self.copy_to_clipboard(prepared_text)
            verified = self.clipboard_manager.verify_text(prepared_text) if copied else False
            fallback_result.copied_to_clipboard = copied
            fallback_result.clipboard_verified = verified
            fallback_result.error = None if copied else "Could not copy text to clipboard."
        return fallback_result

    def test_auto_paste(self):
        test_text = "Test auto-paste z Azor Transcriber"
        target_window_id = self.target_window_id
        if self.session_type == "x11" and not target_window_id:
            target_window_id = self.paste_manager.get_active_window_id()
        result = self.paste_manager.paste_text(test_text, target_window_id)
        self.last_paste_result = result
        output = (
            f"Auto-paste test\n"
            f"success: {result.success}\nmethod: {result.method}\n"
            f"copied_to_clipboard: {result.copied_to_clipboard}\nclipboard_verified: {result.clipboard_verified}\n"
            f"restored_window: {result.restored_window}\ntarget_window_id: {result.target_window_id or 'unknown'}\n"
            f"session_type: {result.session_type}\nerror: {result.error or 'n/a'}"
        )
        logging.info("Auto-paste diagnostic result: %s", result.to_dict())
        self.append_diagnostic_output(output)
        self.set_status("Wklejono do aktywnego inputu" if result.success else "Auto-paste nieudany, tekst jest w schowku")

    def check_clipboard(self):
        text = self.clipboard_manager.read_text()
        targets = self.clipboard_manager.get_targets()
        preview = text if text is not None else ""
        if len(preview) > 1000:
            preview = preview[:1000] + "..."
        output = f"Clipboard\nsession_type: {self.session_type}\ntext_length: {len(text or '')}\nTARGETS: {', '.join(targets) if targets else 'n/a'}\n\n{preview or '(empty or unavailable)'}"
        logging.info("Clipboard diagnostic: text_length=%s targets=%s", len(text or ""), targets)
        self.append_diagnostic_output(output)
        self.set_status("Diagnostyka schowka odświeżona")

    def test_copy_text_plain(self):
        text = "Test text/plain z Azor Transcriber"
        copied = self.clipboard_manager.copy_text(text)
        verified = self.clipboard_manager.verify_text(text) if copied else False
        targets = self.clipboard_manager.get_targets()
        self.append_diagnostic_output(f"Test copy text/plain\ncopied: {copied}\nverified: {verified}\nTARGETS: {targets}")
        self.set_status("Test text/plain zakończony")

    def start_hotkey_listener(self) -> bool:
        self.stop_hotkey_listener()
        if not self.global_hotkey_enabled:
            self.hotkey_status_text = "Global hotkey disabled"
            self.update_dictation_status()
            return True
        if keyboard is None:
            self.hotkey_status_text = "Global hotkey unavailable: install pynput"
            self.update_dictation_status()
            logging.warning("pynput is not installed. Global hotkey disabled.")
            return False
        if self.session_type == "wayland":
            logging.warning("Wayland detected. pynput global hotkeys may not work reliably.")
        try:
            self.hotkey_listener = keyboard.GlobalHotKeys({self.global_hotkey: lambda: self.request_toggle_recording("global_hotkey")})
            self.hotkey_listener.start()
            self.hotkey_status_text = f"Global hotkey active: {self.global_hotkey}"
            self.update_dictation_status()
            logging.info(self.hotkey_status_text)
            return True
        except Exception as e:
            self.hotkey_listener = None
            self.hotkey_status_text = f"Global hotkey failed: {self.global_hotkey}"
            self.update_dictation_status()
            logging.error(f"Could not start global hotkey {self.global_hotkey}: {e}", exc_info=True)
            return False

    def stop_hotkey_listener(self):
        if self.hotkey_listener is not None:
            try:
                self.hotkey_listener.stop()
            except Exception as e:
                logging.warning(f"Could not stop hotkey listener: {e}")
            self.hotkey_listener = None

    def restart_hotkey_listener(self) -> bool:
        return self.start_hotkey_listener()

    def request_toggle_recording(self, trigger_source: str):
        self.master.after(0, lambda: self.toggle_recording(trigger_source))

    def update_floating_window(self):
        if self.floating_window_enabled:
            if self.floating_window is None or not self.floating_window.winfo_exists():
                self.floating_window = ctk.CTkToplevel(self.master)
                self.floating_window.title("Azor Dictate")
                self.floating_window.geometry("190x76")
                self.floating_window.attributes("-topmost", True)
                self.floating_window.protocol("WM_DELETE_WINDOW", self.hide_floating_window)
                self.floating_button = ctk.CTkButton(self.floating_window, text="Dictate", command=lambda: self.request_toggle_recording("floating_button"))
                self.floating_button.pack(fill="both", expand=True, padx=8, pady=8)
            else:
                self.floating_window.deiconify()
            self.update_floating_button_state()
        elif self.floating_window is not None and self.floating_window.winfo_exists():
            self.floating_window.withdraw()

    def hide_floating_window(self):
        self.floating_window_enabled = False
        if hasattr(self, "floating_window_var"):
            self.floating_window_var.set(False)
        self.persist_settings()
        if self.floating_window is not None and self.floating_window.winfo_exists():
            self.floating_window.withdraw()

    def update_floating_button_state(self):
        if not getattr(self, "floating_button", None):
            return
        if self.model_loading:
            self.floating_button.configure(text="Loading...", state="disabled")
        elif self.transcribing:
            self.floating_button.configure(text="Transcribing...", state="disabled")
        elif self.recording:
            self.floating_button.configure(text="Stop", state="normal")
        elif self.asr_pipeline is None:
            self.floating_button.configure(text="Dictate", state="disabled")
        else:
            self.floating_button.configure(text="Dictate", state="normal")

    def update_control_states(self):
        model_busy = self.model_loading
        transcription_busy = self.transcribing or self.recording or self.app_state == "saving"
        model_ready = self.asr_pipeline is not None
        if hasattr(self, "record_button"):
            if model_busy:
                self.record_button.configure(text="Ładowanie modelu...", state="disabled")
            elif self.app_state == "saving":
                self.record_button.configure(text="Zapisywanie...", state="disabled")
            elif self.transcribing:
                self.record_button.configure(text="Transkrypcja...", state="disabled")
            elif not model_ready:
                self.record_button.configure(text="Załaduj model, aby dyktować", state="disabled")
            elif self.recording:
                self.record_button.configure(text="■ Stop nagrywania", state="normal")
            else:
                self.record_button.configure(text="🎙 Start dyktowania", state="normal")
        if hasattr(self, "model_option_menu"):
            self.model_option_menu.configure(state="disabled" if transcription_busy or model_busy else "normal")
        if hasattr(self, "load_model_button"):
            can_load = not transcription_busy and not model_busy and self.selected_model_name != self.current_model_name
            self.load_model_button.configure(state="normal" if can_load else "disabled")
        self.update_floating_button_state()
        self.update_status_cards()

    def toggle_recording(self, trigger_source: str = "main_button"):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording(trigger_source)

    def start_recording(self, trigger_source: str = "main_button"):
        if self.model_loading:
            self.set_status("Poczekaj na załadowanie modelu")
            return
        if self.asr_pipeline is None:
            self.set_status("Najpierw załaduj model w widoku Modele")
            return
        self.recording = True
        self.current_trigger_source = trigger_source
        self.current_record_prompt_processing_enabled = self.prompt_processing_enabled and self.prompt_processing_default_for_recordings
        self.target_window_id = None
        if self.dictation_mode_enabled and self.auto_paste_enabled and self.session_type == "x11":
            self.target_window_id = self.paste_manager.get_active_window_id()
            if not self.target_window_id:
                logging.warning("Could not capture target X11 window before recording.")
        if trigger_source == "main_button" and self.auto_paste_enabled:
            logging.warning("Recording started from the main app. Auto-paste may target the app unless focus can be restored.")
        self.frames = []
        self.start_time = time.time()
        self.audio_levels = []
        self.last_audio_level = 0.0
        self.audio_level_peak = 0.0
        self.audio_level_sum = 0.0
        self.audio_level_count = 0
        self.current_record_limit_seconds = self.get_max_record_duration_seconds()
        logging.info(
            "Recording started. trigger_source=%s, target_window_id=%s, prompt_processing=%s",
            trigger_source,
            self.target_window_id,
            self.current_record_prompt_processing_enabled,
        )
        self.set_app_state("recording")
        self.set_transcription_text("Nagrywanie w toku... (max %ss)" % self.current_record_limit_seconds)
        try:
            input_kwargs = {}
            if self.input_device_index is not None:
                input_kwargs["input_device_index"] = self.input_device_index
            logging.info("Opening PyAudio input stream. selected_input_device_index=%s rate=%s channels=%s", self.input_device_index, RATE, CHANNELS)
            with suppress_native_stderr():
                self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, **input_kwargs)
            self.read_chunk()
            if self.auto_stop_recording_enabled:
                self.record_timer_id = self.master.after(self.current_record_limit_seconds * 1000, self.auto_stop_recording)
            self.update_recording_timer_ui()
            self.update_waveform_canvas()
        except Exception as e:
            self.recording = False
            self.cancel_recording_ui_timers()
            self.set_app_state("error")
            logging.error(f"Microphone stream error on start: {e}", exc_info=True)
            messagebox.showerror("Audio Error", f"Could not open microphone stream: {e}\nCheck your microphone connection and permissions.")
            if self.record_timer_id:
                try:
                    self.master.after_cancel(self.record_timer_id)
                except Exception:
                    logging.debug("Record auto-stop callback was already cleared after microphone error.", exc_info=True)
                self.record_timer_id = None

    def read_chunk(self):
        if self.recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                self.push_audio_level(self.calculate_audio_level(data))
                self.master.after(1, self.read_chunk)
            except IOError as e:
                logging.error(f"Stream read IOError: {e}", exc_info=True)
                self.stop_recording()

    def auto_stop_recording(self):
        if self.recording:
            limit = self.current_record_limit_seconds
            logging.info(f"Automatic stop triggered after {limit} seconds.")
            self.set_status(f"Osiągnięto limit {limit}s. Rozpoczynam transkrypcję...")
            self.stop_recording(auto_stopped=True)

    def stop_recording(self, auto_stopped: bool = False):
        if not self.recording:
            return
        self.recording = False
        if self.record_timer_id:
            try:
                self.master.after_cancel(self.record_timer_id)
            except Exception:
                logging.debug("Record auto-stop callback was already cleared.", exc_info=True)
            self.record_timer_id = None
        self.cancel_recording_ui_timers()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        logging.info("Audio stream closed. auto_stopped=%s elapsed=%.2fs frames=%s", auto_stopped, elapsed, len(self.frames))
        if not self.frames:
            self.set_status("Brak danych audio. Nie uruchamiam transkrypcji.")
            self.set_app_state("error")
            return
        if elapsed < 0.5:
            self.set_status("Nagranie jest zbyt krótkie. Nie uruchamiam transkrypcji.")
            self.set_app_state("error")
            return
        self.is_probably_silent_recording()
        self.set_app_state("saving")
        wave_output_filename = output_filename()
        try:
            with wave.open(wave_output_filename, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(self.frames))
            logging.info(f"File saved successfully to {wave_output_filename}")
            if not self.auto_transcribe_after_stop:
                self.set_status(f"Nagranie zapisane: {wave_output_filename}")
                self.set_app_state("idle")
                return
            self.transcribing = True
            self.set_app_state("transcribing")
            self.set_transcription_text("Transkrypcja w toku...")
            threading.Thread(
                target=self.run_transcription,
                args=(wave_output_filename, self.current_trigger_source, self.target_window_id, self.current_record_prompt_processing_enabled),
                daemon=True,
            ).start()
            logging.info("Transcription thread started.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save WAVE file: {e}")
            self.transcribing = False
            self.set_app_state("error")
            logging.error(f"Error saving wave file: {e}", exc_info=True)

    def run_transcription(self, audio_path, trigger_source="main_button", target_window_id=None, prompt_processing_for_recording=False):
        logging.info(f"Running transcription for {audio_path} in thread: {threading.get_ident()}")
        try:
            if self.asr_pipeline is None:
                raise RuntimeError("No ASR model is loaded.")
            processed_audio_path = preprocess_audio(audio_path)
            duration_seconds = get_audio_duration_seconds(processed_audio_path)
            logging.info(f"Starting transcription. model={self.current_model_name}, device={self.device_name}, language={self.language}, num_beams={self.num_beams}, audio={processed_audio_path}")
            started_at = time.time()
            generate_kwargs = {"task": "transcribe", "num_beams": self.num_beams}
            if self.language != "auto":
                generate_kwargs["language"] = self.language
            call_kwargs = build_asr_call_kwargs(duration_seconds, generate_kwargs)
            logging.info(
                "ASR call options. duration=%.2fs long_form=%s return_timestamps=%s",
                duration_seconds,
                duration_seconds > LONG_FORM_TRANSCRIPTION_THRESHOLD_SECONDS,
                call_kwargs.get("return_timestamps", False),
            )
            result = self.asr_pipeline(processed_audio_path, **call_kwargs)
            transcription = postprocess_transcription(result["text"].strip())
            elapsed_seconds = round(time.time() - started_at, 2)
            logging.info(f"Transcription finished in {elapsed_seconds}s.")
            processed_text = ""
            prompt_processing_metadata = {
                "prompt_processing_enabled_for_recording": bool(prompt_processing_for_recording),
                "processed_text": "",
                "prompt_processing_backend": "",
                "prompt_processing_model": "",
                "prompt_processing_elapsed_seconds": 0,
                "prompt_processing_error": "",
            }
            if prompt_processing_for_recording and transcription.strip():
                self.transcription_queue.put({"type": "status", "status": "Porządkowanie promptu..."})
                prompt_result = self.process_prompt_text(transcription)
                processed_text = prompt_result.text
                prompt_processing_metadata.update({
                    "processed_text": processed_text,
                    "prompt_processing_backend": prompt_result.backend,
                    "prompt_processing_model": prompt_result.model,
                    "prompt_processing_elapsed_seconds": prompt_result.elapsed_seconds,
                    "prompt_processing_error": prompt_result.error or "",
                })
            self.transcription_queue.put({
                "type": "transcription",
                "text": transcription,
                "processed_text": processed_text,
                "audio_path": audio_path,
                "metadata": {
                    "model_name": self.current_model_name,
                    "device": self.device_name,
                    "language": self.language,
                    "duration_seconds": duration_seconds,
                    "auto_paste_enabled": self.auto_paste_enabled,
                    "paste_success": False,
                    "target_window_id": target_window_id,
                    "trigger_source": trigger_source,
                    "hotkey": self.global_hotkey if trigger_source == "global_hotkey" else "",
                    **prompt_processing_metadata,
                },
            })
        except FileNotFoundError:
            logging.error(f"Audio file not found at path: {audio_path}")
            self.transcription_queue.put({"type": "transcription", "text": f"ERROR: Audio file not found at path: {audio_path}", "audio_path": audio_path, "metadata": {}})
        except Exception as e:
            logging.error(f"An unexpected error occurred during transcription: {e}", exc_info=True)
            self.transcription_queue.put({"type": "transcription", "text": f"ERROR: An unexpected error occurred during transcription: {e}", "audio_path": audio_path, "metadata": {}})

    def check_transcription_queue(self):
        try:
            queue_item = self.transcription_queue.get(block=False)
            item_type = queue_item.get("type", "transcription") if isinstance(queue_item, dict) else "transcription"
            if item_type == "model_loaded":
                self.asr_pipeline = queue_item["pipeline"]
                self.current_model_name = queue_item["model_name"]
                self.model_loading = False
                self.persist_settings()
                self.set_app_state("idle")
                self.update_model_info()
                logging.info(f"Model loaded successfully: {self.current_model_name}")
            elif item_type == "model_error":
                self.model_loading = False
                self.set_app_state("error")
                model_name = queue_item.get("model_name", "selected model")
                error = queue_item.get("error", "Unknown error")
                messagebox.showerror("Model Loading Failed", f"Could not load {model_name}.\n\n{error}\n\nCheck transcriber.log for details.")
            elif item_type == "status":
                status_text = queue_item.get("status", "Praca w toku...")
                if "Porządkowanie" in status_text:
                    self.app_state = "processing_llm"
                    self.set_recording_widget_state("processing_llm")
                self.set_status(status_text)
            elif item_type == "llm_connection_status":
                self.apply_llm_connection_status(queue_item["status"])
            elif item_type == "manual_prompt_error":
                self.set_status(f"LLM niedostępny: {queue_item.get('message', 'sprawdź konfigurację')}")
            elif item_type == "transcription":
                self.transcribing = False
                result = queue_item.get("text", "")
                processed_text = queue_item.get("processed_text", "")
                audio_path = queue_item.get("audio_path", "")
                metadata = queue_item.get("metadata", {})
                self.last_transcription_text = "" if is_transcription_error(result) else result
                self.last_processed_text = "" if is_transcription_error(result) else processed_text
                self.last_prompt_processing_error = metadata.get("prompt_processing_error") or None
                if self.last_processed_text and self.paste_output_preference == "processed":
                    self.preview_mode = "processed"
                self.set_preview_mode(self.preview_mode)
                if is_transcription_error(result) or not result.strip() or is_likely_silence_hallucination(result):
                    self.set_app_state("error")
                    logging.warning("Transcription failed, returned empty text, or looked like a silence hallucination.")
                    if is_transcription_error(result):
                        messagebox.showerror("Transcription Failed", "Transcription returned an error. Check logs for details.")
                    elif is_likely_silence_hallucination(result):
                        self.set_status("Wykryto prawdopodobną halucynację ciszy. Nie wklejam tekstu.")
                        self.set_transcription_text("Nie wykryto mowy w nagraniu.")
                    else:
                        self.set_status("Transkrypcja jest pusta. Nie wklejam tekstu.")
                else:
                    logging.info("Successful transcription text_length=%s", len(result))
                    paste_result = self.handle_successful_transcription(result, metadata, processed_text)
                    metadata["copy_to_clipboard_enabled"] = self.copy_to_clipboard_enabled
                    metadata["clipboard_copied"] = paste_result.copied_to_clipboard
                    metadata["clipboard_verified"] = paste_result.clipboard_verified
                    metadata["auto_paste_enabled"] = self.auto_paste_enabled
                    metadata["paste_success"] = paste_result.success
                    metadata["paste_method"] = paste_result.method
                    metadata["paste_error"] = paste_result.error
                    metadata["restored_window"] = paste_result.restored_window
                    metadata["paste_result"] = paste_result.to_dict()
                    self.add_history_entry(result, audio_path, metadata)
                    self.app_state = "idle"
                    self.set_recording_widget_state("done")
                    self.recording_widget_hide_id = self.master.after(2000, self.hide_recording_widget)
            elif item_type == "prompt_reprocessed_last":
                prompt_result = queue_item["result"]
                self.last_processed_text = prompt_result.text
                self.last_prompt_processing_error = prompt_result.error
                self.preview_mode = "processed"
                self.set_preview_mode("processed")
                self.app_state = "idle"
                self.hide_recording_overlay()
                self.hide_recording_widget()
                if prompt_result.error:
                    self.set_status(f"LLM error: {prompt_result.error}")
                else:
                    self.set_status("Prompt LLM gotowy")
            elif item_type == "prompt_reprocessed_history":
                prompt_result = queue_item["result"]
                index = queue_item.get("index")
                if isinstance(index, int) and 0 <= index < len(self.transcription_history):
                    entry = self.transcription_history[index]
                    entry["processed_text"] = prompt_result.text
                    entry["prompt_processing_enabled_for_recording"] = True
                    entry["prompt_processing_backend"] = prompt_result.backend
                    entry["prompt_processing_model"] = prompt_result.model
                    entry["prompt_processing_elapsed_seconds"] = prompt_result.elapsed_seconds
                    entry["prompt_processing_error"] = prompt_result.error or ""
                    self.save_history()
                    self.history_view_mode = "processed"
                    self.refresh_history_list(selected_index=index)
                if prompt_result.error:
                    self.set_status(f"LLM error: {prompt_result.error}")
                else:
                    self.set_status("Prompt LLM w historii zaktualizowany")
            if item_type != "status":
                self.update_control_states()
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.check_transcription_queue)

    def on_closing(self):
        logging.info("Closing application...")
        self.stop_hotkey_listener()
        if self.recording:
            self.stop_recording()
        if self.p:
            self.p.terminate()
        if self.floating_window is not None and self.floating_window.winfo_exists():
            self.floating_window.destroy()
        if self.recording_overlay_window is not None and self.recording_overlay_window.winfo_exists():
            self.recording_overlay_window.destroy()
        self.master.destroy()
        logging.info("Application destroyed.")


# --- Application Startup ---
if __name__ == "__main__":
    logging.info("Whisper model loading might take a moment on first launch...")
    if ctk is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing dependency", "customtkinter is not installed.\nRun: pip install customtkinter")
        raise SystemExit(1)
    startup_settings = load_settings()
    ctk.set_appearance_mode(startup_settings.get("appearance_mode", "dark"))
    ctk.set_default_color_theme("blue")
    try:
        ctk.set_widget_scaling(int(startup_settings.get("ui_scaling", "100%").replace("%", "")) / 100)
    except ValueError:
        ctk.set_widget_scaling(1.0)
    root = ctk.CTk()
    app = AudioRecorderApp(root)
    root.mainloop()
