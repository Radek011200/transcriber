from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
import time
import urllib.error
import urllib.request


DEFAULT_PROMPT_PROCESSING_SYSTEM_PROMPT = """Uporządkuj poniższą surową transkrypcję jako gotowy prompt do wklejenia do innego narzędzia LLM.

Zasady:
- usuń powtórzenia, zająknięcia i przypadkowe wtrącenia,
- zachowaj intencję, sens i ważne szczegóły,
- uzupełnij brakujące ważne aspekty tylko wtedy, gdy jasno wynikają z kontekstu,
- nie dodawaj komentarzy o wykonanej pracy,
- zwróć wyłącznie gotowy prompt jako czysty tekst."""

LLM_MODEL_OPTIONS = [
    {
        "id": "qwen3:0.6b",
        "label": "Qwen3 0.6B",
        "backend": "ollama",
        "recommendation": "najszybszy test lokalnego porządkowania",
    },
    {
        "id": "qwen3:1.7b",
        "label": "Qwen3 1.7B",
        "backend": "ollama",
        "recommendation": "domyślna rekomendacja do dyktowania",
    },
    {
        "id": "qwen2.5:1.5b",
        "label": "Qwen2.5 1.5B",
        "backend": "ollama/llama-cpp",
        "recommendation": "lekki model instruct",
    },
    {
        "id": "llama3.2:1b",
        "label": "Llama 3.2 1B",
        "backend": "ollama",
        "recommendation": "bardzo lekki model Meta",
    },
    {
        "id": "llama3.2:3b",
        "label": "Llama 3.2 3B",
        "backend": "ollama",
        "recommendation": "lepsza jakość kosztem czasu",
    },
    {
        "id": "gemma3:4b",
        "label": "Gemma 3 4B",
        "backend": "ollama",
        "recommendation": "większy lokalny model Google",
    },
]

LLM_MODEL_OPTIONS_BY_ID = {model["id"]: model for model in LLM_MODEL_OPTIONS}


@dataclass
class PromptProcessingResult:
    text: str
    backend: str
    model: str
    elapsed_seconds: float
    error: str | None = None

    def to_metadata(self) -> dict:
        data = asdict(self)
        data["enabled"] = True
        return data


@dataclass
class LLMConnectionStatus:
    connected: bool
    backend: str
    model: str
    message: str
    model_available: bool = False
    elapsed_seconds: float = 0.0


def build_prompt_messages(system_prompt: str, transcription: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": f"Surowa transkrypcja:\n\n{transcription.strip()}"},
    ]


def select_paste_output(
    original_text: str,
    processed_text: str | None,
    paste_output_preference: str,
    processing_error: str | None = None,
    fallback_to_original_on_llm_error: bool = True,
) -> tuple[str, str]:
    if paste_output_preference != "processed":
        return original_text, "original"
    if processed_text and processed_text.strip():
        return processed_text, "processed"
    if processing_error and fallback_to_original_on_llm_error:
        return original_text, "original_fallback_after_llm_error"
    return original_text, "original_no_processed_prompt"


def check_llm_connection(settings: dict) -> LLMConnectionStatus:
    backend = str(settings.get("prompt_processing_backend", "ollama"))
    model = str(settings.get("prompt_processing_model", "qwen3:1.7b"))
    started_at = time.time()
    try:
        if backend == "ollama":
            return _check_ollama_connection(settings, model, started_at)
        if backend == "llama-cpp-python":
            return _check_llama_cpp_connection(settings, model, started_at)
        return LLMConnectionStatus(False, backend, model, f"Nieznany backend LLM: {backend}", False, round(time.time() - started_at, 2))
    except Exception as e:
        logging.error("LLM connection check failed: %s", e, exc_info=True)
        return LLMConnectionStatus(False, backend, model, str(e), False, round(time.time() - started_at, 2))


def _check_ollama_connection(settings: dict, model: str, started_at: float) -> LLMConnectionStatus:
    url = str(settings.get("prompt_processing_ollama_url") or "http://localhost:11434").rstrip("/")
    request = urllib.request.Request(f"{url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return LLMConnectionStatus(False, "ollama", model, f"Ollama nie odpowiada pod {url}.", False, round(time.time() - started_at, 2))
    except json.JSONDecodeError as e:
        return LLMConnectionStatus(False, "ollama", model, f"Ollama zwróciła niepoprawną odpowiedź: {e}", False, round(time.time() - started_at, 2))

    models = data.get("models", []) if isinstance(data, dict) else []
    model_names = {str(item.get("name", "")) for item in models if isinstance(item, dict)}
    model_available = model in model_names
    if model_available:
        message = "Ollama działa, model jest pobrany."
    else:
        message = f"Ollama działa, ale model nie jest pobrany. Uruchom: ollama pull {model}"
    return LLMConnectionStatus(True, "ollama", model, message, model_available, round(time.time() - started_at, 2))


def _check_llama_cpp_connection(settings: dict, model: str, started_at: float) -> LLMConnectionStatus:
    gguf_path = str(settings.get("prompt_processing_gguf_path") or "").strip()
    if not gguf_path:
        return LLMConnectionStatus(False, "llama-cpp-python", model, "Nie wskazano ścieżki do pliku .gguf.", False, round(time.time() - started_at, 2))
    if not os.path.isfile(gguf_path):
        return LLMConnectionStatus(False, "llama-cpp-python", model, f"Plik .gguf nie istnieje: {gguf_path}", False, round(time.time() - started_at, 2))
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return LLMConnectionStatus(False, "llama-cpp-python", model, "Brak llama-cpp-python w środowisku.", False, round(time.time() - started_at, 2))
    return LLMConnectionStatus(True, "llama-cpp-python", model, "llama-cpp-python gotowy, plik .gguf istnieje.", True, round(time.time() - started_at, 2))


class PromptProcessor:
    def __init__(self, settings: dict):
        self.settings = settings

    def process(self, transcription: str) -> PromptProcessingResult:
        backend = str(self.settings.get("prompt_processing_backend", "ollama"))
        model = str(self.settings.get("prompt_processing_model", "qwen3:1.7b"))
        started_at = time.time()
        try:
            if backend == "ollama":
                text = self._process_with_ollama(transcription)
            elif backend == "llama-cpp-python":
                text = self._process_with_llama_cpp(transcription)
            else:
                raise ValueError(f"Nieznany backend lokalnego LLM: {backend}")
            text = (text or "").strip()
            if not text:
                raise RuntimeError("Lokalny LLM zwrócił pusty tekst.")
            return PromptProcessingResult(text, backend, model, round(time.time() - started_at, 2))
        except Exception as e:
            logging.error("Prompt processing failed: %s", e, exc_info=True)
            return PromptProcessingResult("", backend, model, round(time.time() - started_at, 2), str(e))

    def _messages(self, transcription: str) -> list[dict]:
        system_prompt = str(self.settings.get("prompt_processing_system_prompt") or DEFAULT_PROMPT_PROCESSING_SYSTEM_PROMPT)
        return build_prompt_messages(system_prompt, transcription)

    def _process_with_ollama(self, transcription: str) -> str:
        url = str(self.settings.get("prompt_processing_ollama_url") or "http://localhost:11434").rstrip("/")
        model = str(self.settings.get("prompt_processing_model") or "qwen3:1.7b")
        payload = {
            "model": model,
            "messages": self._messages(transcription),
            "stream": False,
            "options": {
                "temperature": float(self.settings.get("prompt_processing_temperature", 0.2)),
                "num_predict": int(self.settings.get("prompt_processing_max_tokens", 1024)),
            },
        }
        request = urllib.request.Request(
            f"{url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 404 or "pull" in body.lower() or "not found" in body.lower():
                raise RuntimeError(f"Model Ollama nie jest dostępny. Uruchom: ollama pull {model}") from e
            raise RuntimeError(f"Ollama HTTP {e.code}: {body[:300]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama nie odpowiada pod {url}. Sprawdź, czy działa lokalny serwer.") from e

        if isinstance(data, dict) and data.get("error"):
            message = str(data["error"])
            if "pull" in message.lower() or "not found" in message.lower():
                raise RuntimeError(f"Model Ollama nie jest dostępny. Uruchom: ollama pull {model}")
            raise RuntimeError(f"Ollama error: {message}")
        message = data.get("message", {}) if isinstance(data, dict) else {}
        return str(message.get("content", "")).strip()

    def _process_with_llama_cpp(self, transcription: str) -> str:
        gguf_path = str(self.settings.get("prompt_processing_gguf_path") or "").strip()
        if not gguf_path:
            raise RuntimeError("Nie wskazano ścieżki do pliku .gguf dla llama-cpp-python.")
        if not os.path.isfile(gguf_path):
            raise RuntimeError(f"Plik .gguf nie istnieje: {gguf_path}")
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError("Brak llama-cpp-python. Zainstaluj pakiet albo użyj backendu Ollama.") from e

        llm = Llama(model_path=gguf_path, n_ctx=4096, verbose=False)
        response = llm.create_chat_completion(
            messages=self._messages(transcription),
            temperature=float(self.settings.get("prompt_processing_temperature", 0.2)),
            max_tokens=int(self.settings.get("prompt_processing_max_tokens", 1024)),
        )
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return str(message.get("content", "")).strip()
