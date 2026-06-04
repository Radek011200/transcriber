import json
import unittest
from unittest import mock
import urllib.error

from prompt_processor import (
    DEFAULT_PROMPT_PROCESSING_SYSTEM_PROMPT,
    PromptProcessor,
    build_ollama_payload,
    build_prompt_messages,
    check_llm_connection,
    get_llama_cpp_model,
    prompt_processing_num_predict,
    select_paste_output,
)


class PromptProcessorTest(unittest.TestCase):
    def test_build_prompt_messages_contains_system_prompt_and_transcription(self):
        messages = build_prompt_messages("System", "surowy tekst")

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("surowy tekst", messages[1]["content"])

    def test_select_paste_output_prefers_original_by_default(self):
        text, used = select_paste_output("oryginał", "prompt", "original")

        self.assertEqual(text, "oryginał")
        self.assertEqual(used, "original")

    def test_select_paste_output_prefers_processed_when_available(self):
        text, used = select_paste_output("oryginał", "prompt", "processed")

        self.assertEqual(text, "prompt")
        self.assertEqual(used, "processed")

    def test_select_paste_output_falls_back_to_original_on_llm_error(self):
        text, used = select_paste_output("oryginał", "", "processed", "error", True)

        self.assertEqual(text, "oryginał")
        self.assertEqual(used, "original_fallback_after_llm_error")

    def test_ollama_success(self):
        settings = {
            "prompt_processing_backend": "ollama",
            "prompt_processing_model": "qwen3:1.7b",
            "prompt_processing_ollama_url": "http://localhost:11434",
            "prompt_processing_system_prompt": DEFAULT_PROMPT_PROCESSING_SYSTEM_PROMPT,
            "prompt_processing_temperature": 0.2,
            "prompt_processing_max_tokens": 1024,
            "prompt_processing_keep_alive": "10m",
        }
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=None)
        response.read.return_value = json.dumps({"message": {"content": "Gotowy prompt"}}).encode("utf-8")

        with mock.patch("urllib.request.urlopen", return_value=response):
            result = PromptProcessor(settings).process("tekst")

        self.assertEqual(result.text, "Gotowy prompt")
        self.assertIsNone(result.error)

    def test_ollama_payload_contains_keep_alive(self):
        settings = {
            "prompt_processing_model": "qwen3:1.7b",
            "prompt_processing_temperature": 0.2,
            "prompt_processing_max_tokens": 1024,
            "prompt_processing_keep_alive": "10m",
        }

        payload = build_ollama_payload(settings, build_prompt_messages("System", "tekst"), "tekst")

        self.assertEqual(payload["keep_alive"], "10m")
        self.assertEqual(payload["options"]["num_predict"], 1024)

    def test_prompt_processing_max_tokens_is_not_lowered_by_default(self):
        settings = {
            "prompt_processing_max_tokens": 2048,
            "prompt_processing_adaptive_max_tokens": False,
        }

        self.assertEqual(prompt_processing_num_predict(settings, "krótki tekst"), 2048)

    def test_llama_cpp_model_is_cached_for_same_path(self):
        fake_llama = mock.Mock()

        with mock.patch.dict("sys.modules", {"llama_cpp": mock.Mock(Llama=mock.Mock(return_value=fake_llama))}):
            first = get_llama_cpp_model("/tmp/model.gguf", 4096)
            second = get_llama_cpp_model("/tmp/model.gguf", 4096)

        self.assertIs(first, fake_llama)
        self.assertIs(second, fake_llama)

    def test_ollama_connection_error_returns_error_result(self):
        settings = {
            "prompt_processing_backend": "ollama",
            "prompt_processing_model": "qwen3:1.7b",
            "prompt_processing_ollama_url": "http://localhost:11434",
        }

        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = PromptProcessor(settings).process("tekst")

        self.assertEqual(result.text, "")
        self.assertIn("Ollama nie odpowiada", result.error)

    def test_ollama_connection_check_success_with_model(self):
        settings = {
            "prompt_processing_backend": "ollama",
            "prompt_processing_model": "qwen3:1.7b",
            "prompt_processing_ollama_url": "http://localhost:11434",
        }
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=None)
        response.read.return_value = json.dumps({"models": [{"name": "qwen3:1.7b"}]}).encode("utf-8")

        with mock.patch("urllib.request.urlopen", return_value=response):
            status = check_llm_connection(settings)

        self.assertTrue(status.connected)
        self.assertTrue(status.model_available)
        self.assertIn("model jest pobrany", status.message)

    def test_ollama_connection_check_model_missing(self):
        settings = {
            "prompt_processing_backend": "ollama",
            "prompt_processing_model": "qwen3:1.7b",
            "prompt_processing_ollama_url": "http://localhost:11434",
        }
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=None)
        response.read.return_value = json.dumps({"models": [{"name": "llama3.2:1b"}]}).encode("utf-8")

        with mock.patch("urllib.request.urlopen", return_value=response):
            status = check_llm_connection(settings)

        self.assertTrue(status.connected)
        self.assertFalse(status.model_available)
        self.assertIn("ollama pull qwen3:1.7b", status.message)

    def test_llama_cpp_missing_gguf_returns_error_result(self):
        settings = {
            "prompt_processing_backend": "llama-cpp-python",
            "prompt_processing_model": "qwen2.5:1.5b",
            "prompt_processing_gguf_path": "",
        }

        result = PromptProcessor(settings).process("tekst")

        self.assertEqual(result.text, "")
        self.assertIn(".gguf", result.error)


if __name__ == "__main__":
    unittest.main()
