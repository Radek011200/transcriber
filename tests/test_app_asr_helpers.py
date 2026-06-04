import unittest
from unittest import mock
import sys

import numpy as np

sys.modules.setdefault("pyaudio", mock.Mock(paInt16=8))
torch_stub = mock.Mock()
torch_stub.float16 = object()
torch_stub.cuda.is_available.return_value = False
torch_stub.cuda.get_device_name.return_value = "Test CUDA"
torch_stub.version.cuda = "test"
torch_stub.version.hip = None
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("transformers", mock.Mock(pipeline=mock.Mock()))
import app


class AsrHelperTest(unittest.TestCase):
    def test_default_num_beams_remains_high_quality(self):
        self.assertEqual(app.DEFAULT_NUM_BEAMS, 5)
        self.assertEqual(app.DEFAULT_ASR_QUALITY_PROFILE, "high")

    def test_cuda_pipeline_uses_float16(self):
        kwargs = app.build_asr_pipeline_kwargs("openai/whisper-small", cuda_available=True)

        self.assertEqual(kwargs["device"], 0)
        self.assertIs(kwargs["torch_dtype"], app.torch.float16)

    def test_cpu_pipeline_does_not_force_float16(self):
        kwargs = app.build_asr_pipeline_kwargs("openai/whisper-small", cuda_available=False)

        self.assertEqual(kwargs["device"], -1)
        self.assertNotIn("torch_dtype", kwargs)

    def test_audio_frames_convert_to_asr_input(self):
        samples = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2")
        audio_input, metrics = app.audio_frames_to_asr_input([samples.tobytes()], trim_silence_enabled=False)

        self.assertEqual(audio_input["sampling_rate"], app.RATE)
        self.assertEqual(audio_input["array"].dtype, np.float32)
        self.assertAlmostEqual(float(audio_input["array"][1]), 0.5, places=4)
        self.assertAlmostEqual(float(audio_input["array"][2]), -0.5, places=4)
        self.assertEqual(metrics["original_duration_seconds"], round(len(samples) / app.RATE, 2))

    def test_detect_acceleration_profile_cpu(self):
        app.torch.cuda.is_available.return_value = False

        detected = app.detect_acceleration_profile()

        self.assertEqual(detected["profile"], "cpu")

    def test_detect_acceleration_profile_amd_hip(self):
        app.torch.cuda.is_available.return_value = True
        app.torch.version.cuda = None
        app.torch.version.hip = "6.0"
        app.torch.cuda.get_device_name.return_value = "AMD Radeon"

        detected = app.detect_acceleration_profile()

        self.assertEqual(detected["profile"], "amd_gpu")

    def test_cpu_defaults_prioritize_fast_transcription(self):
        defaults = app.recommended_defaults_for_profile("cpu")

        self.assertEqual(defaults["selected_model"], app.DEFAULT_CPU_MODEL_NAME)
        self.assertEqual(defaults["num_beams"], 1)
        self.assertEqual(defaults["correction_mode"], "no_llm")


if __name__ == "__main__":
    unittest.main()
