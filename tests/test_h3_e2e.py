"""CPU-only contract tests. Mock LLM/H3/TTS outputs are NOT integration evidence."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs
from urllib.request import urlopen

from research.h3_personal import e2e


def options():
    return argparse.Namespace(llm_model="test-model", llm_url="http://127.0.0.1:8000/v1/chat/completions",
                              h3_url="http://127.0.0.1:8091/v1/videos/sync", h3_model="MiniMaxAI/MiniMax-H3",
                              timeout=10, allow_remote_personal_data=False, agent_instruction="Speak Japanese.",
                              reference_image="", reference_audio="", tts_model="openbmb/VoxCPM2")


class ContractTests(unittest.TestCase):
    def test_endpoint_consent(self):
        for url in ("http://127.0.0.1:8091/v1/videos/sync", "http://localhost:8000", "http://[::1]:9000"):
            e2e.check_endpoint(url)
        for url in ("https://example.com/v1/videos/sync", "http://192.168.1.1", "http://evil.localhost"):
            with self.assertRaisesRegex(ValueError, "allow-remote"):
                e2e.check_endpoint(url)
            e2e.check_endpoint(url, True)
        for url in ("file:///tmp/a", "http://user:pass@localhost", "https://example.com/#x"):
            with self.assertRaises(ValueError):
                e2e.check_endpoint(url, True)
        with self.assertRaisesRegex(ValueError, "Redirect refused"):
            e2e.NoRedirect().redirect_request(None, None, 307, "", {}, "http://remote")

    def test_duration_never_truncates(self):
        self.assertEqual(e2e.duration_for_tts(1), 5)
        for value in (5.1, 12, 14.99, 15):
            self.assertGreaterEqual(e2e.duration_for_tts(value), value)
            self.assertLessEqual(e2e.duration_for_tts(value), 15)
        for value in (0, -1, 15.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                e2e.duration_for_tts(value)

    def test_autonomous_agent_requests_history_not_fixed_completion(self):
        args, history = options(), []
        with patch.object(e2e, "request") as req:
            for i in range(3):
                req.return_value = (json.dumps({"choices": [{"finish_reason": "stop", "message":
                                         {"content": f"Mock-test utterance {i}"}}]}).encode(), {})
                text = e2e.agent_utterance(args, history, i)
                self.assertEqual(text, f"Mock-test utterance {i}")
            payload = json.loads(req.call_args.args[1])
            self.assertEqual(len(payload["messages"]), 6)
            self.assertIn("Mock-test utterance 1", str(payload))
            req.return_value = (b'{"choices":[{"finish_reason":"length","message":{"content":"partial"}}]}', {})
            with self.assertRaises(ValueError):
                e2e.agent_utterance(args, history, 3)

    def test_real_voxcpm_python_api_wiring_with_mock_model(self):
        model = Mock()
        model.tts_model.sample_rate = 48000
        model.generate.return_value = [0.1, 0.2]
        factory, sf = Mock(), Mock()
        factory.from_pretrained.return_value = model
        with patch.dict("sys.modules", {"voxcpm": types.SimpleNamespace(VoxCPM=factory), "soundfile": sf}):
            tts = e2e.LocalVoxCPM2("openbmb/VoxCPM2", Path("voice.wav"))
            tts.synthesize("mock-test speech", Path("tts.wav"))
        factory.from_pretrained.assert_called_once_with("openbmb/VoxCPM2", load_denoiser=False)
        self.assertEqual(model.generate.call_args.kwargs["reference_wav_path"], "voice.wav")
        sf.write.assert_called_once_with("tts.wav", [0.1, 0.2], 48000, subtype="PCM_16")

    def test_h3_form_data_uris_and_raw_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio, output = Path(tmp) / "audio.wav", Path(tmp) / "h3.mp4"
            audio.write_bytes(b"mock-test WAV placeholder")
            with patch.object(e2e, "request", return_value=(b"mock-test MP4 placeholder",
                                      {"Content-Type": "video/mp4", "X-Inference-Time-S": "3.4"})) as req:
                timing = e2e.generate_h3(options(), "data:image/png;base64,eA==", audio, 5, "speech", output, 0)
                fields = parse_qs(req.call_args.args[1].decode())
                self.assertTrue(req.call_args.args[0].endswith("/v1/videos/sync"))
                self.assertEqual(req.call_args.args[2], "application/x-www-form-urlencoded")
                self.assertEqual(json.loads(fields["extra_params"][0])["audio_mode"], "lock_source")
                self.assertEqual(json.loads(fields["extra_params"][0])["task"], "ref2va")
                self.assertTrue(json.loads(fields["audio_reference"][0])["audio_url"].startswith("data:audio/wav;base64,"))
                self.assertIn("image_reference", fields)
                self.assertNotIn("input_reference", fields)
                self.assertNotIn("lora", fields)
                self.assertEqual(timing["server_seconds"], "3.4")
            with patch.object(e2e, "request", return_value=(b'{"url":"remote"}', {"Content-Type": "application/json"})):
                with self.assertRaisesRegex(ValueError, "raw video/mp4"):
                    e2e.generate_h3(options(), "data:image/png;base64,eA==", audio, 5, "speech", output, 0)

    def test_mux_rejects_inadequate_video_and_maps_original(self):
        with patch.object(e2e, "probe", return_value={"streams": [{"codec_type": "video", "duration": "5"}]}), \
             patch.object(e2e, "run_command") as command, patch.object(e2e, "validate_av"):
            with self.assertRaises(ValueError):
                e2e.mux_original("h3.mp4", "original.wav", "out.mp4", 6, 6)
            command.assert_not_called()
            e2e.mux_original("h3.mp4", "original.wav", "out.mp4", 5, 4)
            args = command.call_args.args[0]
            self.assertIn("1:a:0", args)
            self.assertNotIn("-shortest", args)
            self.assertEqual(args[args.index("-t") + 1], "5")

    def test_av_duration_and_start_validation(self):
        for streams in ([{"codec_type": "video", "duration": "5"}],
                        [{"codec_type": "video", "duration": "5"}, {"codec_type": "audio", "duration": "4"}],
                        [{"codec_type": "video", "duration": "5", "start_time": "1"},
                         {"codec_type": "audio", "duration": "5"}]):
            with patch.object(e2e, "probe", return_value={"streams": streams}):
                with self.assertRaises(ValueError):
                    e2e.validate_av("test.mp4", 5)

    def test_mock_orchestrator_publishes_three_then_concat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = options()
            image = root / "image.png"
            image.write_bytes(b"mock-test image placeholder")
            args.reference_image, args.reference_audio = str(image), "reference.wav"
            status = {"state": "starting", "segments": []}
            publications = []
            def publish(root, state):
                publications.append((state["state"], len(state["segments"])))
            with patch.object(e2e, "LocalVoxCPM2"), \
                 patch.object(e2e, "agent_utterance", side_effect=["mock one", "mock two", "mock three"]) as agent, \
                 patch.object(e2e, "probe", return_value={"streams": [{"codec_type": "audio", "duration": "4"}]}), \
                 patch.object(e2e, "generate_h3", return_value={"client_seconds": 1}), \
                 patch.object(e2e, "mux_original", return_value=[5, 5]), \
                 patch.object(e2e, "run_command") as command, \
                 patch.object(e2e, "validate_av") as validate, \
                 patch.object(e2e, "save_status", side_effect=publish):
                e2e.orchestrate(args, root, status)
            self.assertEqual(agent.call_count, 3)
            self.assertEqual(status["state"], "complete")
            self.assertIn(("generating turn 1", 1), publications)
            self.assertIn(("generating turn 2", 2), publications)
            self.assertEqual(command.call_args.args[0][-1], str(root / "demo.mp4"))
            validate.assert_called_once_with(root / "demo.mp4", 15, tolerance=0.3)

    def test_failure_keeps_completed_segments_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = options()
            args.reference_image = str(root / "image.png")
            Path(args.reference_image).write_bytes(b"mock placeholder")
            status = {"segments": [{"file": "segment-0.mp4"}]}
            with patch.object(e2e, "LocalVoxCPM2", side_effect=RuntimeError("test failure")):
                with self.assertRaises(RuntimeError):
                    e2e.orchestrate(args, root, status)
            saved = json.loads((root / "status.json").read_text())
            self.assertEqual(saved["state"], "failed")
            self.assertEqual(len(saved["segments"]), 1)

    def test_local_browser_http_poll_and_asset_allowlist(self):
        import functools
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.html").write_text(e2e.HTML)
            (root / "tts-0.wav").write_bytes(b"private test placeholder")
            e2e.save_status(root, {"state": "generating", "segments": [{"file": "segment-0.mp4"}]})
            server = e2e.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(e2e.DemoHandler, directory=tmp))
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url + "/status.json") as response:
                    self.assertEqual(json.load(response)["state"], "generating")
                with urlopen(url + "/") as response:
                    html = response.read().decode()
                    self.assertIn("setTimeout(poll,1000)", html)
                    self.assertIn("v.onended", html)
                with self.assertRaises(HTTPError) as caught:
                    urlopen(url + "/tts-0.wav")
                caught.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class LocalMediaTests(unittest.TestCase):
    def test_actual_ffmpeg_mux_and_concat_synthetic_placeholders(self):
        """Real CPU media validation; test color/sine fixtures, NOT H3/TTS evidence."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio, video = root / "original.wav", root / "video.mp4"
            e2e.run_command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                             "sine=frequency=440:sample_rate=48000", "-t", "4", str(audio)])
            e2e.run_command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                             "color=c=blue:s=160x96:r=24", "-t", "5", "-c:v", "libx264", str(video)])
            for i in range(3):
                e2e.mux_original(video, audio, root / f"segment-{i}.mp4", 5, 4)
            listing = root / "concat.txt"
            listing.write_text("".join(f"file 'segment-{i}.mp4'\n" for i in range(3)))
            demo = root / "demo.mp4"
            e2e.run_command(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "1", "-i", str(listing),
                             "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(demo)])
            e2e.validate_av(demo, 15, tolerance=0.3)


if __name__ == "__main__":
    unittest.main()
