"""Offline selector receipts only; never enumerate processes or capture audio."""
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(platform.system() == "Darwin" and shutil.which("swiftc"),
                     "requires macOS Swift compiler")
class ZoomCaptureDescriptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="h3-zoom-description-", dir="/tmp")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = Path(cls.directory.name) / "selector"
        source = Path(__file__).resolve().parents[1] / "research/stream_h3_comet_pcm.swift"
        result = subprocess.run(
            ["swiftc", "-swift-version", "6", "-strict-concurrency=complete",
             "-warnings-as-errors", str(source), "-o", str(cls.binary)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode:
            raise AssertionError(f"Swift compilation failed:\n{result.stdout}{result.stderr}")

    def invoke(self, *arguments):
        # A hard execution guard: no test can accidentally invoke the default
        # enumeration path, --check, or a live --stream invocation.
        self.assertIn("--describe", arguments)
        self.assertNotIn("--check", arguments)
        result = subprocess.run([str(self.binary), *arguments],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.stdout, "", "offline inspection must emit no PCM")
        receipts = [json.loads(line) for line in result.stderr.splitlines()]
        self.assertEqual(len(receipts), 1, result.stderr)
        return result.returncode, receipts[0]

    def assert_description(self, arguments, app, bundle, stream=False):
        code, receipt = self.invoke(*arguments)
        self.assertEqual(code, 0, receipt)
        self.assertEqual(receipt["type"], "description")
        self.assertEqual(receipt["app"], app)
        self.assertEqual(receipt["bundle_id"], bundle)
        self.assertEqual(receipt["bundle_match"], "exact or dot descendants")
        for flag in ("runtime_verified", "captured", "enumerated"):
            self.assertIs(receipt[flag], False)
        self.assertIs(receipt["stream_requested"], stream)
        self.assertEqual(receipt["mute_behavior"], "unmuted")
        self.assertIn("selected app remains unmuted", receipt["warning"])
        self.assertIn("app scope, not participant isolation", receipt["warning"])
        self.assertNotIn("processes", receipt)
        return receipt

    def test_default_remains_comet(self):
        default = self.assert_description(("--describe",), "comet", "ai.perplexity.comet")
        explicit = self.assert_description(("--describe", "--app", "comet"),
                                           "comet", "ai.perplexity.comet")
        self.assertEqual(default, explicit)
        self.assertIn("All Comet tabs", default["warning"])

    def test_explicit_app_candidates(self):
        for app, bundle in (("zoom", "us.zoom.xos"), ("discord", "com.hnc.Discord")):
            with self.subTest(app=app):
                receipt = self.assert_description(("--describe", "--app", app), app, bundle)
                self.assertIn("candidate, not runtime verified", receipt["warning"])

    def test_stream_describe_is_description_only_without_operations(self):
        for selection, app, bundle in (
            ((), "comet", "ai.perplexity.comet"),
            (("--app", "zoom"), "zoom", "us.zoom.xos"),
            (("--app", "discord"), "discord", "com.hnc.Discord"),
            (("--app", "comet"), "comet", "ai.perplexity.comet"),
        ):
            with self.subTest(app=app, selection=selection):
                self.assert_description(("--stream", "--describe", *selection),
                                        app, bundle, stream=True)

    def test_invalid_flags_fail_before_operations(self):
        cases = (
            (("--describe", "--app"), "--app requires comet|discord|zoom"),
            (("--describe", "--app", "Zoom"), "--app requires comet|discord|zoom"),
            (("--describe", "--app", "unknown"), "--app requires comet|discord|zoom"),
            (("--describe", "--bogus"), "unknown flag: --bogus"),
            (("--describe", "--describe"), "duplicate flag: --describe"),
            (("--describe", "--app", "zoom", "--app", "comet"), "duplicate flag: --app"),
            (("--describe", "--mute-while-capturing"), "--mute-while-capturing requires --stream"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                code, receipt = self.invoke(*arguments)
                self.assertEqual(code, 2)
                self.assertEqual(receipt, {"error": message})


if __name__ == "__main__":
    unittest.main()
