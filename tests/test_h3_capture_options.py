"""Offline only: compile Swift 6; every executable invocation includes --describe."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/stream_h3_comet_pcm.swift"


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("swiftc"), "requires macOS Swift")
class CaptureOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="h3-capture-options-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.binary = str(Path(cls.temp.name) / "capture")
        subprocess.run(["swiftc", "-swift-version", "6", str(SOURCE), "-o", cls.binary],
                       check=True, capture_output=True, text=True, timeout=120)

    def describe(self, *args, valid=True):
        result = subprocess.run([self.binary, "--describe", *args],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0 if valid else 2, result.stderr)
        self.assertEqual(result.stdout, "", "stdout reserved for raw PCM")
        value = json.loads(result.stderr)
        if valid:
            self.assertFalse(value["captured"])
            self.assertFalse(value["enumerated"])
            self.assertFalse(value["runtime_verified"])
        return value

    def test_comet_defaults_and_legacy_modes(self):
        for args in [(), ("--check",), ("--stream",)]:
            with self.subTest(args=args):
                config = self.describe(*args)
                self.assertEqual(config["app"], "comet")
                self.assertEqual(config["bundle_id"], "ai.perplexity.comet")
                self.assertEqual(config["mute_behavior"], "unmuted")
                self.assertEqual(config["stream_requested"], "--stream" in args)
                self.assertEqual((config["encoding"], config["sample_rate"],
                                  config["channels"], config["max_seconds"]),
                                 ("PCM16LE", 16000, 1, 45))

    def test_app_and_mute(self):
        for app, bundle in [("comet", "ai.perplexity.comet"), ("discord", "com.hnc.Discord")]:
            config = self.describe("--mute-while-capturing", "--app", app, "--stream")
            self.assertEqual(config["bundle_id"], bundle)
            self.assertEqual(config["bundle_match"], "exact or dot descendants")
            self.assertEqual(config["mute_behavior"], "mutedWhenTapped")
            self.assertIn("not participant isolation", config["warning"])
            self.assertIn("muted only during capture", config["warning"])
        config = self.describe("--app", "discord", "--check")
        self.assertIn("candidate, not runtime verified", config["warning"])
        self.assertEqual(config["mute_behavior"], "unmuted")

    def test_invalid_options(self):
        cases = [
            ("--app",), ("--app", "Discord"), ("--app", "other"),
            ("--app=discord",), ("--unknown",), ("discord",),
            ("--check", "--stream"), ("--mute-while-capturing",),
            ("--check", "--mute-while-capturing"),
            ("--describe",), ("--check", "--check"), ("--stream", "--stream"),
            ("--app", "comet", "--app", "discord"),
            ("--stream", "--mute-while-capturing", "--mute-while-capturing"),
        ]
        for args in cases:
            with self.subTest(args=args):
                self.assertIn("error", self.describe(*args, valid=False))

    def test_all_drift_checks_use_selected_app(self):
        source = SOURCE.read_text()
        self.assertNotIn("cometProcesses", source)
        self.assertEqual(source.count("guard try selectedProcesses(app) == processes"), 4)
        self.assertIn('value == bundle || value.hasPrefix(bundle + ".")', source)
        self.assertLess(source.index("if options.describe {"),
                        source.index("let processes = try selectedProcesses(app)"))
        self.assertIn("options.mute ? .mutedWhenTapped : .unmuted", source)


if __name__ == "__main__":
    unittest.main()
