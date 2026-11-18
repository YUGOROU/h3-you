"""Synthetic saved-config fixtures only: no installed OBS/device access."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.h3_personal.obs_delivery_audit import audit, audit_files, main
from research.prepare_h3_obs_generated import AUDIO_KEYS, collection


SOURCE = "Generated UDP"
PROFILE = """[Audio]
MonitoringDeviceId=BlackHole2ch_UID
MonitoringDeviceName=BlackHole 2ch
[Video]
BaseCX=1920
BaseCY=1080
OutputCX=1920
OutputCY=1080
"""


def scene():
    # Reuse the generated-only scene structure without calling its preparer.
    data = collection("synthetic-unused.mp4")
    source = data["sources"][0]
    source.update(name=SOURCE, muted=False, volume=1.0, monitoring_type=1)
    source["settings"] = dict(is_local_file=False,
                              input="udp://127.0.0.1:59513?fifo_size=1000000")
    return data


def codes(report):
    return {f["code"]: f for f in report["findings"]}


class OBSDeliveryAuditTests(unittest.TestCase):
    def test_good_saved_route_is_not_runtime_or_remote_proof(self):
        report = audit(PROFILE, scene(), SOURCE)
        self.assertTrue(report["recorded_config_only"])
        self.assertTrue(report["read_only"])
        self.assertFalse(report["runtime_verified"])
        self.assertFalse(report["remote_verified"])
        self.assertEqual(report["monitoring_mode"], "monitor_only")
        self.assertFalse(any(f["severity"] == "error" for f in report["findings"]))
        self.assertFalse(report["source_contract"]["measured"])
        self.assertEqual(report["source_contract"]["width"], 252)

    def test_device_must_be_explicit(self):
        for replacement in ("", "MonitoringDeviceId=deviceDefault\n",
                            "MonitoringDeviceId=default\n", "MonitoringDeviceId=Other_UID\n"):
            with self.subTest(replacement=replacement):
                profile = PROFILE.replace("MonitoringDeviceId=BlackHole2ch_UID\n", replacement)
                self.assertEqual(codes(audit(profile, scene(), SOURCE))["monitor_device"]["severity"], "error")
        inherited = "[DEFAULT]\nMonitoringDeviceId=BlackHole2ch_UID\n" + PROFILE.replace("MonitoringDeviceId=BlackHole2ch_UID\n", "")
        self.assertEqual(codes(audit(inherited, scene(), SOURCE))["monitor_device"]["severity"], "error")

    def test_duplicate_monitor_keys_fail_closed(self):
        variants = (
            "MonitoringDeviceId = BlackHole2ch_UID\nMonitoringDeviceId=default\n",
            "MonitoringDeviceId=BlackHole2ch_UID\nmonitoringDEVICEid : private-route\n",
            "MonitoringDeviceId=BlackHole2ch_UID\n[Other]\nMonitoringDeviceId=default\n",
            "MonitoringDeviceId=BlackHole2ch_UID\n[audio]\nMonitoringDeviceId=BlackHole2ch_UID\n",
            "MonitoringDeviceId=BlackHole2ch_UID\nMonitoringDeviceName=private-route\n",
        )
        for replacement in variants:
            with self.subTest(replacement=replacement):
                profile = PROFILE.replace("MonitoringDeviceId=BlackHole2ch_UID\n", replacement)
                report = audit(profile, scene(), SOURCE)
                findings = codes(report)
                self.assertEqual(findings["profile_monitor_conflict"]["severity"], "error")
                self.assertEqual(findings["monitor_device"]["severity"], "error")
                self.assertIn("ambiguous", findings["monitor_device"]["message"])
                self.assertNotIn("missing", findings["monitor_device"]["message"])
                self.assertNotIn("private-route", json.dumps(report))

    def test_exact_monitor_keys_only(self):
        profile = PROFILE + "[Other]\nMonitoringDeviceIdBackup=default\n; MonitoringDeviceId=default\n# MonitoringDeviceName=default\n"
        self.assertNotIn("profile_monitor_conflict", codes(audit(profile, scene(), SOURCE)))

    def test_obs_monitor_enum(self):
        for mode, label, severity in ((0, "none", "error"), (1, "monitor_only", "info"),
                                      (2, "monitor_and_output", "info")):
            with self.subTest(mode=mode):
                data = scene()
                data["sources"][0]["monitoring_type"] = mode
                report = audit(PROFILE, data, SOURCE)
                self.assertEqual(report["monitoring_mode"], label)
                self.assertEqual(codes(report)["source_monitoring"]["severity"], severity)
        for mode in (None, True, "1", 3):
            data = scene()
            data["sources"][0]["monitoring_type"] = mode
            self.assertEqual(codes(audit(PROFILE, data, SOURCE))["source_monitoring"]["severity"], "error")

    def test_mute_volume_and_missing_values(self):
        for field, values, code in (("muted", (True, None, 0), "source_mute"),
                                     ("volume", (0, -1, None, "1", True, float("nan"), float("inf")), "source_volume")):
            for value in values:
                with self.subTest(field=field, value=value):
                    data = scene()
                    data["sources"][0][field] = value
                    self.assertEqual(codes(audit(PROFILE, data, SOURCE))[code]["severity"], "error")
            data = scene()
            del data["sources"][0][field]
            self.assertEqual(codes(audit(PROFILE, data, SOURCE))[code]["severity"], "error")

    def test_global_audio_absent_null_vs_configured(self):
        data = scene()
        for key in AUDIO_KEYS:
            data.pop(key)
        self.assertEqual(codes(audit(PROFILE, data, SOURCE))["global_audio_capture"]["severity"], "info")
        for key in AUDIO_KEYS:
            data = scene()
            data[key] = {}  # Even an empty object is not the null/absent contract.
            self.assertEqual(codes(audit(PROFILE, data, SOURCE))["global_audio_capture"]["severity"], "error")

    def test_explicit_capture_source(self):
        data = scene()
        data["sources"].append(dict(id="coreaudio_input_capture"))
        self.assertEqual(codes(audit(PROFILE, data, SOURCE))["explicit_audio_capture"]["severity"], "error")

    def test_portrait_sidebars_are_not_wrong_model_aspect(self):
        report = audit(PROFILE, scene(), SOURCE)
        findings = codes(report)
        self.assertEqual(findings["source_aspect"]["severity"], "info")
        for key in ("canvas_geometry", "output_geometry"):
            self.assertEqual(findings[key]["severity"], "info")
            self.assertIn("sidebars do not imply a landscape model source", findings[key]["message"])
            self.assertIn("neither verified here", findings[key]["message"])
        portrait = PROFILE.replace("1920", "252").replace("1080", "448")
        message = codes(audit(portrait, scene(), SOURCE))["canvas_geometry"]["message"]
        self.assertIn("requires 16:9 input", message)
        self.assertIn("unverified", message)
        odd = PROFILE.replace("1920", "1080")
        self.assertEqual(codes(audit(odd, scene(), SOURCE))["canvas_geometry"]["severity"], "warning")

    def test_missing_duplicate_wrong_type_and_non_udp_sources(self):
        data = scene()
        self.assertEqual(codes(audit(PROFILE, data, "missing"))["generated_source"]["severity"], "error")
        data["sources"].append(copy.deepcopy(data["sources"][0]))
        self.assertEqual(codes(audit(PROFILE, data, SOURCE))["generated_source"]["severity"], "error")
        for settings in ({}, {"is_local_file": True, "input": "udp://localhost:1234"},
                         {"is_local_file": False, "input": "https://example.invalid/clip"}):
            data = scene()
            data["sources"][0]["settings"] = settings
            self.assertEqual(codes(audit(PROFILE, data, SOURCE))["generated_udp"]["severity"], "error")
        data["sources"][0]["id"] = "browser_source"
        self.assertEqual(codes(audit(PROFILE, data, SOURCE))["generated_source"]["severity"], "error")

    def test_no_mutation_or_arbitrary_settings_in_output(self):
        secret = "DO_NOT_DISCLOSE_PRIVATE_VALUE"
        data = scene()
        data["sources"][0]["settings"].update(password=secret, input="udp://user:" + secret + "@127.0.0.1:1234")
        data["modules"] = {"websocket": {"password": secret}}
        original = copy.deepcopy(data)
        profile = PROFILE + "[Private]\nPassword=" + secret + "\n"
        with patch("pathlib.Path.read_text", side_effect=AssertionError("pure API must not read")):
            report = audit(profile, data, SOURCE)
        self.assertEqual(data, original)
        self.assertNotIn(secret, json.dumps(report))
        self.assertNotIn("udp://", json.dumps(report))

    def test_explicit_files_cli_only_reads_inputs_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as root:
            profile = Path(root) / "basic.ini"
            scenes = Path(root) / "scene.json"
            unrelated = Path(root) / "websocket-secret.json"
            profile.write_text(PROFILE)
            scenes.write_text(json.dumps(scene()))
            unrelated.write_text("never read")
            before = {p.name: p.read_bytes() for p in Path(root).iterdir()}
            original = Path.read_text
            reads = []

            def read_only(path, *args, **kwargs):
                self.assertIn(path, (profile, scenes))
                reads.append(path)
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", read_only), patch.object(Path, "write_text", side_effect=AssertionError("no write")):
                report = audit_files(profile, scenes, SOURCE)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    main(["--profile", str(profile), "--scene", str(scenes), "--source-name", SOURCE])
            self.assertEqual(json.loads(output.getvalue()), report)
            self.assertEqual(reads, [profile, scenes, profile, scenes])
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(root).iterdir()})

    def test_invalid_inputs_are_sanitized(self):
        report = audit("secret malformed INI", [], SOURCE)
        self.assertIn("profile_invalid", codes(report))
        self.assertIn("scene_invalid", codes(report))
        self.assertNotIn("secret", json.dumps(report))
        for sources in (None, {}, [None]):
            self.assertIn("sources_invalid", codes(audit(PROFILE, {"sources": sources}, SOURCE)))
        with tempfile.TemporaryDirectory() as root:
            profile = Path(root) / "profile.ini"
            scenes = Path(root) / "scene.json"
            profile.write_text(PROFILE)
            scenes.write_text("private malformed JSON")
            report = audit_files(profile, scenes, SOURCE)
            self.assertIn("input_unreadable", codes(report))
            self.assertNotIn("private", json.dumps(report))
            self.assertNotIn(root, json.dumps(report))


if __name__ == "__main__":
    unittest.main()
