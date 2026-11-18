"""Source/compile-only contracts: never launch the player or inspect devices."""
import pathlib
import platform
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/play_h3_local_av.swift"


class LocalPlayerContract(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE.read_text()

    def test_no_capture_or_global_route_mutation(self):
        for forbidden in ("AudioObjectSetPropertyData", "kAudioHardwarePropertyDefault",
                          "AVAudioSession", "AVCapture", "AVAudioEngine", "BlackHole",
                          "AVAudioPlayer", "Process()"):
            self.assertNotIn(forbidden, self.source)
        self.assertEqual(self.source.count("let player = AVPlayer()"), 1)
        self.assertIn("view.player = player", self.source)
        self.assertIn(".resizeAspect", self.source)

    def test_fail_closed_device_contract(self):
        for required in ("kAudioHardwarePropertyDevices", "kAudioDevicePropertyDeviceUID",
                         "kAudioDeviceTransportTypeBuiltIn", "kAudioDevicePropertyScopeOutput",
                         "kAudioDevicePropertyStreams", "kAudioDevicePropertyDeviceIsAlive",
                         'config.output_uid == "BuiltInSpeakerDevice"',
                         "player.audioOutputDeviceUniqueID = config.output_uid",
                         "player.audioOutputDeviceUniqueID == config.output_uid"):
            self.assertIn(required, self.source)
        self.assertLess(self.source.index("guard builtInOutputExists()"),
                        self.source.index("let player = AVPlayer()"))

    def test_protocol_and_lifecycle(self):
        for required in ("CommandLine.arguments.count == 1", "bytes.count < 65536",
                         "config.max_seconds <= 240", "(1...65535).contains(port)",
                         r"/index\.m3u8$", "FileHandle.standardOutput.write",
                         '"local_player_ready"', '"local_player_telemetry"',
                         '"local_player_done"', '"position_seconds"',
                         "terminal ? 254 : 253", "@MainActor", "SIGTERM",
                         "windowWillClose", "timer?.invalidate()", "Darwin.exit(code)",
                         "AVPlayerItemDidPlayToEndTime", "player?.replaceCurrentItem(with: nil)"):
            self.assertIn(required, self.source)
        self.assertLess(self.source.index('emit(["type": "local_player_ready"'),
                        self.source.index("beginAvailabilityProbe()"))
        self.assertNotIn("localizedDescription", self.source)
        self.assertNotIn("print(", self.source)

    def test_availability_and_completion_contract(self):
        for required in ("URLSessionConfiguration.ephemeral", "connectionProxyDictionary = [:]",
                         "http.statusCode == 200", 'map(playlistCanStart)',
                         "Task.sleep(for: .milliseconds(150))", "try await session.data(from: url)",
                         "probeTask?.cancel()", "probeSession?.invalidateAndCancel()",
                         "automaticallyWaitsToMinimizeStalling = false",
                         "item.preferredForwardBufferDuration = 0", '"buffering"',
                         'once("awaiting_playlist")'):
            self.assertIn(required, self.source)
        self.assertLess(self.source.index("http.statusCode == 200"),
                        self.source.index("let item = AVPlayerItem(url: url)"))
        self.assertLess(self.source.index("Timer.scheduledTimer"),
                        self.source.index("beginAvailabilityProbe()"))
        self.assertIn('if code == 0 && name == "ended" { emit(["type": "local_player_done"]) }', self.source)
        self.assertNotIn('"player_event"', self.source)
        self.assertNotIn("media_seconds", self.source)

    @unittest.skipUnless(platform.system() == "Darwin" and shutil.which("xcrun"),
                         "macOS Xcode toolchain required")
    def test_pure_live_startup_gate_without_devices(self):
        helper=self.source.split('func playlistCanStart(',1)[1].split('// Read-only CoreAudio',1)[0]
        harness='import Foundation\nfunc playlistCanStart('+helper+'''\n@main struct Test {
            static func main() {
                func manifest(_ count: Int, _ ended: Bool) -> String {
                    var text = "#EXTM3U\\n#EXT-X-TARGETDURATION:1\\n"
                    for i in 0..<count { text += String(format: "#EXTINF:1.000,\\nseg-%03d.ts\\n", i) }
                    if ended { text += "#EXT-X-ENDLIST\\n" }
                    return text
                }
                precondition(!playlistCanStart(manifest(1, false)))
                precondition(!playlistCanStart(manifest(2, false)))
                precondition(playlistCanStart(manifest(3, false)))
                precondition(playlistCanStart(manifest(2, true)))
                precondition(!playlistCanStart("#EXTM3U\\n"))
            }
        }
        '''
        with tempfile.TemporaryDirectory() as directory:
            source=pathlib.Path(directory)/'gate.swift';source.write_text(harness)
            binary=pathlib.Path(directory)/'gate'
            subprocess.run(['xcrun','swiftc','-swift-version','6','-parse-as-library',str(source),'-o',str(binary)],check=True,capture_output=True,timeout=60)
            subprocess.run([str(binary)],check=True,capture_output=True,timeout=5)

    @unittest.skipUnless(platform.system() == "Darwin" and shutil.which("xcrun"),
                         "macOS Xcode toolchain required")
    def test_swift6_compile_only(self):
        probe = subprocess.run(["xcrun", "--find", "swiftc"], capture_output=True)
        if probe.returncode:
            self.skipTest("Swift compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="h3-local-av-compile-") as directory:
            result = subprocess.run([
                "xcrun", "swiftc", "-swift-version", "6", "-strict-concurrency=complete",
                "-warnings-as-errors", "-parse-as-library", str(SOURCE),
                "-o", str(pathlib.Path(directory) / "player")],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            # Intentionally no execution, even with invalid input.


if __name__ == "__main__":
    unittest.main()
