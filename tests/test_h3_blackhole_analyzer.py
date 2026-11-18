"""Compile/metadata only; never captures, changes routes, installs assets."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

@unittest.skipUnless(sys.platform == 'darwin' and shutil.which('swiftc'), 'macOS26 SDK required')
class AnalyzerTests(unittest.TestCase):
    def test_compile_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory)/'probe')
            subprocess.run(['swiftc','-swift-version','6','-strict-concurrency=complete',
                            'research/listen_h3_blackhole_analyzer.swift',
                            '-Xlinker','-sectcreate','-Xlinker','__TEXT','-Xlinker','__info_plist',
                            '-Xlinker','research/H3SpeechInfo.plist','-o',binary],
                           check=True,capture_output=True,timeout=60)
            result = subprocess.run([binary,'--check'],check=True,capture_output=True,text=True,timeout=20)
            self.assertIs(json.loads(result.stdout)['captured'],False)
            result = subprocess.run([binary,'--invalid'],capture_output=True,timeout=5)
            self.assertEqual(result.returncode,2)
    def test_bounded_route_and_callback_contract(self):
        text = Path('research/listen_h3_blackhole_analyzer.swift').read_text()
        for value in ('@Sendable (AVAudioPCMBuffer, AVAudioTime)', 'actual == device',
                      'AudioUnitSetProperty', '.bufferingOldest(64)', '.seconds(30)',
                      'finalizeAndFinishThroughEndOfInput', 'node.removeTap',
                      'peakSamples > 0 ? peak as Any : NSNull()'):
            self.assertIn(value,text)
        for value in ('AudioObjectSetPropertyData', 'downloadAndInstall', 'AVAudioFile',
                      'requestAccess', 'SFSpeechRecognizer('):
            self.assertNotIn(value,text)

if __name__ == '__main__':
    unittest.main()
