"""macOS metadata-only tests. Never play audio or read the generated speech fixture."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform=='darwin' and shutil.which('swiftc'),'macOS Swift SDK required')
class VirtualAudioTests(unittest.TestCase):
    def test_compile_check_and_reject_invalid_operation(self):
        with tempfile.TemporaryDirectory() as folder:
            binary=str(Path(folder)/'virtual-audio')
            subprocess.run(['swiftc','-swift-version','6','-strict-concurrency=complete','research/play_h3_virtual_audio.swift','-o',binary],check=True,capture_output=True,timeout=60)
            check=subprocess.run([binary,'--check'],check=True,capture_output=True,text=True,timeout=15)
            result=json.loads(check.stdout);self.assertFalse(result['played']);self.assertIsInstance(result['available'],bool)
            bad=subprocess.run([binary,'--record'],capture_output=True,text=True,timeout=15)
            self.assertEqual(bad.returncode,2)
            if not result['available']:
                # Fail before creating any audio engine or opening any media file.
                blocked=subprocess.run([binary,'--play'],capture_output=True,text=True,timeout=15)
                self.assertEqual(blocked.returncode,1);self.assertFalse(json.loads(blocked.stdout)['played'])

    def test_input_routing_helper_check_never_applies(self):
        with tempfile.TemporaryDirectory() as folder:
            binary=str(Path(folder)/'virtual-input')
            subprocess.run(['swiftc','-swift-version','6','-strict-concurrency=complete','research/set_h3_virtual_input.swift','-o',binary],check=True,capture_output=True,timeout=60)
            check=subprocess.run([binary,'--check'],capture_output=True,text=True,timeout=15)
            result=json.loads(check.stdout)
            self.assertIn(check.returncode,(0,1));self.assertFalse(result['changed'])
        source=Path('research/set_h3_virtual_input.swift').read_text()
        self.assertNotIn('inputNode',source)
        self.assertEqual(source.count('AudioObjectSetPropertyData('),1)
        self.assertIn('var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice',source)

    def test_no_capture_or_global_device_setter(self):
        source=Path('research/play_h3_virtual_audio.swift').read_text()
        self.assertNotIn('inputNode',source);self.assertNotIn('AudioObjectSetPropertyData',source)
        self.assertIn('setDeviceID(device)',source);self.assertIn('watermark_required',source)
        self.assertIn('SHA256.hash(data: bytes)',source)

if __name__=='__main__':unittest.main()
