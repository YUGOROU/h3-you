"""No capture/permission prompts: strict compile, metadata, invalid arguments only."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

@unittest.skipUnless(sys.platform=='darwin' and shutil.which('swiftc'),'macOS Speech SDK required')
class LocalSpeechTests(unittest.TestCase):
    def test_metadata_does_not_capture_or_authorize(self):
        with tempfile.TemporaryDirectory() as folder:
            binary=str(Path(folder)/'speech')
            subprocess.run(['swiftc','-swift-version','6','-strict-concurrency=complete','research/listen_h3_speech.swift',
                            '-Xlinker','-sectcreate','-Xlinker','__TEXT','-Xlinker','__info_plist','-Xlinker','research/H3SpeechInfo.plist','-o',binary],check=True,capture_output=True,timeout=60)
            p=subprocess.run([binary,'--check'],check=True,capture_output=True,text=True,timeout=15)
            value=json.loads(p.stdout)
            self.assertIs(value['captured'],False);self.assertIs(value['cloud_fallback'],False)
            p=subprocess.run([binary,'--bad-argument'],capture_output=True,text=True,timeout=5)
            self.assertEqual(p.returncode,2)
    def test_on_device_and_explicit_builtin_only(self):
        code=Path('research/listen_h3_speech.swift').read_text()
        self.assertIn('request.requiresOnDeviceRecognition = true',code)
        self.assertIn('kAudioDeviceTransportTypeBuiltIn',code)
        self.assertNotIn('AudioObjectSetPropertyData',code)
        self.assertIn('session.stopRunning(); queue.sync {}; state.endAudio()',code)
        for field in ('"buffers"', '"frames"', '"peak"', '"measured_samples"',
                      '"partial_callbacks"', '"error_domain"', '"error_code"'):
            self.assertIn(field, code)
        self.assertIn('diagnostics["session_running_after_start"] = runningAfterStart', code)
        self.assertIn('CMSampleBufferCopyPCMDataIntoAudioBufferList', code)
        self.assertNotIn('CMBlockBufferGetDataPointer', code)
        self.assertIn('AVLinearPCMIsFloatKey:true', code)
        self.assertIn('AVLinearPCMIsBigEndianKey:false', code)
        self.assertIn('frame * pcm.stride', code)
        self.assertIn('let task = meterOnly ? nil :', code)
        self.assertIn('"input_unique_id":input.device.uniqueID', code)

if __name__=='__main__':unittest.main()
