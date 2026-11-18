"""Compile/negative-argument checks only: never opens any camera."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

@unittest.skipUnless(sys.platform=='darwin' and shutil.which('swiftc'),'macOS SDK required')
class VirtualCameraProbeTests(unittest.TestCase):
    def test_explicit_capture_flag_required(self):
        with tempfile.TemporaryDirectory() as folder:
            binary=str(Path(folder)/'probe')
            subprocess.run(['swiftc','-swift-version','6','-strict-concurrency=complete','research/probe_h3_virtual_camera.swift','-o',binary],check=True,capture_output=True,timeout=60)
            for args in ([],['--physical-camera'],['--probe-obs-only','extra']):
                result=subprocess.run([binary,*args],capture_output=True,text=True,timeout=5)
                self.assertEqual(result.returncode,2)
                self.assertIn('usage:',result.stdout)

if __name__=='__main__':unittest.main()
