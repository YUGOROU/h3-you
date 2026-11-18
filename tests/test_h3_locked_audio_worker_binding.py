"""Persistent workers must observe per-turn bindings through atomic shared files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from research.h3_personal.external_audio import plan_windows,window_manifest

class WorkerBindingTests(unittest.TestCase):
    def test_spawned_process_observes_later_window_without_environment_update(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'window.json'
            plan=plan_windows(11.64,wav='/generated/reply.wav',sha256_value='a'*64)
            first=json.dumps(window_manifest(plan,0));path.write_text(first)
            env=dict(os.environ,H3_SMD_LOCKED_AUDIO_WINDOW=first,H3_SMD_LOCKED_AUDIO_WINDOW_FILE=str(path))
            code="from research.h3_personal.external_audio import load_manifest;print('ready',flush=True);input();print(load_manifest()['index'],flush=True)"
            p=subprocess.Popen([sys.executable,'-c',code],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                self.assertEqual(p.stdout.readline().strip(),'ready')
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(window_manifest(plan,1)));tmp.replace(path)
                stdout,stderr=p.communicate('\n',timeout=10)
                self.assertEqual(p.returncode,0,stderr);self.assertEqual(stdout.strip(),'1')
            finally:
                if p.poll() is None:p.kill();p.wait()

if __name__=='__main__':unittest.main()
