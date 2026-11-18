import ast
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research import launch_h3_fast_ref2va as launch

class FastRefGPUDeploymentTests(unittest.TestCase):
    def test_wrapper_selects_separate_inspector_bootstrap_and_user_request(self):
        with patch.object(launch,'launch',return_value=0) as submit:
            self.assertEqual(launch.main(['--stage','/private/fresh','--ledger','/private/ledger']),0)
        args=submit.call_args.kwargs
        self.assertIs(args['inspect_fn'],launch.inspect)
        self.assertEqual(args['bootstrap_entry'],'research/bootstrap_h3_fast_ref2va.py')
        self.assertIn('USER gpu検証までgo',args['approval_request'])

    def test_gpu_output_is_outside_deployed_repository(self):
        from research.bootstrap_h3_fast_ref2va import output_directory
        root=Path('/work/fast-ref2va');output=output_directory(root)
        self.assertEqual(output,Path('/work/fast-ref2va-generated'))
        self.assertFalse(output.is_relative_to(root))

    def test_generated_output_only_and_no_installs(self):
        from research import bootstrap_h3_fast_ref2va as b
        source=Path(b.__file__).read_text();ast.parse(source)
        self.assertIn("names=['profile.json']+[f'segment-{i:03d}.mp4' for i in range(4)]",source)
        self.assertNotIn('pip install',source);self.assertNotIn('upload_folder',source)
        self.assertIn("asset['role'] in ('personal','reference')",source)
        self.assertIn("subprocess.run(command,cwd=root)",source)
        self.assertIn("'-xerror'",source)

    def test_real_ref_timing_encoder_pts_and_full_decode(self):
        import shutil,subprocess
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):self.skipTest('FFmpeg required')
        from research.h3_personal.fast_ref2va import segment_timing
        from research.run_h3_realtime_first import encode_rgb_segment
        with tempfile.TemporaryDirectory() as folder:
            for i in (0,1):
                p=Path(folder)/f'{i}.mp4';t=segment_timing(i)
                encode_rgb_segment(bytes(32*32*3*124),p,t,width=32,height=32)
                subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True,capture_output=True,timeout=20)
                self.assertEqual(t.newly_emitted_frames,124)
