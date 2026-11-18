import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch,Mock
from research.run_h3_realtime_first import encode_command,validate_packets,shift_fragment_decode_times,encode_rgb_segment,encoder_preflight
from research.h3_personal.realtime_segments import segment_timing

class RealtimeConfigRoundtripTests(unittest.TestCase):
    def test_json_config_roundtrip(self):
        from dataclasses import asdict
        from research.h3_personal.realtime_first import RealtimeFirstConfig,FASTVIDEO_SHA
        config=RealtimeFirstConfig(FASTVIDEO_SHA,'a'*40,'b'*40)
        self.assertEqual(RealtimeFirstConfig.from_dict(json.loads(json.dumps(asdict(config)))),config)

class RealtimeEncoderRegressionTests(unittest.TestCase):
    def test_ffmpeg44_compatible_and_thread_bounded(self):
        command=encode_command(Path('test.mp4'),segment_timing(0))
        self.assertNotIn('-fps_mode',command)
        self.assertEqual(command[command.index('-vsync')+1],'0')
        self.assertEqual(command[command.index('-threads')+1],'2')
        self.assertEqual(command[command.index('-filter_threads')+1],'1')

    def test_failure_contains_stderr_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            def failure(*args,**kwargs):
                process=Mock(returncode=1)
                process.communicate.side_effect=lambda *a,**k:kwargs['stderr'].write(b'Unrecognized option fps_mode')
                return process
            with patch('research.run_h3_realtime_first.subprocess.Popen',side_effect=failure):
                with self.assertRaisesRegex(RuntimeError,'exit=1.*Unrecognized option fps_mode'):
                    encode_rgb_segment(b'',Path(directory)/'failed.mp4',segment_timing(0))

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class RealtimeRunnerTests(unittest.TestCase):
    def test_early_probe_runs_shared_production_path(self):
        with tempfile.TemporaryDirectory() as directory:
            result=encoder_preflight(Path(directory)/'probe')
            self.assertTrue(result['encode_packet_pts_full_decode'])
            self.assertEqual(result['source'],'shared_production_encoder')

    def test_fragmented_segment_actual_packet_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in (0,1,2):
                timing=segment_timing(index);path=Path(directory)/f'{index}.mp4'
                subprocess.run(encode_command(path,timing,width=32,height=32),
                    input=bytes(32*32*3*timing.newly_emitted_frames),check=True,timeout=15)
                shift_fragment_decode_times(path,timing);validate_packets(path,timing)
                subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,timeout=10)
                self.assertIn(b'moof',path.read_bytes())
