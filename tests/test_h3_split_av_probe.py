"""Offline contracts only: never construct a real sink or open devices/UDP."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from research import run_h3_split_av_probe as probe
from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, encode
from research.h3_personal.paired_av_wire import envelope_record


class ProbeTests(unittest.TestCase):
    def fixture(self, root):
        pcm = b'\x01\x00' * 480000
        with wave.open(str(root / 'speech.wav'), 'wb') as w:
            w.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            w.writeframes(pcm)
        assets = dict(original_wav_sha256=probe.AUDIO_SHA, source_frame_start=634,
                      source_frame_stop=874, geometry=[252, 448], fps=24)
        binding = LiveAVBinding('split-av-probe-test', probe.sha(probe.canonical(assets)), 0,
                                max_segments=3, max_total_frames=240)
        envelopes = []
        for i, (start, stop) in enumerate(probe.RANGES):
            video = b'fake-video-' + bytes([i])
            (root / f'segment-{i}.mp4').write_bytes(video)
            part = pcm[start*4000:stop*4000]
            region = AudioRegion('speech', start*2000, stop*2000, 'original-pcm', 0,
                                 start*2000, probe.sha(part))
            e = encode(binding, i, start, stop, video, part, (region,), i == 2)
            envelopes.append(dict(schema='h3-paired-av-wire-v1', envelope=envelope_record(e)))
        m = dict(schema='h3-split-av-probe-v1', assets=assets, envelopes=envelopes,
                 speech_wav_sha256=probe.file_sha(root / 'speech.wav'), speech_pcm_sha256=probe.sha(pcm))
        body = probe.canonical(m)
        (root / 'manifest.json').write_bytes(body)
        (root / 'manifest.sha256').write_text(probe.sha(body))
        return m

    def test_metadata(self):
        s = dict(width=252, height=448, r_frame_rate='24/1', avg_frame_rate='24/1', nb_read_frames='874')
        probe.validate_video(dict(streams=[s]))
        for key, value in [('width', 448), ('nb_read_frames', '873'), ('avg_frame_rate', '25/1')]:
            with self.assertRaises(ValueError):
                probe.validate_video(dict(streams=[dict(s, **{key: value})]))
        self.assertEqual([b-a for a, b in probe.RANGES], [119, 102, 19])

    def test_inspect_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            _, _, binding, pairs = probe.inspect(root)
            self.assertEqual(binding.max_total_frames, 240)
            self.assertEqual(sum(len(x.audio_pcm) for _, x in pairs), 960000)
            (root / 'segment-1.mp4').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                probe.inspect(root)

    def test_manifest_seal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            with (root / 'manifest.json').open('ab') as f:
                f.write(b' ')
            with self.assertRaises(ValueError):
                probe.inspect(root)

    def test_approval_before_inspection(self):
        with patch.object(probe, 'inspect', side_effect=AssertionError('must not inspect')):
            with self.assertRaises(ValueError):
                probe.run('/missing', audio_binary='/missing')

    def test_cli_gates_and_offline_dispatch(self):
        for args in (['--run', '/missing'], ['--run', '/missing', '--audio-binary', '/missing']):
            with self.assertRaises(SystemExit):
                probe.main(args)
        with patch.object(probe, 'prepare') as prepare, patch.object(probe, 'run', side_effect=AssertionError('no playback')):
            self.assertEqual(probe.main(['--prepare', '/private/new']), 0)
            prepare.assert_called_once_with(Path('/private/new'))

    def test_fake_sink_success_cleanup_and_no_replay(self):
        class FakeSink:
            instances = []
            def __init__(self, *args, **kwargs):
                self.leases = []
                self.closed = False
                self.audio_evidence = {'samples': 480000}
                self.timing = {'remote_verified': False}
                self.instances.append(self)
            def accept(self, path, lease):
                assert not Path(path).is_absolute(), 'sink requires relative snapshot path'
                assert Path(path).name==str(path)
                self.leases.append(lease)
            def finish(self):
                assert len(self.leases) == 3
            def close(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            binary = root / 'fake-binary'
            binary.write_bytes(b'not executed')
            binary.chmod(0o700)
            with patch('research.h3_personal.split_av_sink.SplitAVSink', FakeSink):
                result = probe.run(root, audio_binary=binary, playback_approved=True)
                self.assertEqual(result['status'], 'finished')
                self.assertTrue(FakeSink.instances[-1].closed)
                with self.assertRaises(FileExistsError):
                    probe.run(root, audio_binary=binary, playback_approved=True)
            self.assertFalse(json.loads((root / 'result.json').read_text())['remote_verified'])

    def test_fake_sink_failure_redacts_exception(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            binary = root / 'fake-binary'
            binary.write_bytes(b'not executed')
            binary.chmod(0o700)
            with patch('research.h3_personal.split_av_sink.SplitAVSink', side_effect=ValueError('private secret')):
                result = probe.run(root, audio_binary=binary, playback_approved=True)
            self.assertEqual(result['failure_type'], 'ValueError')
            self.assertNotIn('private secret', (root / 'result.json').read_text())

    def test_repository_output_rejected(self):
        with self.assertRaises(ValueError):
            probe.private_directory(Path(probe.__file__).resolve().parents[1] / 'never-created', fresh=True)


if __name__ == '__main__':
    unittest.main()
