"""Local pipes and injected workers only: no devices, UDP, cloud or GPU."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import time
import shutil

from research.h3_personal.split_av_sink import SplitAVSink, video_output_arguments
from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, ReceivedAVLease, encode


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class Audio:
    def __init__(self, clock):
        self.clock = clock
        self.messages = []
        self.closed = False
        self.bad = False
    def ready(self):
        return dict(type='ready', host_seconds=self.clock(), sample_rate=48000,
                    channels=1, device_uid='BlackHole2ch_UID', capture=False)
    def append(self, message):
        self.messages.append(message)
        result = dict(message, type='queued', expected_start_host_seconds=
                      message['epoch_host_seconds'] + message['sample_start']/48000)
        if self.bad: result['pcm_sha256'] = '0'*64
        return result
    def finish(self, count): return dict(type='finished', samples=count,
        device_uid='BlackHole2ch_UID',preserved_defaults=True,capture=False)
    def cancel(self): self.closed = True


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root/'worker'
        self.binary.write_text('#!/bin/sh\nexit 0\n')
        self.binary.chmod(0o700)
        self.clock = time.monotonic
        self.audio = Audio(self.clock)
        self.binding = LiveAVBinding('test', 'a'*64, 0, max_total_frames=10)
        self.commands = []

    def factory(self, command, **kwargs):
        self.commands.append(command)
        # Consume raw video locally; never execute the UDP command.
        return subprocess.Popen([sys.executable, '-c',
            'import sys; sys.stdin.buffer.read()'], **kwargs)

    def sink(self, **kwargs):
        sink = SplitAVSink(self.root, self.binding, playback_approved=True,
            port=12345, audio_binary=self.binary, process_factory=self.factory,
            audio_client=self.audio, clock=self.clock,
            decoder=kwargs.pop('decoder', lambda sink, path, lease, deadline:
                bytes((lease.segment.frame_stop-lease.segment.frame_start)*252*448*3)),
            **kwargs)
        self.addCleanup(sink.cancel)
        return sink

    def lease(self, index=0):
        pcm = b'\x01\x00'*2000
        video = b'synthetic verified packet'
        region = AudioRegion('speech', index*2000, (index+1)*2000,
                             'source', 0, index*2000, hashlib.sha256(pcm).hexdigest())
        header = encode(self.binding, index, index, index+1, video, pcm, (region,), False)
        return ReceivedAVLease(header, video, pcm)

    def test_command_never_audio(self):
        args = video_output_arguments(12345)
        self.assertIn('-an', args)
        self.assertNotIn('-c:a', args)
        self.assertNotIn('-b:a', args)
        for port in (None, True, 80, 65536):
            with self.assertRaises(ValueError): video_output_arguments(port)

    def test_strict_budget_and_protocol_names(self):
        for kwargs in ({'max_frames':True},{'max_frames':125},{'max_total_frames':99}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.sink(**kwargs)
        sink=self.sink(max_frames=124,max_total_frames=self.binding.max_total_frames)
        sink.accept('unused',self.lease())
        self.assertIn('sequence',self.audio.messages[0]);self.assertNotIn('seq',self.audio.messages[0])
        self.assertTrue(sink.audio_evidence['obs_audio_disabled'])

    def test_gate_before_spawn(self):
        with self.assertRaises(ValueError):
            SplitAVSink(self.root, self.binding, port=12345,
                        audio_binary=self.binary, process_factory=self.factory)
        self.assertEqual(self.commands, [])

    def test_exact_pcm_and_shared_epoch(self):
        sink = self.sink()
        first = sink.accept('unused', self.lease())
        second = sink.accept('unused', self.lease(1))
        self.assertEqual((first.frame_stop, second.sample_stop), (1, 4000))
        self.assertAlmostEqual(sink.audio_epoch-sink.video_epoch, .2)
        self.assertIsNone(first.first_av_submission_monotonic) 
        import base64
        self.assertEqual(base64.b64decode(self.audio.messages[0]['pcm_base64']), self.lease().audio_pcm)
        evidence = sink.finish()
        self.assertEqual(evidence['source_pcm_samples'], 4000)
        self.assertTrue(evidence['device_queue_only'])
        self.assertTrue(evidence['encoder_input_only'])
        self.assertFalse(evidence['remote_verified'])
        self.assertTrue(self.audio.closed)

    def test_bad_ack_terminal(self):
        sink = self.sink()
        self.audio.bad = True
        with self.assertRaises(ValueError): sink.accept('unused', self.lease())
        with self.assertRaises(RuntimeError): sink.accept('unused', self.lease())
        self.assertEqual(len(self.audio.messages), 1)
        self.assertEqual(sink.audio_evidence['ack_count'], 0)

    def test_late_segment_no_replay(self):
        sink = self.sink()
        sink.accept('unused', self.lease())
        time.sleep(1)
        with self.assertRaises(RuntimeError): sink.accept('unused', self.lease(1))
        self.assertEqual(len(self.audio.messages), 1)

    def test_delay_and_clock_rejection(self):
        for delay in (float('nan'), float('inf'), -.1, 1.1, True):
            with self.assertRaises(ValueError): self.sink(audio_delay_seconds=delay)
        self.audio.ready = lambda: dict(type='ready', host_seconds=0,
            sample_rate=48000, channels=1, device_uid='BlackHole2ch_UID', capture=False)
        with self.assertRaises(ValueError): self.sink()
        self.assertEqual(self.commands, [])

    def test_three_packets_decode_ahead_and_drain(self):
        self.binding = LiveAVBinding('test', 'a'*64, 0, max_total_frames=72)
        def decoder(sink, path, lease, deadline):
            time.sleep(.2)
            return bytes(24*252*448*3)
        sink = self.sink(decoder=decoder)
        for index in range(3):
            pcm = b'\x01\x00' * 48000
            video = b'packet'
            start = index*48000
            region = AudioRegion('speech', start, start+48000, 'source', 0,
                                 start, hashlib.sha256(pcm).hexdigest())
            header = encode(self.binding, index, index*24, (index+1)*24,
                            video, pcm, (region,), False)
            sink.accept('unused', ReceivedAVLease(header, video, pcm))
            if index == 1:
                self.assertEqual(sink.timing['video_frames_written'], 0)
        sink.finish()
        self.assertEqual(sink.timing['queue_highwater'], 2)
        self.assertEqual(sink.timing['video_frames_written'], 72)
        self.assertFalse(sink._playout.is_alive())

    def test_writer_failure_and_cancel_join(self):
        sink = self.sink()
        sink.accept('unused', self.lease())
        sink._encoder.terminate()
        sink._encoder.wait(timeout=2)
        time.sleep(.05)
        with self.assertRaises(RuntimeError):
            sink.finish()
        self.assertFalse(sink._playout.is_alive())
        other = self.sink()
        other.accept('unused', self.lease())
        other.cancel()
        self.assertFalse(other._playout.is_alive())
        self.assertTrue(all(p.poll() is not None for p in other._processes))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_production_decode_local_file(self):
        from research.h3_personal.split_av_sink import decode_video
        self.binding = LiveAVBinding('test', 'a'*64, 0, max_total_frames=8)
        path = self.root/'clip.ts'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
            'color=size=252x448:rate=24', '-frames:v', '8', '-c:v', 'libx264',
            '-bf', '0', '-threads', '1', '-muxdelay', '0', '-f', 'mpegts', str(path)], check=True)
        video = path.read_bytes()
        pcm = b'\x01\x00'*16000
        region = AudioRegion('speech', 0, 16000, 'source', 0, 0,
                             hashlib.sha256(pcm).hexdigest())
        header = encode(self.binding, 0, 0, 8, video, pcm, (region,), False)
        output = self.root/'output.ts'
        def factory(command, **kwargs):
            if command[0] == 'ffmpeg' and 'rawvideo' in command and 'pipe:0' in command:
                command = command[:command.index('-an')] + ['-an', '-c:v', 'libx264',
                    '-threads', '1', '-f', 'mpegts', str(output)]
            return subprocess.Popen(command, **kwargs)
        sink = SplitAVSink(self.root, self.binding, playback_approved=True,
            port=12345, audio_binary=self.binary, audio_client=self.audio,
            process_factory=factory, decoder=decode_video)
        self.addCleanup(sink.cancel)
        sink.accept('clip.ts', ReceivedAVLease(header, video, pcm))
        sink.finish()
        self.assertEqual(sink.timing['video_frames_written'], 8)
        self.assertGreater(output.stat().st_size, 0)

    def test_mutated_payload_rejected(self):
        sink = self.sink()
        lease = self.lease()
        invalid = ReceivedAVLease(lease.segment, lease.video, bytes(len(lease.audio_pcm)))
        with self.assertRaises(ValueError): sink.accept('unused', invalid)
        self.assertFalse(self.audio.messages)


if __name__ == '__main__':
    unittest.main()
