"""Opt-in split output: original PCM device queue and video-only localhost TS.

Evidence proves submission/scheduling only. OBS/browser latency is UNKNOWN and
requires user review; neither device audibility nor remote playback is verified.
Injected audio clients implement ready(), append(message), finish(count), cancel().
Injected decoders implement (sink, relative_path, lease, deadline) -> RGB bytes.
"""
from __future__ import annotations

from collections import deque
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import time

from .headless_av_sink import HeadlessAVSink, Submission, _lease_payload, _pcm16_evidence
from .obs_av_sink import output_arguments
from .paired_av import LiveAVBinding, ReceivedAVLease
from .buffered_stream import _asset

_LIMIT = 1024 * 1024


def video_output_arguments(port):
    """Pure command builder; never carries an audio encoder or audio mapping."""
    args = output_arguments(port)
    for option in ('-c:a', '-b:a'):
        i = args.index(option)
        del args[i:i + 2]
    return ['-an', *args]


class _AudioClient:
    def __init__(self, sink, binary, samples):
        self.sink = sink
        self.process = sink._spawn([str(binary), '--run', '--max-samples', str(samples),
            '--timeout-seconds', str(10800 if sink.long_session else min(900, math.ceil(samples / 48000 + 60)))]+
            (['--pause-on-underflow'] if sink.today_pause_on_underflow else [])+
            (['--three-hour-session'] if sink.long_session else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.pending = bytearray()
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def _read(self):
        deadline = self.sink._clock() + self.sink.timeout
        while b'\n' not in self.pending:
            self.sink._check(deadline)
            if select.select([self.process.stdout], [], [], .02)[0]:
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError('audio worker EOF')
                self.pending.extend(data)
                if len(self.pending) > _LIMIT:
                    raise ValueError('audio JSONL bound')
        line, _, rest = self.pending.partition(b'\n')
        self.pending = bytearray(rest)
        result = json.loads(line)
        if type(result) is not dict:
            raise ValueError('audio object required')
        return result

    def _exchange(self, message):
        data = json.dumps(message, separators=(',', ':'), allow_nan=False).encode() + b'\n'
        if len(data) > _LIMIT:
            raise ValueError('audio JSONL bound')
        self.sink._write(self.process.stdin.fileno(), data,
                         self.sink._clock() + self.sink.timeout)
        return self._read()  # Never retry a possibly committed write.

    def ready(self):
        return self._read()

    def append(self, message):
        return self._exchange(message)

    def resume(self, sample_start, epoch):
        return self._exchange(dict(type='resume', sample_start=sample_start,
                                   epoch_host_seconds=epoch))

    def finish(self, count):
        return self._exchange(dict(type='finish', samples=count))

    def cancel(self):
        # One best-effort nonblocking cancellation, never replay an unknown write.
        if self.process.poll() is None and not self.process.stdin.closed:
            try:
                os.write(self.process.stdin.fileno(), b'{"type":"cancel"}\n')
            except (OSError, ValueError):
                pass
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None:
                pipe.close()


def decode_video(sink, relative_path, lease, deadline):
    """Private bounded snapshot, native path/hash checks, full global PTS probe."""
    s = lease.segment
    relative = Path(relative_path)
    if (relative.is_absolute() or not relative.parts or '..' in relative.parts or
            any((sink.root / Path(*relative.parts[:i])).is_symlink()
                for i in range(1, len(relative.parts) + 1))):
        raise ValueError('relative nonsymlink video required')
    path = _asset(sink.root, relative_path, s.video_sha256)
    import stat
    with tempfile.TemporaryDirectory(prefix='h3-split-segment-') as directory:
        snapshot = Path(directory) / 'video'
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError('regular video required')
            data = source.read(16 * 1024 * 1024 + 1)
        if (len(data) != s.video_size or len(data) > 16 * 1024 * 1024 or
                hashlib.sha256(data).hexdigest() != s.video_sha256 or data != lease.video):
            raise ValueError('video snapshot mismatch')
        snapshot.write_bytes(data)
        sink._probe_video(snapshot, deadline)
        probe = sink._spawn(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time',
            '-of', 'json', str(snapshot)], stdout=subprocess.PIPE)
        metadata = sink._bounded_read(probe, _LIMIT, deadline)
        frames = json.loads(metadata)['frames']
        n = s.frame_stop - s.frame_start
        if len(frames) != n or any(abs(float(f['best_effort_timestamp_time']) -
                (s.frame_start + i) / 24) > .00001 for i, f in enumerate(frames)):
            raise ValueError('global video PTS/count mismatch')
        p = sink._spawn(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(snapshot),
            '-map', '0:v:0', '-an', '-vsync', '0', '-frames:v', str(n + 1),
            '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
            stdout=subprocess.PIPE)
        raw = sink._bounded_read(p, n * 252 * 448 * 3, deadline)
        if len(raw) != n * 252 * 448 * 3:
            raise ValueError('decoded frame count mismatch')
        return raw


class SplitAVSink(HeadlessAVSink):
    """Serialized ReceiverPump sink; failures are terminal, with no clock reset."""
    def __init__(self, root, binding, *, playback_approved=False, port=None,
                 audio_binary=None, audio_delay_seconds=.2, timeout=30,
                 today_pause_on_underflow=False, long_session=False,
                 max_frames=124, max_total_frames=None,
                 process_factory=subprocess.Popen, clock=time.monotonic,
                 sleeper=time.sleep, audio_client=None, decoder=decode_video):
        if playback_approved is not True:
            raise ValueError('explicit playback approval required')
        tail = video_output_arguments(port)
        if (type(binding) is not LiveAVBinding or type(audio_delay_seconds) not in (int, float)
                or not math.isfinite(audio_delay_seconds) or not 0 <= audio_delay_seconds <= 1
                or type(timeout) not in (int, float) or not 0 < timeout <= 300):
            raise ValueError('finite clock/binding bounds required')
        binding.__post_init__()
        if (type(max_frames) is not int or not 1<=max_frames<=124 or
                (max_total_frames is not None and (type(max_total_frames) is not int or
                    max_total_frames!=binding.max_total_frames))):
            raise ValueError('fixed bounded split frame budget required')
        if audio_binary is None:
            raise ValueError('external compiled audio binary required')
        binary = Path(audio_binary).resolve(strict=True)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError('executable audio binary required')
        self.root = Path(root).resolve(strict=True)
        self.binding = binding
        self.sample_rate, self.channels = 48000, 1
        self.width, self.height = 252, 448
        if type(today_pause_on_underflow) is not bool:
            raise ValueError('explicit pause workaround selection required')
        if type(long_session) is not bool or (long_session and not today_pause_on_underflow):
            raise ValueError('three-hour sink requires explicit pause workaround')
        self.long_session = long_session
        self.today_pause_on_underflow = today_pause_on_underflow
        self._pause_events = []
        self.timeout = timeout
        self.max_frames=max_frames
        self.max_total_frames = binding.max_total_frames
        self._clock, self._sleep = clock, sleeper
        self._factory, self._decoder = process_factory, decoder
        self._processes = []
        self._cancel = threading.Event()
        self._closed = self.finished = False
        self._frame = self._index = 0
        self._sample = 0
        self._acks = []
        self._audio_source_samples = self._audio_padding_samples = 0
        self._audio_nonzero = self._audio_peak = self._audio_squares = 0
        self._timing_lock = threading.RLock()
        self._condition = threading.Condition(self._timing_lock)
        self._queue = deque()
        self._outstanding = self._highwater = 0
        self._eos = False
        self._error = None
        self._playout = None
        self._video_frames_written = 0
        self._max_late = 0.
        self._first_desired = None
        self.video_epoch = self.audio_epoch = None
        self.audio_delay_seconds = audio_delay_seconds
        self._first_av_submission_monotonic = None
        self._audio = None
        self._audio_drained=False
        try:
            before = clock()
            self._audio = audio_client or _AudioClient(self, binary, binding.max_total_frames * 2000)
            ready = self._audio.ready()
            after = clock()
            host = ready.get('host_seconds')
            if (ready.get('type') != 'ready' or ready.get('sample_rate') != 48000 or
                    ready.get('channels') != 1 or ready.get('device_uid') != 'BlackHole2ch_UID'
                    or ready.get('capture') is not False or type(host) not in (int, float)
                    or not math.isfinite(host) or not before <= host <= after):
                raise ValueError('incompatible audio clock/device ready')
            self._encoder = self._spawn(['ffmpeg', '-v', 'error', '-nostdin',
                '-probesize','32','-analyzeduration','1','-fpsprobesize','0',
                '-max_probe_packets','1','-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '252x448',
                '-framerate', '24','-nofind_stream_info','-blocksize','4096','-i', 'pipe:0', '-map', '0:v:0', *tail],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
            os.set_blocking(self._encoder.stdin.fileno(), False)
            self._playout = threading.Thread(target=self._play_video, name='h3-video-playout')
            self._playout.start()
        except BaseException:
            self.close()
            raise

    def _spawn(self, args, **kwargs):
        if self._closed:
            raise RuntimeError('closed sink')
        p = self._factory(args, stderr=subprocess.DEVNULL, start_new_session=True,
                          **kwargs)
        self._processes[:] = [child for child in self._processes if child.poll() is None]
        self._processes.append(p)
        return p

    def _check(self, deadline):
        if self._error is not None:
            raise RuntimeError('video playout failed') from self._error
        if self._closed or self._cancel.is_set():
            raise RuntimeError('sink cancelled')
        if self._clock() >= deadline:
            raise TimeoutError('split sink deadline')

    def _wait(self, p, deadline):
        while p.poll() is None:
            self._check(deadline)
            self._sleep(.005)
        if p.wait(timeout=1) != 0:
            raise RuntimeError('split subprocess failed')

    def _bounded_read(self, p, limit, deadline):
        data = bytearray()
        os.set_blocking(p.stdout.fileno(), False)
        try:
            while True:
                self._check(deadline)
                if not select.select([p.stdout], [], [], .02)[0]:
                    continue
                block = os.read(p.stdout.fileno(), min(65536, limit + 1 - len(data)))
                if not block:
                    break
                data.extend(block)
                if len(data) > limit:
                    raise ValueError('subprocess output bound')
            self._wait(p, deadline)
            return bytes(data)
        finally:
            p.stdout.close()

    def _write(self, fd, data, deadline):
        view = memoryview(data)
        while view:
            self._check(deadline)
            if select.select([], [fd], [], .02)[1]:
                try:
                    n = os.write(fd, view)
                    if n <= 0:
                        raise RuntimeError('short pipe write')
                    view = view[n:]
                except BlockingIOError:
                    pass

    @property
    def timing(self):
        with self._condition:
            return self._timing_snapshot()

    def _timing_snapshot(self):
        from types import MappingProxyType
        return MappingProxyType(dict(video_epoch=self.video_epoch,
            audio_epoch=self.audio_epoch, audio_delay_seconds=self.audio_delay_seconds,
            first_av_submission_monotonic=self._first_av_submission_monotonic,
            accepted_monotonic=tuple(self._acks), ack_count=len(self._acks),
            queue_highwater=self._highwater, max_decoded_outstanding=2,
            video_frames_written=self._video_frames_written,
            desired_first_video_write=self._first_desired,
            actual_first_video_write=self._first_av_submission_monotonic,
            first_video_write_skew=(None if self._first_av_submission_monotonic is None else
                self._first_av_submission_monotonic-self._first_desired),
            max_video_lateness=self._max_late,
            today_pause_on_underflow=self.today_pause_on_underflow,
            pause_events=tuple(self._pause_events),
            continuous_playback_verified=False,
            ack_scope='local_raw_queue_and_device_audio_queue',
            schedule_alignment_only=True, remote_verified=False))

    @property
    def audio_evidence(self):
        with self._condition:
            return self._audio_snapshot()

    def _audio_snapshot(self):
        result = super().audio_evidence
        result.update(device_queue_only=True, obs_audio_disabled=True,
                      device_playback_completed=self._audio_drained, schedule_alignment_only=True,
                      video_epoch=self.video_epoch, audio_epoch=self.audio_epoch,
                      audio_delay_seconds=self.audio_delay_seconds)
        return result

    def accept(self, path, lease):
        if self._closed or self.finished:
            raise RuntimeError('sink closed')
        try:
            if type(lease) is not ReceivedAVLease:
                raise ValueError('immutable ReceivedAVLease required')
            s, pcm = _lease_payload(lease)
            if (s.binding != self.binding or s.index != self._index or
                    s.frame_start != self._frame or s.sample_start != self._sample or
                    s.frame_stop > self.max_total_frames or
                    s.frame_stop-s.frame_start>self.max_frames):
                raise ValueError('noncontiguous lease')
            deadline = self._clock() + self.timeout
            with self._condition:
                while self._outstanding >= 2:
                    self._check(deadline)
                    self._condition.wait(.02)
                self._check(deadline)
                self._outstanding += 1
                self._highwater = max(self._highwater, self._outstanding)
            raw = self._decoder(self, path, lease, deadline)
            n = s.frame_stop - s.frame_start
            stride = 252 * 448 * 3
            if type(raw) is not bytes or len(raw) != n * stride:
                raise ValueError('raw RGB count mismatch')
            self._check(deadline)
            if self.video_epoch is None:
                self.video_epoch = self._clock() + .75
                self.audio_epoch = self.video_epoch + self.audio_delay_seconds
            target = self.video_epoch + s.frame_start / 24
            if self._clock() - target > .1:
                if not self.today_pause_on_underflow:
                    raise RuntimeError('expired video segment')
                # Do not retime any queued media. Drain earlier video and audio,
                # then move BOTH schedules by the same measured pause duration.
                with self._condition:
                    while self._outstanding > 1:
                        self._check(deadline)
                        self._condition.wait(.02)
                old_epoch = self.video_epoch
                new_epoch = self._clock() + .75 - s.frame_start / 24
                audio_epoch = new_epoch + self.audio_delay_seconds
                resumed = self._audio.resume(s.sample_start, audio_epoch)
                if resumed != dict(type='resumed', sample_start=s.sample_start,
                                   epoch_host_seconds=audio_epoch):
                    raise ValueError('ambiguous paired pause resume ACK')
                with self._condition:
                    self.video_epoch, self.audio_epoch = new_epoch, audio_epoch
                    self._pause_events.append(dict(frame_start=s.frame_start,
                        sample_start=s.sample_start, pause_seconds=new_epoch-old_epoch))
            message = dict(type='append', sequence=s.index, sample_start=s.sample_start,
                samples=s.sample_stop - s.sample_start, pcm_sha256=s.pcm_sha256,
                pcm_base64=base64.b64encode(pcm).decode('ascii'),
                epoch_host_seconds=self.audio_epoch)
            ack = self._audio.append(message)
            expected = self.audio_epoch + s.sample_start / 48000
            if (any(ack.get(k) != v for k, v in message.items() if k not in ('type', 'pcm_base64'))
                    or ack.get('type') != 'queued'
                    or type(ack.get('expected_start_host_seconds')) not in (float, int)
                    or not math.isfinite(ack['expected_start_host_seconds'])
                    or abs(ack['expected_start_host_seconds'] - expected) > .000001):
                raise ValueError('ambiguous audio queue ACK')
            self._check(deadline)
            if self._clock() - expected > .1:
                raise RuntimeError('expired audio queue submission')
            nonzero,peak,squares=_pcm16_evidence(pcm)
            now = self._clock()
            with self._condition:
                self._audio_source_samples += len(pcm)//2
                self._audio_nonzero += nonzero
                self._audio_peak = max(self._audio_peak, peak)
                self._audio_squares += squares
                self._check(deadline)
                self._acks.append(now)
                self._frame, self._sample = s.frame_stop, s.sample_stop
                self._index += 1
                self._queue.append((s.frame_start, n, raw))
                self._condition.notify_all()
                return Submission(self._frame, self._sample, now, self._first_av_submission_monotonic)
        except BaseException:
            self.close()
            raise

    def _play_video(self):
        try:
            while True:
                with self._condition:
                    idle_deadline = self._clock() + self.timeout
                    while not self._queue and not self._eos:
                        self._check(self._clock()+self.timeout if self.today_pause_on_underflow else idle_deadline)
                        if self._encoder.poll() is not None:
                            raise RuntimeError('video encoder exited')
                        self._condition.wait(.02)
                    self._check(self._clock() + self.timeout)
                    if not self._queue:
                        return
                    start, count, raw = self._queue.popleft()
                stride = 252 * 448 * 3
                for i in range(count):
                    scheduled = self.video_epoch + (start+i)/24
                    with self._condition:
                        while self._clock() < scheduled:
                            self._check(scheduled + .1)
                            if self._encoder.poll() is not None:
                                raise RuntimeError('video encoder exited')
                            self._condition.wait(min(.02, scheduled-self._clock()))
                    self._check(scheduled + .1)
                    self._write(self._encoder.stdin.fileno(),
                                memoryview(raw)[i*stride:(i+1)*stride], scheduled+.1)
                    now = self._clock()
                    if now-scheduled > .1:
                        raise RuntimeError('expired video write')
                    with self._condition:
                        if self._first_av_submission_monotonic is None:
                            self._first_desired = scheduled
                            self._first_av_submission_monotonic = now
                        self._video_frames_written += 1
                        self._max_late = max(self._max_late, now-scheduled)
                del raw
                with self._condition:
                    self._outstanding -= 1
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                if not self._cancel.is_set():
                    self._error = error
                self._condition.notify_all()

    def finish(self):
        if self._closed or self.finished or not self._frame:
            raise RuntimeError('closed or empty sink')
        try:
            with self._condition:
                self._eos = True
                self._condition.notify_all()
            self._playout.join(self.max_total_frames/24 + self.timeout + 1)
            if self._playout.is_alive():
                raise TimeoutError('video drain timeout')
            self._check(self._clock() + self.timeout)
            self._encoder.stdin.close()
            self._wait(self._encoder, self._clock() + self.timeout)
            receipt = self._audio.finish(self._sample)
            if (receipt.get('type') != 'finished' or type(receipt.get('samples')) is not int or
                    receipt['samples'] != self._sample or receipt.get('device_uid')!='BlackHole2ch_UID' or
                    receipt.get('preserved_defaults') is not True or receipt.get('capture') is not False):
                raise ValueError('audio drain receipt mismatch')
            for p in self._processes:
                self._wait(p, self._clock() + self.timeout)
            self._audio_drained=True
            self.finished = True
            return self.audio_evidence
        except BaseException:
            self.close()
            raise
        finally:
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._cancel.set()
        with self._condition:
            self._queue.clear()
            self._condition.notify_all()
        if self._playout is not None:
            self._playout.join(2)
            if self._playout.is_alive():
                raise RuntimeError('video playout cancellation timeout')
        if self._audio is not None:
            try:
                self._audio.cancel()
            except Exception:
                pass
        for p in self._processes:
            for name in ('stdin', 'stdout'):
                pipe = getattr(p, name, None)
                if pipe is not None and not pipe.closed:
                    pipe.close()
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=2)

    cancel = close
