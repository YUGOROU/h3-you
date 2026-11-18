"""Continuous, optionally read-paced, device-free FFV1/PCM NUT transport.

accept() acknowledges only both pipe writes, NOT durable/external delivery. The
parent serializes leases and owns Delivery.ack/sink_failed. No runtime adapter.
One encoder clock spans all segments; output is private and never uploaded.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
import errno
import math
import sys
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import tempfile
import threading
import time

from .av_delivery import Binding, Lease
from .buffered_stream import _asset


def _clock_binding(binding):
    """Accept one-source or conversation-span bindings without weakening checks.

    Returns (sample_rate, channels, fps). A conversation binding carries the
    whole conversation clock, so its first span must still start at frame zero.
    """
    if isinstance(binding, Binding):
        if binding.source_frame != 0:
            raise ValueError('zero-origin Binding required')
        if binding.source.sample_width != 2:
            raise ValueError('24fps and original signed S16LE PCM required')
        return binding.source.sample_rate, binding.source.channels, binding.fps
    from .paired_av import LiveAVBinding
    if type(binding) is LiveAVBinding:
        binding.__post_init__()
        return binding.sample_rate,binding.channels,Fraction(binding.fps)
    from .conversation_timeline import ConversationBinding
    if not isinstance(binding, ConversationBinding):
        raise ValueError('Binding or ConversationBinding required')
    if binding.spans[0].frame_start != 0 or binding.sources[0].sample_width != 2:
        raise ValueError('zero-origin conversation and S16LE PCM required')
    return binding.sample_rate, binding.channels, binding.fps


def _lease_payload(lease):
    """Return (segment, audio bytes) for one-source or conversation leases."""
    if isinstance(lease, Lease):
        return lease.segment, lease.original_pcm
    from .paired_av import ReceivedAVLease,_bodies
    if type(lease) is ReceivedAVLease:
        lease.segment.__post_init__()
        _bodies(lease.segment,lease.video,lease.audio_pcm)
        return lease.segment,lease.audio_pcm
    from .conversation_av import ConversationLease
    if isinstance(lease, ConversationLease):
        return lease.segment, lease.audio_pcm
    raise ValueError('Lease required')


def _pcm16_evidence(pcm):
    """Exact scalar S16LE statistics; no content retained in the result."""
    values = array('h')
    values.frombytes(pcm)
    if sys.byteorder != 'little':
        values.byteswap()
    return (sum(value != 0 for value in values),
            max((abs(value) for value in values), default=0),
            sum(value * value for value in values))


@dataclass(frozen=True)
class Submission:
    frame_stop: int
    sample_stop: int
    accepted_monotonic: float
    first_av_submission_monotonic: float
    frame_start: int | None = None
    sample_start: int | None = None
    omitted_frame_range: tuple[int, int] | None = None
    disposition: str = 'sink-submitted'
    regions: tuple = ()


@dataclass(frozen=True)
class _SuffixView:
    """Encoder-only view, deliberately NOT an AVEnvelope or wire receipt."""
    wire_segment: object
    binding: object
    index: int
    frame_start: int
    frame_stop: int
    sample_start: int
    sample_stop: int
    padding_samples: int
    regions: tuple


def _suffix_view(segment, pcm, frame):
    """Called only after full wire/hash/video validation; never changes the wire."""
    sample = frame * 2000
    trimmed = pcm[(sample-segment.sample_start)*2:]
    regions = []
    for region in segment.regions:
        lo = max(sample, region.sample_start)
        if lo >= region.sample_stop:
            continue
        if region.kind == 'speech':
            body = trimmed[(lo-sample)*2:(region.sample_stop-sample)*2]
            regions.append(replace(region, sample_start=lo,
                source_offset=region.source_offset+lo-region.sample_start,
                source_sha256=hashlib.sha256(body).hexdigest()))
        else:
            regions.append(replace(region, sample_start=lo))
    if not regions or regions[0].sample_start != sample or regions[-1].sample_stop != segment.sample_stop:
        raise ValueError('suffix regions must retain exact full suffix')
    return _SuffixView(segment, segment.binding, segment.index, frame,
        segment.frame_stop, sample, segment.sample_stop, segment.padding_samples,
        tuple(regions)), trimmed


class ReviewArchive:
    """One finite local-only review archive, outside the repository.

    Creates a fresh 0700 leaf in an existing resolved nonsymlink parent. Files
    are exclusive 0600; the shared cap includes sidecars and the current index.
    It preserves encoder bytes, NOT proof that a device displayed/heard them.
    Owned, reaped encoder spans use a hard link (no full-file hash scan); only
    EXDEV falls back to a bounded hashed copy. Callers must not write the source
    after capture. No media is decoded, uploaded or played by this class.
    """
    MAX_BYTES = 2 * 1024**3
    MAX_ENTRIES = 288

    def __init__(self, root, deadline, max_bytes=MAX_BYTES):
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or not 0 < deadline-time.monotonic() <= 1800
                or type(max_bytes) is not int or not 0 < max_bytes <= self.MAX_BYTES):
            raise ValueError('finite review deadline and byte cap required')
        path = Path(root).absolute()
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError('nonsymlink review path required')
        parent = path.parent.resolve(strict=True)
        path = parent / path.name
        repo = Path(__file__).resolve().parents[2]
        if path == repo or repo in path.parents:
            raise ValueError('review media must remain outside repository')
        path.mkdir(mode=0o700)
        path.chmod(0o700)
        self.root, self.deadline, self.max_bytes = path, deadline, max_bytes
        self._identity = (path.stat().st_dev, path.stat().st_ino)
        self._lock = threading.RLock()
        self._entries = []
        self._data_bytes = self._index_bytes = 0
        self._manifest = {}
        try:
            self._commit_index([], 0)
        except BaseException:
            path.rmdir()
            raise

    def _check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('review archive deadline')
        info = self.root.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077
                or (info.st_dev, info.st_ino) != self._identity
                or any(p.is_symlink() for p in (self.root, *self.root.parents))):
            raise ValueError('review directory ownership/path changed')

    def _budget(self, count):
        self._check()
        if self._data_bytes+self._index_bytes+count > self.max_bytes:
            raise ValueError('review byte cap exceeded; no truncation')

    @staticmethod
    def _json(value):
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
        if len(raw) > 256*1024:
            raise ValueError('bounded review metadata required')
        return raw

    def _write(self, name, source, *, expected_size=None, link_path=None):
        """Atomically publish a new exclusive file; caller rolls back on failure."""
        started = time.monotonic()
        self._check()
        target = self.root/name
        method = 'copy'
        if link_path is not None:
            info = os.fstat(source.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_size != expected_size):
                raise ValueError('owned stable encoder span required')
            self._budget(expected_size)
            # The encoder is reaped by HeadlessAVSink before reaching here.
            # Keep the original path/open fds intact; ordinary temp cleanup only
            # unlinks that name. No full-span read/hash/fsync on the interrupt path.
            if stat.S_IMODE(info.st_mode) not in (0o600, 0o400):
                os.fchmod(source.fileno(), 0o600)
            try:
                os.link(link_path, target, follow_symlinks=False)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                method = 'copy-exdev'
            else:
                try:
                    linked = target.lstat()
                    if (not stat.S_ISREG(linked.st_mode)
                            or (linked.st_dev, linked.st_ino, linked.st_size)
                               != (info.st_dev, info.st_ino, expected_size)):
                        raise ValueError('encoder span changed before hard link')
                    self._check()
                except BaseException:
                    target.unlink(missing_ok=True)
                    raise
                self._data_bytes += expected_size
                return dict(name=name, bytes=expected_size, sha256=None,
                    sha256_status='deferred_no_full_file_read_on_interrupt_path',
                    method='hard-link', duration_seconds=time.monotonic()-started)
        fd, temp = tempfile.mkstemp(prefix='.review-', dir=self.root)
        os.fchmod(fd, 0o600)
        temp = Path(temp)
        total = 0; digest = hashlib.sha256()
        try:
            with os.fdopen(fd, 'wb') as output:
                while block := source.read(1024*1024):
                    total += len(block)
                    self._budget(total)
                    if expected_size is not None and total > expected_size:
                        raise ValueError('review source grew during capture')
                    digest.update(block); output.write(block)
                if expected_size is not None and total != expected_size:
                    raise ValueError('review source changed during capture')
                output.flush()
            self._check()
            # link is atomic and refuses any existing file/symlink, unlike replace.
            os.link(temp, target, follow_symlinks=False)
            self._data_bytes += total
            return dict(name=name, bytes=total, sha256=digest.hexdigest(),
                        method=method, duration_seconds=time.monotonic()-started)
        finally:
            temp.unlink(missing_ok=True)

    def _commit_index(self, entries, data_bytes):
        guessed = self._index_bytes
        for _ in range(8):
            manifest = dict(schema='h3-local-review-archive-v1',
                clock='local_monotonic', deadline_monotonic=self.deadline,
                max_bytes=self.max_bytes, data_bytes=data_bytes,
                index_bytes=guessed, bytes_used=data_bytes+guessed,
                device_output_verified=False, all_buffered_audio_observed=False,
                entries=entries)
            raw = self._json(manifest)
            if len(raw) == guessed:
                break
            guessed = len(raw)
        else:
            raise ValueError('review index size did not converge')
        # Include both old and replacement index while the atomic temp exists.
        self._budget(len(raw))
        fd, temp = tempfile.mkstemp(prefix='.index-', dir=self.root)
        os.fchmod(fd, 0o600)
        temp = Path(temp)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw); stream.flush()
            self._check()
            os.replace(temp, self.root/'review-index.json')
        finally:
            temp.unlink(missing_ok=True)
        self._entries = entries
        self._index_bytes = len(raw)
        self._manifest = manifest

    def _capture(self, kind, sources, metadata):
        with self._lock:
            self._check()
            if len(self._entries) >= self.MAX_ENTRIES:
                raise ValueError('review entry bound exceeded')
            # Bounded strict JSON copy: no arbitrary object repr/content leakage.
            metadata = json.loads(self._json(metadata))
            before = self._data_bytes
            made = []
            media_started = time.monotonic()
            try:
                for item in sources:
                    name, source, size = item[:3]
                    link_path = item[3] if len(item) == 4 else None
                    made.append(self._write(name, source, expected_size=size, link_path=link_path))
                record = dict(kind=kind, captured_monotonic=time.monotonic(),
                    files=made, metadata=metadata,
                    media_retention_seconds=time.monotonic()-media_started,
                    retention_time_is_gpu_generation=False, device_output_verified=False,
                    all_buffered_audio_observed=False)
                import io
                sidecar = f'entry-{len(self._entries):03d}.json'
                raw = self._json(record)
                saved = self._write(sidecar, io.BytesIO(raw), expected_size=len(raw))
                made.append(saved)
                self._commit_index(self._entries+[record], self._data_bytes)
                return json.loads(self._json(record))
            except BaseException:
                for item in made:
                    (self.root/item['name']).unlink(missing_ok=True)
                self._data_bytes = before
                raise

    def capture_span(self, path, origin, interrupted, metadata):
        """Preserve an owned, quiescent NUT; caller must have reaped its encoder."""
        if (type(origin) is not tuple or len(origin) != 2
                or any(type(x) is not int or x < 0 for x in origin)
                or type(interrupted) is not bool or type(metadata) is not dict):
            raise ValueError('explicit span origin/state/metadata required')
        path = Path(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                raise ValueError('nonempty regular encoder NUT required')
            with self._lock:
                name = f'span-{len(self._entries):03d}.nut'
                return self._capture('encoder-span', [(name, source, info.st_size, path)],
                    dict(origin_index=origin[0], origin_frame=origin[1],
                        interrupted=interrupted,
                        state='interrupted_encoder_bytes' if interrupted else 'finished_encoder_bytes',
                        semantics='encoder-submitted bytes; not actual device playback',
                        child=metadata))

    def capture_visual_direction(self, *, proposed, active, intent_epoch, target_epoch):
        """Private text for semantic diagnosis, never native/visual success proof.

        Shared archive cap and exclusive sidecars apply; no reply/transcript,
        credentials or images are included. At most two published turn records.
        """
        from .conversation_conditioning import validate_visual_intent, role_record, visual_prompt
        proposed = validate_visual_intent(proposed)
        if type(target_epoch) is not int or not 1 <= target_epoch <= 2:
            raise ValueError('bounded review turn epoch required')
        roles = {role: dict(binding=role_record(role, visual_intents=True, active=active,
                    intent_epoch=intent_epoch, target_epoch=target_epoch),
                    prompt=visual_prompt(role, active)) for role in ('listening', 'speech')}
        value = dict(schema='h3-private-review-visual-direction-v1', target_epoch=target_epoch,
                     intent_epoch=intent_epoch, proposed=proposed, roles=roles,
                     stage='control_published_not_native_or_visual_achievement')
        if len(self._json(value)) > 32768:
            raise ValueError('private visual text bound')
        with self._lock:
            if any(e['kind'] == 'private-visual-direction'
                   and e['metadata']['target_epoch'] == target_epoch for e in self._entries):
                raise ValueError('duplicate private review turn')
            return self._capture('private-visual-direction', [], value)

    def capture_pair(self, envelope, video, pcm):
        from .paired_av import AVEnvelope, _bodies
        from .paired_av_wire import envelope_record
        if type(envelope) is not AVEnvelope or envelope.presentation_cut is None:
            raise ValueError('only explicit cut packets are review references')
        envelope.__post_init__(); _bodies(envelope, video, pcm)
        import io
        with self._lock:
            stem = f'pair-{envelope.index:03d}'
            c = envelope.presentation_cut
            return self._capture('full-native-cut-reference',
                [(stem+'.mp4', io.BytesIO(video), len(video)),
                 (stem+'.s16le', io.BytesIO(pcm), len(pcm))],
                dict(envelope=envelope_record(envelope),
                    semantics='full verified native packet; not full presentation',
                    omitted_frame_range=[envelope.frame_start, c.frame],
                    suffix_frame_range=[c.frame, envelope.frame_stop],
                    global_pts=dict(fps=24, sample_rate=48000)))

    def snapshot(self):
        """Export timing/file evidence without private direction text.

        Full diagnostic text stays in the 0600 on-disk index/entry sidecars;
        parent stdout/result timing must not implicitly publish it.
        """
        with self._lock:
            value = json.loads(self._json(self._manifest))
            for entry in value.get('entries', []):
                if entry['kind'] == 'private-visual-direction':
                    data = entry['metadata']
                    entry['metadata'] = dict(schema=data['schema'],
                        target_epoch=data['target_epoch'], intent_epoch=data['intent_epoch'],
                        stage=data['stage'], private_text_retained=True,
                        prompt_sha256_by_role={r: v['binding']['prompt_sha256']
                                              for r, v in data['roles'].items()})
            return value


class HeadlessAVSink:
    """Single serialized lease, bounded segment disk and two finite writers.

    source_frame must be zero: this file's continuous clock starts at zero.
    close/cancel aborts unfinished output; finish closes and headlessly validates.
    Context exit always reaps children. Read pacing is not device/SM timing.
    """
    def __init__(self, root, binding, *, width=252, height=448,
                 max_frames=119, max_total_frames=10000,
                 max_video_bytes=128 * 1024 * 1024, timeout=30, paced=False,
                 discontinuity_origin=None, review_archive=None):
        if review_archive is not None and type(review_archive) is not ReviewArchive:
            raise ValueError('explicit local ReviewArchive required')
        self._review_archive = review_archive
        self._review_capture_attempted = False
        self._review_capture_seconds = None
        self._review_capture_method = None
        self._review_lock = threading.Lock()
        if type(paced) is not bool:
            raise ValueError('paced must be bool')
        self._paced = paced
        rate, channels, fps = _clock_binding(binding)
        if fps != Fraction(24):
            raise ValueError('24fps and original signed S16LE PCM required')
        if any(type(x) is not int or x <= 0 for x in
               (width, height, max_frames, max_total_frames, max_video_bytes)):
            raise ValueError('positive finite bounds required')
        if not 0 < timeout <= 300:
            raise ValueError('finite timeout <=300s required')
        self.root = Path(root).resolve(strict=True)
        self.binding = binding
        self.sample_rate, self.channels = rate, channels
        self.width, self.height = width, height
        self.max_frames, self.max_total_frames = max_frames, max_total_frames
        self.max_video_bytes, self.timeout = max_video_bytes, timeout
        # Explicit review mode keeps the owned temp on the archive filesystem,
        # so interruption normally needs only a hard link, not a large copy.
        self._temp = tempfile.TemporaryDirectory(prefix='h3-headless-',
            dir=None if review_archive is None else review_archive.root.parent)
        self.directory = Path(self._temp.name)
        self.output = self.directory / 'continuous.nut'
        self._processes = []
        self._pipes = []
        self._cancel = threading.Event()
        self._close_lock = threading.Lock()
        self._writers = []
        self._frame = self._index = 0
        if discontinuity_origin is not None:
            from .paired_av import LiveAVBinding
            if (type(binding) is not LiveAVBinding or type(discontinuity_origin) is not tuple
                    or len(discontinuity_origin) != 2
                    or any(type(v) is not int or v < 0 for v in discontinuity_origin)):
                self._temp.cleanup()
                raise ValueError('explicit paired (index, global frame) discontinuity origin required')
            index, frame = discontinuity_origin
            if index >= binding.max_segments or frame >= self.max_total_frames:
                self._temp.cleanup()
                raise ValueError('discontinuity origin exceeds finite session bounds')
            self._index, self._frame = index, frame
        # Each private NUT span has local encoder timestamps; its absolute session
        # mapping is explicit, never a silent rewind of transport/native clocks.
        self._discontinuity_origin = (self._index, self._frame)
        self._timing_lock = threading.Lock()
        self._first_writes = [None, None]
        self._first_av_submission_monotonic = None
        self._acks = []
        self._timing_events = []
        self._audio_source_samples = self._audio_padding_samples = 0
        self._audio_nonzero = self._audio_peak = self._audio_squares = 0
        self.finished = False
        self._closed = False
        try:
            vr, vw = os.pipe(); self._pipes.append(vw)
            ar, aw = os.pipe(); self._pipes.append(aw)
            try:
                fd = os.open(self.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    # Both raw formats are fully specified. Stream-info discovery
                    # can wait for future PCM even with analyzeduration=1 (0 means
                    # the default, not disabled). Skip it entirely; retain every
                    # packet (never nobuffer). Bound pipe reads without AVIO direct,
                    # whose nonseekable header lookahead can consume initial bytes.
                    self._encoder = self._spawn([
                        'ffmpeg', '-v', 'error', '-nostdin',
                        '-probesize', '32', '-analyzeduration', '1',
                        '-fpsprobesize', '0', '-max_probe_packets', '1',
                        '-f', 'rawvideo', '-pixel_format', 'rgb24',
                        '-video_size', f'{width}x{height}', '-framerate', '24', '-r', '24',
                        '-nofind_stream_info', '-blocksize', '4096',
                        *(['-re'] if paced else []), '-i', f'pipe:{vr}',
                        '-probesize', '32', '-analyzeduration', '1',
                        '-fpsprobesize', '0', '-max_probe_packets', '1',
                        '-f', 's16le', '-ar', str(self.sample_rate),
                        '-ac', str(self.channels), '-nofind_stream_info',
                        '-blocksize', '4096',
                        *(['-re'] if paced else []), '-i', f'pipe:{ar}',
                        '-map', '0:v:0', '-map', '1:a:0', *self._encoder_tail()],
                        stdout=fd, pass_fds=(vr, ar))
                finally:
                    os.close(fd)
            finally:
                os.close(vr); os.close(ar)
            for pipe in self._pipes:
                os.set_blocking(pipe, False)
        except BaseException:
            self.close()
            raise

    def _encoder_tail(self):
        return ['-c:v','ffv1','-threads','1','-c:a','pcm_s16le','-f','nut','pipe:1']

    def _validate_output(self,deadline):
        p=self._spawn(['ffmpeg','-v','error','-xerror','-nostdin',
            '-i',str(self.output),'-map','0:v','-map','0:a','-f','null','-'],stdout=subprocess.DEVNULL)
        self._wait(p,deadline)
        return self.output

    @property
    def paced(self):
        return self._paced

    @property
    def first_av_submission_monotonic(self):
        with self._timing_lock:
            return self._first_av_submission_monotonic

    @property
    def timing(self):
        """Read-only snapshot: pipe submission/ACKs, not playback or SM timing."""
        from types import MappingProxyType
        with self._timing_lock:
            return MappingProxyType(dict(
                paced=self._paced,
                global_index_origin=self._discontinuity_origin[0],
                global_frame_origin=self._discontinuity_origin[1],
                output_clock='span_local_with_explicit_global_origin',
                review_retention=dict(method=self._review_capture_method,
                    duration_seconds=self._review_capture_seconds,
                    included_in_gpu_generation=False),
                first_video_bytes_write_monotonic=self._first_writes[0],
                first_audio_bytes_write_monotonic=self._first_writes[1],
                first_av_submission_monotonic=self._first_av_submission_monotonic,
                accepted_monotonic=tuple(self._acks[:144]),
                events=tuple(dict(event) for event in self._timing_events),
                ack_count=len(self._acks)))

    @property
    def audio_evidence(self):
        """Defensive JSON-ready aggregate of ACKed encoder input, not playback.

        Source/padding counts are sample frames; PCM16 statistics count scalar
        channel values and exclude explicit padding. RMS is in PCM16 units.
        No hashes, text, or PCM are retained in this snapshot.
        """
        with self._timing_lock:
            count = self._audio_source_samples * self.channels
            return dict(source_pcm_samples=self._audio_source_samples,
                        padding_pcm_samples=self._audio_padding_samples,
                        source_pcm16_samples=count,
                        nonzero_pcm16_samples=self._audio_nonzero,
                        peak_pcm16=self._audio_peak,
                        sum_squares_pcm16=self._audio_squares,
                        rms_pcm16=math.sqrt(self._audio_squares / count) if count else 0.0,
                        ack_count=len(self._acks),
                        encoder_input_only=True, remote_verified=False)

    def _spawn(self, args, **kwargs):
        # Serialize creation with cancellation: no untracked child after reaping.
        with self._close_lock:
            if self._closed or self._cancel.is_set():
                raise RuntimeError('closed sink refuses process creation')
            # Private temporary stderr avoids pipe deadlocks and preserves a
            # diagnostic on failure. Never include media paths/content in errors.
            errors = tempfile.TemporaryFile()
            try:
                p = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                                     stderr=errors, start_new_session=True, **kwargs)
            except BaseException:
                errors.close()
                raise
            p._h3_errors = errors
            self._processes.append(p)
            return p

    def _wait(self, p, deadline):
        while p.poll() is None:
            if self._cancel.is_set():
                raise RuntimeError('sink cancelled')
            if time.monotonic() >= deadline:
                raise TimeoutError('headless operation deadline')
            time.sleep(.005)
        if p.wait() != 0:
            errors = getattr(p, '_h3_errors', None)
            category = 'unclassified'
            if errors is not None:
                errors.seek(0)
                diagnostic = errors.read(8192).lower()
                for marker, label in ((b'unrecognized option', 'unsupported-option'),
                                      (b'option not found', 'unsupported-option'),
                                      (b'ff_frame_thread_encoder_init failed', 'encoder-thread-init'),
                                      (b'resource temporarily unavailable', 'resource-unavailable'),
                                      (b'invalid data found', 'invalid-input'),
                                      (b'unknown encoder', 'missing-encoder')):
                    if marker in diagnostic:
                        category = label
                        break
            raise RuntimeError(f'FFmpeg failed: exit={p.returncode}, category={category}')

    def _probe_video(self, snapshot, deadline):
        p = self._spawn(['ffprobe', '-v', 'error', '-show_entries',
            'stream=codec_type,width,height,r_frame_rate,avg_frame_rate',
            '-of', 'json', str(snapshot)], stdout=subprocess.PIPE)
        data = bytearray()
        try:
            os.set_blocking(p.stdout.fileno(), False)
            while True:
                if self._cancel.is_set():
                    raise RuntimeError('sink cancelled')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('headless probe deadline')
                if not select.select([p.stdout], [], [], min(.05, remaining))[0]:
                    continue
                block = os.read(p.stdout.fileno(), 65537 - len(data))
                if not block:
                    break
                data.extend(block)
                if len(data) > 65536:
                    raise ValueError('video probe output bound exceeded')
            self._wait(p, deadline)
        finally:
            p.stdout.close()
        try:
            streams = json.loads(data)['streams']
            if len(streams) != 1 or streams[0]['codec_type'] != 'video':
                raise ValueError('exactly one video stream and no other streams required')
            video = streams[0]
            if (video['width'], video['height']) != (self.width, self.height):
                raise ValueError('video resolution mismatch')
            if any(Fraction(video[key]) != 24 for key in ('r_frame_rate', 'avg_frame_rate')):
                raise ValueError('video frame rate must be 24fps')
        except (KeyError, TypeError, ZeroDivisionError) as exc:
            raise ValueError('invalid video probe metadata') from exc

    def accept(self, segment_path, lease, *, presentation_start_frame=None):
        """Submit full view or a declared paired suffix, never a full wire ACK.

        Even a suffix first verifies every original byte/region and decodes the
        complete original video. The returned Submission names the used range.
        """
        if self._closed or self.finished:
            raise RuntimeError('sink closed')
        accept_started = time.monotonic()
        deadline = accept_started + self.timeout
        try:
            s, pcm = _lease_payload(lease)
            wire_segment = s
            n = s.frame_stop - s.frame_start
            skip_frames = 0
            cut = getattr(s, 'presentation_cut', None)
            if presentation_start_frame is not None:
                from .paired_av import AVEnvelope, ReceivedAVLease
                if (type(lease) is not ReceivedAVLease or type(s) is not AVEnvelope
                        or type(presentation_start_frame) is not int or cut is None
                        or presentation_start_frame != cut.frame):
                    raise ValueError('exact declared paired presentation cut required')
                skip_frames = presentation_start_frame-s.frame_start
            elif cut is not None:
                raise ValueError('cut packet cannot be submitted as a full presentation')
            start_frame = s.frame_start+skip_frames
            if (s.binding != self.binding or s.index != self._index or
                    start_frame != self._frame or n > self.max_frames or
                    s.frame_stop > self.max_total_frames):
                raise ValueError('noncontiguous or out-of-bounds lease')
            stride = self.channels * 2
            if type(pcm) is not bytes or len(pcm) != (s.sample_stop-s.sample_start)*stride:
                raise ValueError('lease PCM sample range mismatch')
            relative = Path(segment_path)
            if (relative.is_absolute() or '..' in relative.parts or
                    not relative.parts or any((self.root / Path(*relative.parts[:i])).is_symlink()
                    for i in range(1, len(relative.parts)+1))):
                raise ValueError('relative nonsymlink video required')
            if (self.root / relative).stat().st_size > self.max_video_bytes:
                raise ValueError('video input byte bound exceeded')
            path = _asset(self.root, segment_path, s.video_sha256)
            snapshot = self.directory / 'segment.video'
            raw = self.directory / 'segment.rgb'
            digest = hashlib.sha256()
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as src, snapshot.open('xb') as dst:
                if not stat.S_ISREG(os.fstat(src.fileno()).st_mode):
                    raise ValueError('regular video required')
                total = 0
                while block := src.read(1024*1024):
                    total += len(block)
                    if total > self.max_video_bytes or time.monotonic() >= deadline:
                        raise ValueError('video snapshot bound exceeded')
                    digest.update(block); dst.write(block)
            if digest.hexdigest() != s.video_sha256:
                raise ValueError('snapshot SHA mismatch')
            decode_started = time.monotonic()
            self._probe_video(snapshot, deadline)
            with raw.open('xb') as dst:
                p = self._spawn(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(snapshot),
                    '-map', '0:v:0', '-an',
                    '-vsync', '0', '-frames:v', str(n+1),
                    '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], stdout=dst)
                self._wait(p, deadline)
            if raw.stat().st_size != n*self.width*self.height*3:
                raise ValueError('decoded byte/frame count mismatch')
            # Only now derive the encoder view: the immutable wire lease and
            # original verified snapshot are never rewritten/rehashed as suffix.
            if skip_frames:
                s, pcm = _suffix_view(s, pcm, start_frame)
                n -= skip_frames
            # Stage only after lease/media validation; commit with ACK below.
            nonzero, peak, squares = _pcm16_evidence(pcm)
            # No implicit truncation/resampling or silence except declared tail pad.
            audio = pcm + bytes(s.padding_samples*stride)
            pipe_started = time.monotonic()
            errors = []
            def writer(fd, source, stream_index):
                try:
                    while block := source.read(65536):
                        view = memoryview(block)
                        while view:
                            if self._cancel.is_set():
                                raise RuntimeError('cancelled writer')
                            remaining = deadline-time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError('pipe write deadline')
                            if select.select([], [fd], [], min(.05, remaining))[1]:
                                try:
                                    written = os.write(fd, view)
                                    if written > 0:
                                        now = time.monotonic()
                                        with self._timing_lock:
                                            if self._first_writes[stream_index] is None:
                                                self._first_writes[stream_index] = now
                                                if all(t is not None for t in self._first_writes):
                                                    self._first_av_submission_monotonic = max(self._first_writes)
                                    view = view[written:]
                                except BlockingIOError:
                                    pass
                except BaseException as exc:
                    errors.append(exc)
                    self._cancel.set()
            import io
            with raw.open('rb') as video, io.BytesIO(audio) as pcm:
                video.seek(skip_frames*self.width*self.height*3)
                self._writers = [threading.Thread(target=writer, args=(fd, src, index),
                    name='h3-headless-writer')
                    for index, (fd, src) in enumerate(zip(self._pipes, (video, pcm)))]
                for t in self._writers: t.start()
                for t in self._writers: t.join()
            if errors:
                raise errors[0]
            if self._encoder.poll() is not None:
                raise RuntimeError('encoder exited before EOF')
            now = time.monotonic()
            with self._timing_lock:
                self._acks.append(now)
                if len(self._timing_events) < 144:
                    self._timing_events.append(dict(
                        sequence=s.index, frame_start=s.frame_start, frame_stop=s.frame_stop,
                        sample_start=s.sample_start, sample_stop=s.sample_stop,
                        padding_samples=s.padding_samples,
                        accept_started_monotonic=accept_started, accepted_monotonic=now,
                        decode_seconds=pipe_started-decode_started,
                        pipe_seconds=now-pipe_started, total_seconds=now-accept_started,
                        media_seconds=n/24,
                        verified_full_frame_range=(wire_segment.frame_start, wire_segment.frame_stop),
                        omitted_frame_range=(wire_segment.frame_start, start_frame) if skip_frames else None,
                        disposition='partially-presented-interrupted' if skip_frames else 'sink-submitted'))
                self._audio_source_samples += s.sample_stop - s.sample_start
                self._audio_padding_samples += s.padding_samples
                self._audio_nonzero += nonzero
                self._audio_peak = max(self._audio_peak, peak)
                self._audio_squares += squares
            self._frame = s.frame_stop
            self._index += 1
            snapshot.unlink(); raw.unlink()
            return Submission(s.frame_stop, s.sample_stop, now, self.first_av_submission_monotonic,
                frame_start=s.frame_start, sample_start=s.sample_start,
                omitted_frame_range=(wire_segment.frame_start, start_frame) if skip_frames else None,
                disposition='partially-presented-interrupted' if skip_frames else 'sink-submitted',
                regions=tuple(getattr(s, 'regions', ())))
        except BaseException:
            self.close()
            raise

    def finish(self):
        if self._closed or self.finished or not self._frame:
            raise RuntimeError('sink closed or empty')
        try:
            deadline = time.monotonic() + self.timeout
            for fd in self._pipes: os.close(fd)
            self._pipes.clear()
            self._wait(self._encoder, deadline)
            output=self._validate_output(deadline)
            self.finished = True
            self._capture_review_span(interrupted=False)
            return output
        except BaseException:
            self.close()
            raise

    def cancel(self):
        self.close()

    def close(self):
        self._cancel.set()
        with self._close_lock:
            self._close_resources()

    def _capture_review_span(self, *, interrupted):
        # finish and cancellation can race; cleanup must wait for the one copy.
        with self._review_lock:
            if self._review_archive is None or self._review_capture_attempted:
                return
            encoder = getattr(self, '_encoder', None)
            if encoder is not None and encoder.poll() is None:
                raise RuntimeError('review span requires a reaped encoder')
            self._review_capture_attempted = True
            if not self.output.exists() or self.output.stat().st_size == 0:
                return
            started = time.monotonic()
            try:
                saved = self._review_archive.capture_span(self.output, self._discontinuity_origin,
                    interrupted, dict(semantics='encoder-submitted bytes, not actual device playback',
                        encoder_reaped=True, device_output_verified=False,
                        all_buffered_audio_observed=False,
                        timing=dict(self.timing), audio_input_evidence=self.audio_evidence,
                        global_frame_end_acknowledged=self._frame))
                self._review_capture_method = saved['files'][0]['method']
            finally:
                self._review_capture_seconds = time.monotonic()-started

    def _close_resources(self):
        if self._closed:
            return
        self._cancel.set()
        for p in self._processes:
            if p.poll() is None:
                try: os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            p.wait()
            errors = getattr(p, '_h3_errors', None)
            if errors is not None:
                errors.close()
        for t in self._writers:
            if t is not threading.current_thread(): t.join()
        for fd in self._pipes:
            os.close(fd)
        self._pipes.clear()
        self._closed = True
        try:
            self._capture_review_span(interrupted=not self.finished)
        finally:
            self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
