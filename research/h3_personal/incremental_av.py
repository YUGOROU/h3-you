"""Device-free incremental AV adapter, optionally wall-clock read-paced. Never owns the GPU runtime.

The final review mux is audit-only; only the continuous sink produces delivery
 audio. ACK means in-process pipe submission, not external durability.
"""
from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import shutil
import threading
import time
import wave

from .accepted_profiles import AcceptedQualityProfile
from .av_delivery import Binding, Delivery, Segment
from .conversation_state import SpeechSource
from .headless_av_sink import HeadlessAVSink
from .live_voice_runtime import RenderResult


@dataclass(frozen=True)
class IncrementalAVResult:
    result: RenderResult
    output: Path
    delivery_receipt: dict


def render_incremental_av(runtime, wav, digest, turn_dir, output, timeout=60, *, paced=False):
    """Render with one lazy sink/consumer and a two-segment ownership bound.

    Runtime rendering itself is synchronous and must retain its own finite GPU
    timeout. This adapter bounds every local wait by one shared deadline.
    """
    if type(paced) is not bool:
        raise ValueError('explicit boolean pacing required')
    profile = getattr(runtime, 'quality_profile', None)
    if type(profile) is not AcceptedQualityProfile:
        raise ValueError('explicit AcceptedQualityProfile required')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError('finite timeout <=300 required')
    deadline = time.monotonic() + timeout
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('incremental AV deadline')
        return value
    root = Path(runtime.paths['output']).resolve(strict=True)
    repo = Path(__file__).resolve().parents[2]
    turn = Path(turn_dir).resolve()
    dest = Path(output).absolute()
    parent = dest.parent.resolve(strict=True)
    dest = parent / dest.name
    if (root.is_relative_to(repo) or not turn.is_relative_to(root) or turn == root
            or turn.exists() or not dest.is_relative_to(root) or dest.is_relative_to(turn)
            or dest.exists() or dest.is_symlink()):
        raise ValueError('fresh private turn and separate private output required')
    if root.stat().st_mode & 0o077 or parent.stat().st_mode & 0o077:
        raise ValueError('private output directories required')
    path = Path(wav)
    if path.stat().st_size > 5_000_000:
        raise ValueError('WAV byte bound')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('WAV SHA mismatch')
    with wave.open(io.BytesIO(raw), 'rb') as w:
        rate, channels, width, samples = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if (rate not in (44100, 48000) or channels not in (1, 2) or width != 2
                or not 0 < samples <= rate * 20 or w.getcomptype() != 'NONE'):
            raise ValueError('bounded positive original S16LE WAV required')
        pcm = w.readframes(samples + 1)
        if len(pcm) != samples * channels * width:
            raise ValueError('truncated PCM')
    source = SpeechSource(hashlib.sha256(pcm).hexdigest(), rate, samples, channels, width)
    condition = threading.Condition()
    delivery = sink = worker = None
    paths = {}
    failure = []
    stopped = False
    highwater = 0
    ranges = []
    first = None

    def check():
        remaining()
        if failure:
            raise RuntimeError('incremental AV consumer failed') from failure[0]
        if stopped:
            raise RuntimeError('incremental AV cancelled')

    def consume():
        nonlocal first
        try:
            while True:
                with condition:
                    check()
                    lease = delivery.lease()
                    while lease is None:
                        if delivery.closed:
                            return
                        condition.wait(remaining())
                        check()
                        lease = delivery.lease()
                    name = paths[lease.segment.index]
                sink.timeout = remaining()
                submission = sink.accept(name, lease)
                with condition:
                    check()
                    delivery.ack(lease)  # Never retry an uncertain ACK.
                    paths.pop(lease.segment.index)
                    s = lease.segment
                    ranges.append(dict(frame_start=s.frame_start, frame_stop=s.frame_stop,
                        sample_start=s.sample_start, sample_stop=s.sample_stop,
                        padding_samples=s.padding_samples))
                    if first is None:
                        first = submission.first_av_submission_monotonic
                    condition.notify_all()
        except BaseException as exc:
            with condition:
                failure.append(exc)
                if delivery.terminal is None:
                    delivery.sink_failed()
                condition.notify_all()

    def callback(event):
        nonlocal delivery, sink, worker, highwater
        with condition:
            check()
            if type(event) is not dict:
                raise ValueError('typed committed callback required')
            for key in ('index', 'frame_count', 'frame_offset', 'pts_start', 'pts_end',
                        'time_base_numerator', 'time_base_denominator'):
                if type(event.get(key)) is not int:
                    raise ValueError('strict callback integers required')
            n, start = event['frame_count'], event['frame_offset']
            if (not 0 < n <= 119 or start < 0 or event['pts_start'] != start
                    or event['pts_end'] != start+n or event['time_base_numerator'] != 1
                    or event['time_base_denominator'] != 24 or event.get('video_only') is not True
                    or event.get('committed', True) is not True):
                raise ValueError('invalid committed segment boundary')
            video = Path(event['path'])
            if video.is_symlink() or video.resolve(strict=True).parent != turn:
                raise ValueError('callback must be directly contained in turn')
            if delivery is None:
                binding = Binding(event['session_id'], source.owner, source, profile.digest)
                delivery = Delivery(binding, pcm, max_segments=240, max_frames=119)
                sink = HeadlessAVSink(turn, binding, width=profile.exported_geometry[0],
                    height=profile.exported_geometry[1], max_total_frames=480, timeout=remaining(), paced=paced)
                worker = threading.Thread(target=consume, name='h3-incremental-av')
                worker.start()
            if event['session_id'] != delivery.binding.session_id:
                raise ValueError('stale session callback')
            segment = Segment(delivery.binding, event['index'], start, start+n, event['sha256'])
            while delivery.outstanding >= 2:
                condition.wait(remaining())
                check()
            delivery.enqueue(segment)
            paths[segment.index] = video.name
            highwater = max(highwater, delivery.outstanding)
            condition.notify_all()

    published = False
    try:
        result = runtime.render_continuation(wav, digest, turn, on_segment=callback)
        returned = time.monotonic()
        with condition:
            check()
            if delivery is None:
                raise ValueError('render returned no segments')
            delivery.close_producer()
            condition.notify_all()
        worker.join(remaining())
        with condition:
            check()
            if worker.is_alive() or not delivery.final_flush():
                raise TimeoutError('consumer did not flush')
        sink.timeout = remaining()
        completed = sink.finish()
        remaining()
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        published = True
        with os.fdopen(fd, 'wb') as dst, completed.open('rb') as src:
            while block := src.read(1024 * 1024):
                remaining()
                dst.write(block)
        dest.chmod(0o400)
        return IncrementalAVResult(result, dest, dict(classification='headless-read-paced' if paced else 'headless-only-unpaced',
            profile_sha256=profile.digest, session_id=delivery.binding.session_id,
            source_pcm_sha256=source.digest, timing=dict(sink.timing),
            render_return_monotonic=returned, delivery_finished_monotonic=time.monotonic(),
            ranges=ranges, sample_rate=rate, sample_clock='original PCM floor(frame*rate/24)',
            firstAVsubmittedbefore_renderreturn=first is not None and first < returned,
            bounded_queue_highwater=highwater, audio_producers=1, externalDurability=False))
    except BaseException:
        if published:
            dest.unlink(missing_ok=True)
        raise
    finally:
        with condition:
            stopped = True
            condition.notify_all()
        if sink is not None:
            sink.cancel()
        if worker is not None:
            worker.join(timeout=max(1, min(timeout, 5)))
