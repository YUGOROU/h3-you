"""Device-free paired output invalidation over existing FFmpeg headless spans.

No browser/OBS/device implementation is implied. The session clock stays global;
new private encoder spans have an explicit global origin and discontinuity. A
span restart never restarts H3 or its persistent decoder. No padding/rate change.
"""
import threading
import time

from .headless_av_sink import HeadlessAVSink, _lease_payload
from .paired_av import LiveAVBinding


class PresentationInvalidated(RuntimeError):
    """The payload must receive an explicit interrupted-discard disposition."""


class _DecodedOutput:
    """Actual FFmpeg PCM output of a growing private NUT span, never a device.

    Bounded file-follow/pipe feed; no queue or media rewriting. All stdout bytes,
    including those drained after kill, reach the accounting callback. EOF after
    process termination bounds cessation; an epoch selector is not the evidence.
    """
    def __init__(self, child, on_pcm, *, autostart=True):
        import subprocess
        self.child = child; self.on_pcm = on_pcm
        self.stop_feed = threading.Event(); self.errors = []; self._started=False
        self.eof_monotonic = None; self._source = open(child.output, 'rb', buffering=0)
        self._stderr = __import__('tempfile').TemporaryFile()
        try:
            self.process = subprocess.Popen(['ffmpeg','-v','error','-nostdin',
                '-probesize','32','-analyzeduration','1','-f','nut','-nofind_stream_info',
                '-i','pipe:0','-map','0:a:0','-threads','1','-c:a','pcm_s16le',
                '-f','s16le','pipe:1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, start_new_session=True)
            self._feed = threading.Thread(target=self._follow, name='h3-sink-nut-input')
            self._read = threading.Thread(target=self._pcm, name='h3-sink-decoded-pcm')
            if autostart:self.start()
        except BaseException:
            if hasattr(self, 'process'):
                self.process.kill(); self.process.wait()
            self._source.close(); self._stderr.close()
            raise

    def start(self):
        if self._started:raise RuntimeError('decoder observer already started')
        self._started=True
        self._read.start(); self._feed.start()

    def _follow(self):
        import os, select
        try:
            fd = self.process.stdin.fileno(); os.set_blocking(fd, False)
            total = 0
            while not self.stop_feed.is_set():
                data = self._source.read(65536)
                if not data:
                    if self.child._encoder.poll() is not None:break
                    self.stop_feed.wait(.005); continue
                total += len(data)
                if total > 256*1024*1024:raise ValueError('bounded headless NUT span exceeded')
                view = memoryview(data)
                while view and not self.stop_feed.is_set():
                    if select.select([], [fd], [], .05)[1]:
                        try:view = view[os.write(fd, view):]
                        except BlockingIOError:pass
        except BaseException as exc:
            if not self.stop_feed.is_set():self.errors.append(type(exc).__name__)
        finally:
            self.process.stdin.close(); self._source.close()

    def _pcm(self):
        import os
        pending = b''
        try:
            while True:
                data = os.read(self.process.stdout.fileno(), 4096)
                if not data:break
                data = pending+data; aligned = len(data)//2*2
                if aligned:self.on_pcm(data[:aligned], time.monotonic())
                pending = data[aligned:]
            if pending:raise ValueError('unaligned decoded PCM EOF')
            self.eof_monotonic = time.monotonic()
        except BaseException as exc:
            self.errors.append(type(exc).__name__)
        finally:self.process.stdout.close()

    def finish(self, *, interrupted=False):
        import subprocess
        if not self._started:
            # A stale constructor never fed any media; no decoded output claim.
            self.stop_feed.set()
            if self.process.poll() is None:self.process.kill()
            self.process.wait(timeout=2)
            self.process.stdin.close();self.process.stdout.close()
            self._source.close();self._stderr.close()
            return None
        if interrupted:
            self.stop_feed.set()
            if self.process.poll() is None:self.process.kill()
        self._feed.join(2 if interrupted else self.child.timeout)
        if self._feed.is_alive():
            self.stop_feed.set(); self.process.kill()
        try:self.process.wait(timeout=2 if interrupted else self.child.timeout)
        except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait(timeout=2)
            self.errors.append('decoder_timeout')
        self._read.join(2); self._feed.join(2)
        self._stderr.close()
        if (self._read.is_alive() or self._feed.is_alive() or self.errors
                or self.eof_monotonic is None
                or (not interrupted and self.process.returncode != 0)):
            raise RuntimeError('actual headless PCM output termination unconfirmed')
        return self.eof_monotonic


class EpochAVSink:
    def __init__(self, root, binding, *, sink_factory=HeadlessAVSink, observe_output=False,
                 paired_cuts=False, **options):
        from pathlib import Path
        if type(binding) is not LiveAVBinding or not callable(sink_factory):
            raise ValueError('explicit paired binding/sink factory required')
        self.root = Path(root).resolve(strict=True); self.binding = binding
        if type(paired_cuts) is not bool:raise ValueError('explicit paired cut selection required')
        self.supports_paired_cuts=paired_cuts
        self.review_archive=options.get('review_archive')
        self._factory = sink_factory; self._options = options
        self._origin=time.monotonic();self._read_events=[];self._read_dropped=0;self._partial=0
        self._lock = threading.Lock(); self._serial = threading.Lock()
        self._floor = 0; self._invalid_sources = set()
        self._child = None; self._closed = self._finished = False
        self._index = self._frame = 0
        self._events = []; self._discarded = self._accepted = 0
        self._first = None
        if type(observe_output) is not bool:raise ValueError('explicit output observation required')
        self._observe = observe_output; self._observer = None
        self._regions = []; self._audio = {}; self._output_cursor = 0
        self._audio_cv = threading.Condition(self._lock)

    def _current(self, envelope):
        cut=getattr(envelope, 'presentation_cut', None)
        if cut is not None:
            if not self.supports_paired_cuts:raise ValueError('paired cut was not selected')
            if envelope.presentation_epoch != self._floor:return False
            if cut.previous_epoch>=self._floor or cut.interrupted_source_epoch not in self._invalid_sources:
                raise ValueError('cut not bound to local interruption')
        start=envelope.sample_start if cut is None else cut.frame*2000
        return (envelope.presentation_epoch >= self._floor
                and not any(r.kind == 'speech' and r.source_epoch in self._invalid_sources
                            and r.sample_stop>start for r in envelope.regions))

    def should_discard(self, lease):
        envelope, _ = _lease_payload(lease)
        with self._lock:
            if self._closed or self._finished:
                raise RuntimeError('epoch sink closed')
            return not self._current(envelope)

    def invalidate(self, new_epoch, *, interrupted_source_epoch=None):
        """Invalidate first, then stop both tracks in the old encoder.

        Return times distinguish local admission invalidation from process stop.
        Neither is proof of acoustic/device latency. Never hold the state lock
        while reaping processes or waiting for writers.
        """
        if (type(new_epoch) is not int or not 0 < new_epoch < 2**63
                or (interrupted_source_epoch is not None and
                    (type(interrupted_source_epoch) is not int
                     or not 0 <= interrupted_source_epoch < 2**63))):
            raise ValueError('bounded presentation/source epochs required')
        detected = time.monotonic()
        with self._lock:
            if self._closed or self._finished or new_epoch <= self._floor:
                raise ValueError('new monotonic presentation epoch required')
            if len(self._events) >= self.binding.max_segments:
                raise RuntimeError('finite interruption limit')
            self._floor = new_epoch
            if interrupted_source_epoch is not None:
                self._invalid_sources.add(interrupted_source_epoch)
            child, self._child = self._child, None
            observer, self._observer = self._observer, None
            old_audio = self._audio.get(interrupted_source_epoch, {})
            samples_at_interrupt = old_audio.get('samples', 0)
            invalidated = time.monotonic()
            row = dict(epoch=new_epoch, detected_monotonic=detected,
                       invalidated_monotonic=invalidated, stopped_monotonic=None,
                       global_frame=self._frame, global_index=self._index,
                       reason='user_interrupt', clock='local_monotonic')
            self._events.append(row)
        try:
            output_eof = None if observer is None else observer.finish(interrupted=True)
            if child is not None:
                child.cancel()
        except BaseException:
            with self._lock:
                self._closed = True
            raise
        with self._lock:
            row['stopped_monotonic'] = time.monotonic()
            old_audio = self._audio.get(interrupted_source_epoch, {})
            row.update(old_sink_last_pcm_monotonic=old_audio.get('last'),
                       old_sink_eof_monotonic=output_eof,
                       old_pcm_samples_after_interrupt=old_audio.get('samples', 0)-samples_at_interrupt,
                       stop_upper_bound_seconds=None if output_eof is None else output_eof-detected,
                       actual_decoded_sink_observed=observer is not None)
            return dict(row)

    def _check_sequence(self, envelope):
        if (envelope.binding != self.binding or envelope.index != self._index
                or envelope.frame_start != self._frame):
            raise ValueError('global paired sequence cannot rewind or skip')

    def discard(self, lease):
        """Advance transport accounting only; never submit payload to FFmpeg."""
        envelope, _ = _lease_payload(lease)
        with self._serial, self._lock:
            if self._closed or self._finished or self._current(envelope):
                raise ValueError('only explicitly invalidated payloads may be discarded')
            self._check_sequence(envelope)
            self._index += 1; self._frame = envelope.frame_stop; self._discarded += 1

    def accept(self, path, lease):
        envelope, _ = _lease_payload(lease)
        with self._serial:
            with self._lock:
                if self._closed or self._finished:
                    raise RuntimeError('epoch sink closed')
                self._check_sequence(envelope)
                if not self._current(envelope):
                    raise PresentationInvalidated('old presentation epoch')
                child = self._child
                cut=getattr(envelope, 'presentation_cut', None)
                start=envelope.frame_start if cut is None else cut.frame
                if cut is not None and child is not None:
                    raise ValueError('cut requires an explicit new discontinuity span')
            with self._lock:
                self._regions.extend(envelope.regions)
                if len(self._regions) > self.binding.max_segments*8:
                    raise ValueError('bounded decoded-source accounting required')
            if child is None:
                candidate = self._factory(self.root, self.binding,
                    discontinuity_origin=(envelope.index, start), **self._options)
                with self._lock:
                    current = (not self._closed and not self._finished and self._current(envelope))
                    if current:
                        self._child = candidate
                        child = candidate
                if not current:
                    candidate.cancel()
                    raise PresentationInvalidated('invalidated during encoder creation')
                if self._observe:
                    cursor = [start*2000]
                    def observed(pcm, when):
                        self._record_output(cursor[0], pcm, when)
                        cursor[0] += len(pcm)//2
                    try:
                        observer = _DecodedOutput(child, observed, autostart=False)
                    except BaseException:
                        with self._lock:stale=not self._current(envelope) or self._closed
                        if stale:
                            candidate.cancel()
                            raise PresentationInvalidated('invalidated during decoder construction') from None
                        raise
                    with self._lock:
                        valid = self._current(envelope) and not self._closed
                        if valid:
                            self._observer = observer
                            observer.start()
                    if not valid:
                        observer.finish(interrupted=True); candidate.cancel()
                        raise PresentationInvalidated('invalidated during sink decoder creation')
            try:
                result = child.accept(path, lease,
                    **({'presentation_start_frame':cut.frame} if cut is not None else {}))
            except BaseException:
                with self._lock:
                    stale = not self._current(envelope)
                if stale:
                    raise PresentationInvalidated('invalidated during paired writes') from None
                raise
            with self._lock:
                if not self._current(envelope):
                    raise PresentationInvalidated('invalidated after paired writes')
                self._index += 1; self._frame = envelope.frame_stop
                if cut is None:self._accepted += 1
                else:self._partial += 1
                if self._first is None:
                    self._first = child.first_av_submission_monotonic
                return result

    def _record_output(self, start, pcm, when):
        import hashlib
        stop = start+len(pcm)//2
        with self._audio_cv:
            if len(self._read_events)<4096:
                self._read_events.append(dict(sample_start=start,sample_stop=stop,observed_monotonic=when))
            else:self._read_dropped+=1
            for region in self._regions:
                if region.kind != 'speech':continue
                lo, hi = max(start, region.sample_start), min(stop, region.sample_stop)
                if lo >= hi:continue
                row = self._audio.setdefault(region.source_epoch,
                    dict(samples=0, digest=hashlib.sha256(), first=when, last=when))
                row['samples'] += hi-lo; row['last'] = when
                row['digest'].update(pcm[(lo-start)*2:(hi-start)*2])
            self._audio_cv.notify_all()

    def wait_for_speech_output(self, epoch, deadline):
        with self._audio_cv:
            while epoch not in self._audio:
                if self._closed or self._finished:raise RuntimeError('sink ended before speech output')
                left = deadline-time.monotonic()
                if left <= 0:raise TimeoutError('actual decoded speech output deadline')
                self._audio_cv.wait(min(.05, left))
            return self._audio[epoch]['first']

    @property
    def first_av_submission_monotonic(self):
        with self._lock:
            return self._first

    @property
    def timing(self):
        with self._lock:
            return dict(clock='local_monotonic',clock_domain='mac',origin_monotonic=self._origin,
                        accepted=self._accepted,partially_presented_interrupted=self._partial,
                        decoded_reads=list(self._read_events),decoded_reads_dropped=self._read_dropped,
                        review=None if self.review_archive is None else self.review_archive.snapshot(),
                        discarded_interrupted=self._discarded,
                        discontinuities=[dict(row) for row in self._events],
                        global_frame=self._frame, global_index=self._index,
                        presentation_complete=self._discarded == 0 and not self._events,
                        decoded_source_output={str(k):dict(samples=v['samples'],
                            sha256=v['digest'].hexdigest(), first_monotonic=v['first'],
                            last_monotonic=v['last']) for k,v in self._audio.items()},
                        device_output_verified=False)

    def finish(self):
        with self._serial:
            with self._lock:
                if self._closed or self._finished:
                    raise RuntimeError('epoch sink closed')
                child = self._child; observer = self._observer
            output = None if child is None else child.finish()
            if observer is not None:observer.finish()
            with self._lock:
                self._finished = True
            return output

    def cancel(self):
        self.close()

    def close(self):
        with self._lock:
            self._closed = True
            child, self._child = self._child, None
            observer, self._observer = self._observer, None
            self._audio_cv.notify_all()
        try:
            if observer is not None and observer.eof_monotonic is None:
                observer.finish(interrupted=True)
        finally:
            if child is not None:child.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
