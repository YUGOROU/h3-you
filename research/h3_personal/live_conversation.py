"""CPU-only, serialized dynamic conversation protocol (no native adapter).

LiveConversation(root, binding=LiveAVBinding, context=ResamplingContext) owns one
admission controller for its entire lifetime. attach(PCMIngress), prepare(),
complete_native(prepared, result), retire_speech(), request_stop() are explicit
boundaries. prepare returns None on WAIT; inspect exposes end/limit/busy.
step(backend) is a convenience, NOT a native owner factory: the caller must keep
one outer native owner alive, including across nonfinal listening.

Backend result: video:bytes (already encoded), count:int, committed_frame:int,
emitted_frame:int, conditioning_prefix_sha256:str|None, receipt_sha256:str.
Receipt authenticity/native decoding is the adapter's responsibility; this
module validates bindings/counts, not codec contents or receipt signatures.
Returned Packet is not a sink ACK. An optional emitter is called once, after
native/decoded ACK; emitter failure is terminal and never replays the backend.
All deadlines are operation-checked, not background timers. No async/thread API.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

from .paired_av import LiveAVBinding, AudioRegion, encode
from .streaming_admission import StreamingAdmission, NativeCommitEvidence
from .streaming_pcm import PCMIngress, _int
from .streaming_trial import manifest_for, write_wav
from .streaming_waveform import read_original_prefix
from .streaming_native_stages import validate_manifest


def _hash(body):
    return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class Prepared:
    ticket: object
    manifest_json: str

    @property
    def manifest(self):
        import json
        return json.loads(self.manifest_json)


@dataclass(frozen=True)
class Packet:
    envelope: object
    video: bytes
    pcm: bytes


class LiveConversation:
    def __init__(self, root, *, binding, context, max_idle_windows=12,
                 max_turns=4, max_source_bytes=16*1024*1024,
                 max_total_source_bytes=32*1024*1024, timeout=120,
                 emitter=None, extended_endurance=False, long_session=False):
        if type(binding) is not LiveAVBinding:
            raise ValueError('paired binding required')
        binding.__post_init__()
        _int(max_source_bytes, 2, 64*1024*1024)
        if type(long_session) is not bool:
            raise ValueError('explicit long session selection required')
        _int(max_total_source_bytes, max_source_bytes, 10800*96000 if long_session else 256*1024*1024)
        if emitter is not None and not callable(emitter):
            raise ValueError('callable bounded emitter required')
        if binding.max_total_frames<102*binding.max_segments+22:
            raise ValueError('frame budget must cover the declared native window budget')
        # Capacity reservations bound appendable ingress too, not just bytes
        # observed at attachment. Ingress itself enforces those capacities.
        self.binding = binding
        self.control = StreamingAdmission(binding.session, context=context,
            max_windows=binding.max_segments, max_idle_windows=max_idle_windows,
            max_turns=max_turns, timeout=timeout, extended_endurance=extended_endurance,
            long_session=long_session)
        self.root = Path(root).resolve()
        if self.root.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError('private source media must remain outside repository')
        self.root.mkdir(mode=0o700, exist_ok=False)
        self._source_limit = max_source_bytes
        self._total_limit = max_total_source_bytes
        self._capacity = 0
        self._stream = None
        self._sources = []
        self._spans = []
        self._active = None
        self._cursor = 0
        self._idle_epoch = -1
        self._idle_index = 0
        self._last_role = None
        self._failed = False
        self._closed = False
        self._owned_paths=[]
        self._emitter = emitter
        self._pending_interruption = None

    def _live(self):
        if self._failed or self._closed:
            raise RuntimeError('terminal conversation; never replay native')

    def attach(self, stream):
        self._live()
        if type(stream) is not PCMIngress:
            raise ValueError('PCMIngress required')
        b = stream.binding
        if (b.voice_binding_sha256 != self.binding.profile_sha256
                or b.sample_rate != 48000 or b.channels != 1
                or stream.max_bytes > self._source_limit
                or self._capacity + stream.max_bytes > self._total_limit):
            raise ValueError('voice/48k mono/source capacity mismatch')
        self.control.attach(stream)
        self._capacity += stream.max_bytes
        self._stream = stream
        self._sources.append(stream)  # retain through decoder tail delivery

    def inspect(self):
        self._live()
        return self.control.inspect()

    def retire_speech(self):
        self._live()
        self.control.finish_speech()
        self._stream = None

    def request_speech_interrupt(self, epoch):
        """Request only. Caller must already invalidate BOTH presentation tracks.

        Safe to request during synchronous native work; that work still completes
        exactly once. This method does not interrupt CUDA or discard AV packets.
        """
        self._live()
        return self.control.request_speech_interrupt(epoch)

    def retire_interrupted_speech(self):
        """Serial owner calls after decoded ACK, before its next prepare().

        Retain old source and spans for the decoder's five-frame lookahead; the
        receiver must quarantine interrupted presentation separately. Never label
        this path EOS, flush the native decoder, or rewind the global clock.
        """
        self._live()
        marker = self.control.retire_interrupted_speech()
        if marker is not None:
            self._pending_interruption = marker
        self._stream = None
        return marker

    def cancel_pending_speech(self):
        self._live()
        stream=self._stream
        self.control.cancel_pending_speech()
        self._stream=None
        self._sources.remove(stream)
        self._capacity-=stream.max_bytes

    def close(self):
        if self._closed:return
        self._closed=True
        if self.control.snapshot()['phase']!='closed':self.control.fail()
        for stream in self._sources:stream.cancel()
        self._sources.clear();self._spans.clear();self._stream=None;self._active=None
        self._capacity=0
        for path in self._owned_paths:path.unlink(missing_ok=True)
        self._owned_paths.clear()

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()

    def request_stop(self):
        self._live()
        self.control.request_stop()

    def prepare(self):
        import json
        self._live()
        decision = self.control.inspect()
        if decision.status == 'wait':
            return None
        if decision.status != 'ready':
            raise RuntimeError('admission: ' + decision.status)
        ticket = self.control.reserve(decision)
        try:
            wav = self.root / f'prefix-{ticket.sequence}.wav'
            pcm = (bytes(96000) if decision.audio is None else
                   read_original_prefix(self._stream, decision.audio))
            write_wav(wav, pcm)
            self._owned_paths.append(wav)
            m = manifest_for(ticket, conversation_id=self.binding.session,
                profile_sha256=self.binding.profile_sha256, wav=wav,
                max_windows=self.binding.max_segments,
                native_window_prefix='live_'+_hash(self.binding.session.encode())[:24],
                long_session=self.binding.long_session)
            if decision.role == 'listening':
                if self._last_role != 'listening':
                    self._idle_epoch += 1
                    self._idle_index = 0
                m['streaming_audio']['logical_source_id'] = f'idle-epoch-{self._idle_epoch}'
                m['streaming_audio']['logical_index'] = self._idle_index
            if self._pending_interruption is not None:
                m['streaming_audio']['previous_speech_interrupted'] = dict(self._pending_interruption)
            validate_manifest(m)
            self._active = Prepared(ticket, json.dumps(m, sort_keys=True))
            return self._active
        except Exception:
            self._fail()
            raise

    def _fail(self):
        self._failed = True
        self.control.fail()

    def _compose(self, start, stop):
        parts, regions = [], []
        for a, z, stream, onset in self._spans:
            lo, hi = max(a, start), min(z, stop)
            if lo >= hi:
                continue
            if stream is None:
                body = bytes((hi-lo)*2)
                region = AudioRegion('silence', lo, hi)
            else:
                offset = lo-onset
                read = stream.read(offset, hi-onset)
                if read.status != 'ready':
                    raise ValueError('unavailable future PCM cannot be padded')
                body = read.pcm
                b = stream.binding
                region = AudioRegion('speech', lo, hi, b.source_id,
                                     b.epoch, offset, _hash(body))
            parts.append(body)
            regions.append(region)
        return b''.join(parts), tuple(regions)

    def complete_native(self, prepared, result):
        self._live()
        if prepared is not self._active or prepared is None:
            self._fail()
            raise ValueError('current preparation required; native outcome uncertain')
        t = prepared.ticket
        d = t.decision
        end = d.window.end-(0 if d.conversation_final else 5)
        try:
            expected = None if d.audio is None else d.audio.prefix_sha256
            if (type(result['video']) is not bytes or not result['video']
                    or type(result['count']) is not int
                    or result['count'] != end-self._cursor
                    or type(result['emitted_frame']) is not int
                    or result['emitted_frame'] != end
                    or result['conditioning_prefix_sha256'] != expected):
                raise ValueError('native decoded evidence mismatch')
            self.control.commit(t, NativeCommitEvidence(result['committed_frame'],
                                expected, result['receipt_sha256']))
            # Append only newly committed ranges, not overlapping conditioning.
            start = 0 if t.sequence == 0 else d.window.start+22
            lo, hi = start*2000, d.window.end*2000
            if d.audio is None:
                self._spans.append((lo, hi, None, None))
            else:
                onset = d.audio.onset_frame*2000
                received = self._stream.snapshot()['received_samples']
                speech_stop = min(hi, onset+received)
                if speech_stop < hi and not d.audio.eos:
                    raise ValueError('unsealed source cannot supply silence')
                if speech_stop > lo:
                    self._spans.append((lo, speech_stop, self._stream, onset))
                if speech_stop < hi:
                    self._spans.append((max(lo, speech_stop), hi, None, None))
            pcm, regions = self._compose(self._cursor*2000, end*2000)
            envelope = encode(self.binding, t.sequence, self._cursor, end,
                              result['video'], pcm, regions, d.conversation_final)
            self.control.ack_decoded(t, end)
            packet = Packet(envelope, result['video'], pcm)
            self._cursor = end
            self._last_role = d.role
            if d.role == 'listening':
                self._idle_index += 1
            self._active = None
            if 'previous_speech_interrupted' in prepared.manifest['streaming_audio']:
                self._pending_interruption = None
            # All native ranks/decoder have ACKed this input; never retain a
            # growing collection of full-prefix WAV copies across windows.
            path=Path(prepared.manifest['wav'])
            path.unlink();self._owned_paths.remove(path)
            if self._emitter is not None:
                self._emitter(packet)
            return packet
        except Exception:
            self._fail()
            raise

    def step(self, backend):
        from contextlib import nullcontext
        profiler = getattr(self, '_window_profiler', None)
        index = self.control.snapshot()['windows'] if profiler is not None else None
        def span(stage):
            return nullcontext() if profiler is None else profiler.span(index, stage)
        with span('driver_total'):
            with span('driver_prepare'):
                prepared = self.prepare()
            if prepared is None:
                return None
            try:
                with span('driver_backend'):
                    result = backend(prepared.manifest)
            except Exception:
                self._fail()
                raise
            with span('driver_complete'):
                return self.complete_native(prepared, result)
