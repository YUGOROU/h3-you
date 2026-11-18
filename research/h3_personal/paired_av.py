"""CPU-only paired AV wire contract; receipts are not sink acknowledgements.

No transport or sink is executed here. Lease attributes form a structural sink
seam; existing sinks with nominal type guards need an explicit adapter.
"""
from dataclasses import dataclass
import hashlib
import math
import time


class ProtocolError(ValueError):
    pass


class CreditError(ProtocolError):
    """Recoverable backpressure: the offered segment was not admitted."""


def _int(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ProtocolError('integer outside bounds')


def _text(value):
    if type(value) is not str or not value or len(value) > 256:
        raise ProtocolError('bounded nonempty identity required')


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ProtocolError('lowercase SHA256 required')


def _digest(body):
    return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class LiveAVBinding:
    session: str
    profile_sha256: str
    epoch: int
    sample_rate: int = 48000
    channels: int = 1
    fps: int = 24
    max_segments: int = 128
    max_total_frames: int = 13100
    max_inflight_bytes: int = 36 * 1024 * 1024
    long_session: bool = False

    def __post_init__(self):
        _text(self.session)
        _sha(self.profile_sha256)
        _int(self.epoch, 0, 2**63 - 1)
        for value, exact in ((self.sample_rate, 48000), (self.channels, 1), (self.fps, 24)):
            _int(value, exact, exact)
        # Immutable session validation selects legacy or explicit three-hour caps.
        if type(self.long_session) is not bool:
            raise ValueError('explicit long_session selection required')
        if self.long_session:
            _int(self.max_segments, 1, 2542)
            _int(self.max_total_frames, 1, 259306)
        else:
            _int(self.max_segments, 1, 144)
            _int(self.max_total_frames, 1, 14710)
        _int(self.max_inflight_bytes, 1, 2**63 - 1)


@dataclass(frozen=True)
class AudioRegion:
    """Global half-open samples; speech SHA hashes this region's original PCM.

    source_offset is in samples within source_id/source_epoch. Silence has no
    source metadata and is checked byte-for-byte zero, never implicit padding.
    """
    kind: str
    sample_start: int
    sample_stop: int
    source_id: str | None = None
    source_epoch: int | None = None
    source_offset: int | None = None
    source_sha256: str | None = None

    def __post_init__(self):
        _int(self.sample_start, 0, 29420000)
        _int(self.sample_stop, self.sample_start + 1, 29420000)
        if self.kind == 'speech' and type(self.kind) is str:
            _text(self.source_id)
            _int(self.source_epoch, 0, 2**63 - 1)
            _int(self.source_offset, 0, 2**63 - 1)
            _sha(self.source_sha256)
        elif type(self.kind) is str and self.kind == 'silence':
            if any(x is not None for x in (self.source_id, self.source_epoch, self.source_offset, self.source_sha256)):
                raise ProtocolError('silence has no source')
        else:
            raise ProtocolError('explicit silence/speech required')


@dataclass(frozen=True)
class PresentationCut:
    """Verified full packet; only the suffix is submitted to presentation."""
    frame: int
    previous_epoch: int
    interrupted_source_epoch: int

    def __post_init__(self):
        _int(self.frame, 1, 14709)
        _int(self.previous_epoch, 0, 2**63 - 1)
        _int(self.interrupted_source_epoch, 0, 2**63 - 1)


@dataclass(frozen=True)
class AVEnvelope:
    binding: LiveAVBinding
    index: int
    frame_start: int
    frame_stop: int
    sample_start: int
    sample_stop: int
    video_sha256: str
    video_size: int
    pcm_sha256: str
    pcm_size: int
    regions: tuple[AudioRegion, ...]
    conversation_final: bool
    audio_format: str = 's16le'
    padding_samples: int = 0
    # Application presentation generation, independent of fixed binding.epoch.
    # Zero preserves the original v1 wire encoding; no epoch inference is done.
    presentation_epoch: int = 0
    presentation_cut: PresentationCut | None = None

    def __post_init__(self):
        if type(self.binding) is not LiveAVBinding:
            raise ProtocolError('binding required')
        self.binding.__post_init__()
        b = self.binding
        _int(self.index, 0, b.max_segments - 1)
        _int(self.frame_start, 0, b.max_total_frames - 1)
        _int(self.frame_stop, self.frame_start + 1, min(self.frame_start + 124, b.max_total_frames))
        _int(self.sample_start, self.frame_start * b.sample_rate // b.fps, self.frame_start * b.sample_rate // b.fps)
        _int(self.sample_stop, self.frame_stop * b.sample_rate // b.fps, self.frame_stop * b.sample_rate // b.fps)
        _int(self.video_size, 1, 16 * 1024 * 1024)
        _int(self.pcm_size, (self.sample_stop - self.sample_start) * 2, (self.sample_stop - self.sample_start) * 2)
        _sha(self.video_sha256)
        _sha(self.pcm_sha256)
        if type(self.conversation_final) is not bool or type(self.audio_format) is not str or self.audio_format != 's16le':
            raise ProtocolError('final bool and original S16LE required')
        _int(self.padding_samples, 0, 0)
        _int(self.presentation_epoch, 0, 2**63 - 1)
        if type(self.regions) is not tuple or not 1 <= len(self.regions) <= min(8,self.sample_stop - self.sample_start):
            raise ProtocolError('bounded region tuple required')
        cursor = self.sample_start
        for region in self.regions:
            if type(region) is not AudioRegion:
                raise ProtocolError('AudioRegion required')
            region.__post_init__()
            if region.sample_start != cursor or region.sample_stop > self.sample_stop:
                raise ProtocolError('regions must partition audio exactly')
            cursor = region.sample_stop
        if cursor != self.sample_stop:
            raise ProtocolError('incomplete regions')
        if self.presentation_cut is not None:
            cut = self.presentation_cut
            if type(cut) is not PresentationCut:
                raise ProtocolError('exact presentation cut required')
            cut.__post_init__()
            if not (self.frame_start < cut.frame < self.frame_stop
                    and cut.previous_epoch < self.presentation_epoch):
                raise ProtocolError('cut must omit a strict prefix from an older presentation')
            sample = cut.frame * 2000
            for region in self.regions:
                if region.kind != 'speech':
                    continue
                if region.source_epoch <= cut.interrupted_source_epoch and region.sample_stop > sample:
                    raise ProtocolError('interrupted speech extends after cut')
                if region.source_epoch > cut.interrupted_source_epoch and region.sample_start < sample:
                    raise ProtocolError('cut would omit new speech head')
        if self.video_size + self.pcm_size > b.max_inflight_bytes:
            raise ProtocolError('segment exceeds byte budget')


def _bodies(envelope, video, pcm):
    if type(video) is not bytes or type(pcm) is not bytes:
        raise ProtocolError('immutable bytes required')
    if len(video) != envelope.video_size or len(pcm) != envelope.pcm_size:
        raise ProtocolError('body size mismatch')
    if _digest(video) != envelope.video_sha256 or _digest(pcm) != envelope.pcm_sha256:
        raise ProtocolError('body hash mismatch')
    for r in envelope.regions:
        part = pcm[(r.sample_start - envelope.sample_start)*2:(r.sample_stop - envelope.sample_start)*2]
        if r.kind == 'silence' and any(part):
            raise ProtocolError('nonzero silence')
        if r.kind == 'speech' and _digest(part) != r.source_sha256:
            raise ProtocolError('source region hash mismatch')


def encode(binding, index, frame_start, frame_stop, video, pcm, regions, final, *, presentation_epoch=0,
           presentation_cut=None):
    """Build immutable paired bytes with an explicit presentation generation.

    presentation_epoch is independent of binding.epoch and defaults to legacy 0.
    It identifies listening/mixed-tail video too; source epochs cannot replace it.
    """
    if type(binding) is not LiveAVBinding or type(video) is not bytes or type(pcm) is not bytes:
        raise ProtocolError('binding and bytes required')
    _int(frame_start, 0, binding.max_total_frames - 1)
    _int(frame_stop, frame_start + 1, min(frame_start + 124, binding.max_total_frames))
    if len(video) > 16 * 1024 * 1024 or len(pcm) != (frame_stop-frame_start)*4000:
        raise ProtocolError('body bounds')
    header = AVEnvelope(binding, index, frame_start, frame_stop,
                        frame_start*2000, frame_stop*2000, _digest(video), len(video),
                        _digest(pcm), len(pcm), regions, final,
                        presentation_epoch=presentation_epoch, presentation_cut=presentation_cut)
    _bodies(header, video, pcm)
    return header


@dataclass(frozen=True)
class VerifiedReceipt:
    envelope: AVEnvelope


@dataclass(frozen=True, eq=False)
class ReceivedAVLease:
    segment: AVEnvelope
    video: bytes
    audio_pcm: bytes


class Receiver:
    """Serialized caller API. Absolute monotonic deadline never renews.

    At most two pending+leased pairs; one sink lease at a time. CreditError is
    backpressure, all other protocol failures are terminal. No replay after a
    lease: uncertainty, terminal drop, cancellation or disconnect destroys credits.
    Explicit certain interrupted-discard releases a FIFO lease without pretending
    sink acceptance. Transport completion and presentation completion are separate.
    """
    def __init__(self, binding, *, deadline):
        if type(binding) is not LiveAVBinding:
            raise ProtocolError('binding required')
        binding.__post_init__()
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ProtocolError('finite future monotonic deadline required')
        self._binding, self._deadline = binding, deadline
        self._pending = []
        self._active = None
        self._bytes = self._index = self._frame = self._acked = self._discarded = self._partial_acked = 0
        self._final = False
        self._terminal = None
        self._sources = {}
        self._active_source = None
        self._max_source_epoch = -1

    @property
    def binding(self):
        return self._binding

    def _stop(self, reason):
        self._terminal = reason
        self._pending.clear()
        self._active = None
        self._bytes = 0
        self._sources.clear()

    def _check(self):
        if self._terminal is not None:
            raise ProtocolError(self._terminal)
        if time.monotonic() >= self._deadline:
            self._stop('deadline expired')
            raise ProtocolError(self._terminal)

    def offer(self, envelope, video, pcm):
        self._check()
        try:
            if type(envelope) is not AVEnvelope:
                raise ProtocolError('envelope required')
            envelope.__post_init__()
            if envelope.binding != self._binding or self._final:
                raise ProtocolError('stale binding or final conversation')
            if envelope.index != self._index or envelope.frame_start != self._frame:
                raise ProtocolError('duplicate, gap or reordered segment')
            _bodies(envelope, video, pcm)
            sources = dict(self._sources)
            active=self._active_source;epoch=self._max_source_epoch
            for r in envelope.regions:
                if r.kind == 'speech':
                    key = (r.source_id, r.source_epoch)
                    if key!=active:
                        if key in sources or r.source_epoch<=epoch or r.source_offset!=0:
                            raise ProtocolError('retired, nonfresh or stale logical source')
                        active=key;epoch=r.source_epoch
                    if key in sources and r.source_offset != sources[key]:
                        raise ProtocolError('source PCM replay or gap')
                    if any(k[0] == r.source_id and k[1] > r.source_epoch for k in sources):
                        raise ProtocolError('stale source epoch')
                    sources[key] = r.source_offset + r.sample_stop-r.sample_start
            size = len(video) + len(pcm)
            if len(self._pending) + (self._active is not None) >= 2 or self._bytes + size > self._binding.max_inflight_bytes:
                raise CreditError('receiver credits exhausted')
            self._pending.append(ReceivedAVLease(envelope, video, pcm))
            self._bytes += size
            self._index += 1
            self._frame = envelope.frame_stop
            self._final = envelope.conversation_final
            self._sources = sources
            self._active_source,self._max_source_epoch=active,epoch
            return VerifiedReceipt(envelope)
        except CreditError:
            raise
        except (ValueError, TypeError) as exc:
            self._stop('protocol failure')
            raise ProtocolError(str(exc)) from exc

    def lease(self):
        self._check()
        if self._active is not None or not self._pending:
            return None
        self._active = self._pending.pop(0)
        return self._active

    def _dispose(self, lease, certain, *, interrupted=False, partial=False):
        self._check()
        if type(certain) is not bool or not certain or self._active is None or lease is not self._active:
            self._stop('ambiguous or invalid interrupted discard' if interrupted else
                       'ambiguous or invalid sink ACK')
            raise ProtocolError(self._terminal)
        if not interrupted and ((lease.segment.presentation_cut is not None) != partial):
            self._stop('cut requires an honest partial disposition')
            raise ProtocolError(self._terminal)
        self._bytes -= len(lease.video) + len(lease.audio_pcm)
        if partial:
            self._partial_acked += 1
        elif interrupted:
            self._discarded += 1
        else:
            self._acked += 1
        self._active = None

    def ack(self, lease, certain=True):
        self._dispose(lease, certain)

    def ack_partial(self, lease, *, certain):
        """Release full verified FIFO bytes, acknowledging only declared suffix."""
        self._dispose(lease, certain, partial=True)

    def discard_interrupted(self, lease, *, certain):
        """Dispose the exact verified FIFO lease, not a sink ACK or epoch guess.

        Caller must independently establish that paired A/V presentation was
        invalidated for this interruption. Offered frame/source clocks retain
        their verified progress; the next pair must remain contiguous. No
        payload replay, native-state mutation or inferred source epoch occurs.
        """
        self._dispose(lease, certain, interrupted=True)

    def disconnect(self):
        self._stop('disconnected; no replay')

    def cancel(self):
        self._stop('cancelled')

    def drop(self):
        self._stop('dropped segment')

    def snapshot(self):
        if self._terminal is None and time.monotonic() >= self._deadline:
            self._stop('deadline expired')
        presentation_complete = self._final and self._acked == self._index and self._terminal is None
        transport_complete = (self._final and self._acked + self._discarded + self._partial_acked == self._index
                              and self._terminal is None)
        return dict(pending=len(self._pending), leased=int(self._active is not None),
                    inflight_bytes=self._bytes, verified=self._index,frame_stop=self._frame, sink_acked=self._acked,
                    discarded=self._discarded, partial_acked=self._partial_acked,
                    disposed=self._acked+self._discarded+self._partial_acked,
                    # A receiver drains after every verified pair is settled;
                    # this includes deliberate interrupted discards, not only
                    # presentation ACKs. Never confuse this with playback.
                    final=self._final, complete=transport_complete,
                    transport_complete=transport_complete, presentation_complete=presentation_complete,
                    terminal=self._terminal, deadline=self._deadline)
