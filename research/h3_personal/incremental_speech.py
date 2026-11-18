"""CPU-only, single-owner speech seams. No IO, inference or durable logging.

Provider attestations are protocol inputs, NOT proof of watermark correctness.
Only a separately audited provider adapter may set produced_watermarked.
PCMIngress intentionally retains a finite whole source: backpressure is terminal
capacity until cancellation, not permission to silently drop audio.
"""
from dataclasses import dataclass, field
import hashlib

from .streaming_pcm import PCMIngress, PCMChunk, PCMStreamBinding, _int, _sha


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class TextRevision:
    epoch: int
    revision: int
    offset: int
    text: str = field(repr=False)  # replacement suffix after immutable prefix
    stable_stop: int
    prefix_sha256: str
    sha256: str
    eos: bool = False
    origin: str = 'asr_partial'


class IncrementalText:
    """Offsets count Unicode code points, not UTF-8 bytes or model tokens.

    Partial ASR is display/speculation only; only explicit provider final or
    application endpoint commits input. Tool JSON is never a reply channel.
    Stable means provider-committed prefix, not a repeated-hypothesis heuristic.
    """
    def __init__(self, epoch, *, channel, max_chars=160, max_revisions=1024):
        _int(epoch, 0, 1_000_000)
        _int(max_chars, 1, 4096); _int(max_revisions, 1, 65536)
        if channel not in ('asr', 'reply'):
            raise ValueError('explicit semantic channel required')
        self.epoch, self.channel = epoch, channel
        self.max_chars, self.max_revisions = max_chars, max_revisions
        self.text = ''; self.stable_stop = self.revision = 0
        self.closed = self.cancelled = False
        self.commit_origin = None

    def accept(self, event):
        if self.closed or self.cancelled:
            raise RuntimeError('terminal text seam')
        if type(event) is not TextRevision or event.epoch != self.epoch:
            raise ValueError('stale text epoch')
        _int(event.revision, self.revision+1, self.max_revisions)
        _int(event.offset, self.stable_stop, len(self.text))
        if type(event.text) is not str or type(event.eos) is not bool:
            raise ValueError('immutable text and explicit EOS required')
        allowed = ('asr_partial', 'asr_final', 'application_endpoint') if self.channel == 'asr' else ('human_reply',)
        if event.origin not in allowed:
            raise ValueError('wrong semantic channel; never speak tool JSON')
        if self.channel == 'asr' and event.eos != (event.origin != 'asr_partial'):
            raise ValueError('partial hypothesis is not committed final')
        prefix = self.text[:event.offset]
        candidate = prefix+event.text
        _sha(event.prefix_sha256); _sha(event.sha256)
        if text_hash(prefix) != event.prefix_sha256 or text_hash(candidate) != event.sha256:
            raise ValueError('text prefix/content hash mismatch')
        _int(event.stable_stop, self.stable_stop, len(candidate))
        if len(candidate) > self.max_chars:
            raise BufferError('text capacity; do not truncate')
        if event.eos and (not candidate.strip() or event.stable_stop != len(candidate)):
            raise ValueError('EOS must commit complete nonempty text')
        self.text, self.stable_stop = candidate, event.stable_stop
        self.revision, self.closed = event.revision, event.eos
        if event.eos:
            self.commit_origin = event.origin
        return self.text[:self.stable_stop]

    def committed_input(self):
        if self.channel != 'asr' or not self.closed or self.cancelled:
            raise RuntimeError('no committed ASR input')
        return self.text

    def cancel(self):
        self.cancelled = True
        self.text = ''; self.stable_stop = 0


@dataclass(frozen=True)
class ProducedPCM:
    chunk: PCMChunk = field(repr=False)
    text_revision: int
    text_sha256: str
    prefix_sha256: str
    chunk_sha256: str
    production: str  # produced_watermarked / complete_wav_segment
    watermark_applied: bool


class IncrementalSpeech:
    """Adapter into existing original-PCM contract; not a production TTS adapter.

    A reply revision is frozen at construction. New hypotheses/replies need a
    new source/epoch; never graft changed text onto already audible speech.
    Existing deadline, ordering, bounded retention, EOS hashes and cancel are
    delegated to PCMIngress. Owner serializes all calls and cancels on abandon.
    """
    def __init__(self, binding, reply, *, production='produced_watermarked', **limits):
        if type(reply) is not IncrementalText or reply.channel != 'reply' or reply.cancelled:
            raise ValueError('explicit human reply seam required')
        if not reply.closed:
            raise ValueError('validated committed reply required; tool argument deltas are not speech')
        if type(binding) is not PCMStreamBinding or binding.epoch != reply.epoch:
            raise ValueError('text/PCM epoch mismatch')
        if production not in ('produced_watermarked', 'complete_wav_segment'):
            raise ValueError('explicit production mode required')
        self.production = production
        self.revision, self.text_sha256 = reply.revision, text_hash(reply.text)
        self.ingress = PCMIngress(binding, **limits)
        self.samples = 0

    @property
    def genuinely_incremental(self):
        return self.production == 'produced_watermarked'

    def append(self, event):
        if type(event) is not ProducedPCM:
            raise ValueError('produced PCM envelope required')
        if (event.text_revision != self.revision or event.text_sha256 != self.text_sha256
                or event.production != self.production or event.watermark_applied is not True):
            raise ValueError('stale revision, production drift or unwatermarked audio')
        _sha(event.prefix_sha256); _sha(event.chunk_sha256)
        if (event.prefix_sha256 != self.ingress.prefix_digest(self.samples)
                or type(event.chunk) is not PCMChunk
                or hashlib.sha256(event.chunk.pcm).hexdigest() != event.chunk_sha256):
            raise ValueError('PCM prefix/chunk hash mismatch')
        self.samples = self.ingress.append(event.chunk)
        return self.samples

    def read(self, start, stop):
        return self.ingress.read(start, stop)

    def seal(self, *, epoch, revision, total_samples, sha256):
        if epoch != self.ingress.binding.epoch or revision != self.revision:
            raise ValueError('stale EOS')
        return self.ingress.seal(total_samples=total_samples, sha256=sha256)

    def cancel(self):
        self.ingress.cancel()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cancel()
