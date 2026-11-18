"""Bounded full-text native PCM bridge; protocol END is not model EOS.

The transport owns framing/END verification and must raise on invalid exhaustion.
No resampling, files, fallback, playback, or model watermark claim is made here.
"""
from dataclasses import dataclass
import hashlib
import math
import struct
import time

from .streaming_pcm import PCMChunk, PCMIngress
from .tts_preroll import PCMPrerollDecision, PCMPrerollGate, PCMPrerollPolicy

MAX_SAMPLES = 30 * 48000
MAX_NATIVE_CHUNK = 256 * 1024


@dataclass(frozen=True)
class NativePCMReceipt:
    stream: PCMIngress
    native_sha256: str
    converted_sha256: str
    native_samples: int
    converted_samples: int
    saturation_count: int
    native_format: str = '48000Hz mono float32le'
    converted_format: str = '48000Hz mono S16LE'
    completion: str = 'protocol_END'
    watermark: bool = False
    model_eos_verified: bool = False
    preroll: PCMPrerollDecision | None = None


def produce(tts, reply, epoch, binding, deadline, publish, is_current, *, preroll=None,
            observe=None, allow_interrupted=False):
    """Publish once at first PCM; seal only after verified remote exhaustion.

    Saturation counts samples whose rounded scaled value exceeds S16 range
    (including +1.0). Explicit preroll gates publication only, never adds PCM
    or changes clocks; short sources publish after verified seal. Its wait cap
    is observation-checked, not a hard interruption of blocking transport I/O.
    Cancellation is cooperative around transport calls;
    the transport must enforce the supplied deadline during blocking I/O.
    """
    stream = iterator = None
    gate = None
    decision = None
    published = False
    try:
        if type(allow_interrupted) is not bool:raise ValueError('explicit interrupted-source opt-in required')
        if observe is not None and not callable(observe):raise ValueError('callable coverage observer required')
        if preroll is not None:
            if type(preroll) is not PCMPrerollPolicy:
                raise ValueError('explicit immutable preroll policy required')
            gate = PCMPrerollGate(preroll)
        if (type(reply) is not str or not reply.strip() or len(reply) > 160
                or any(c in reply for c in '{}[]`')):
            raise ValueError('bounded human reply required')
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError('finite monotonic deadline required')
        if (type(epoch) is not int or epoch != binding.epoch
                or binding.sample_rate != 48000 or binding.channels != 1):
            raise ValueError('matching epoch and 48k mono binding required')

        def check():
            if not is_current():
                raise InterruptedError('stale or cancelled native synthesis')
            if time.monotonic() >= deadline:
                raise TimeoutError('native synthesis deadline')

        check()
        stream = PCMIngress(binding, max_bytes=MAX_SAMPLES*2,
                            max_samples=MAX_SAMPLES, max_chunk_bytes=64000,
                            max_chunks=3072,
                            timeout=min(300, max(.001, deadline-time.monotonic())))
        native_hash, converted_hash = hashlib.sha256(), hashlib.sha256()
        samples = index = saturated = native_chunks = 0
        iterator = iter(tts.stream(reply, epoch, deadline))
        while True:
            check()
            try:
                raw = next(iterator)
            except StopIteration:
                break  # Transport verifies END before returning StopIteration.
            check()
            native_chunks += 1
            if native_chunks > 1024:
                raise BufferError('native chunk count exceeds bound')
            if type(raw) is not bytes or len(raw) > MAX_NATIVE_CHUNK or len(raw) % 4:
                raise ValueError('bounded aligned immutable float32le chunk required')
            count = len(raw)//4
            if samples + count > MAX_SAMPLES:
                raise BufferError('aggregate native/converted PCM exceeds 30s')
            pcm = bytearray()
            for (value,) in struct.iter_unpack('<f', raw):
                if not math.isfinite(value):
                    raise ValueError('finite float32 samples required')
                scaled = round(max(-1.0, min(1.0, value))*32768)
                saturated += int(value < -1 or value > 1 or scaled > 32767)
                pcm.extend(struct.pack('<h', max(-32768, min(32767, scaled))))
            native_hash.update(raw)
            for offset in range(0, len(pcm), 64000):
                check()
                body = bytes(pcm[offset:offset+64000])
                stream.append(PCMChunk(binding, index, samples, body))
                samples += len(body)//2
                converted_hash.update(body)
                index += 1
                if gate is not None:
                    decision = gate.observe(samples, time.monotonic())
                if not published and (gate is None or decision.status == 'ready'):
                    published = True
                    publish(stream)
                if observe is not None:observe(stream)
        check()
        terminal = getattr(tts, 'last_completion', None)
        was_interrupted = type(terminal) is dict and terminal.get('finish_reason') == 'cancelled'
        if was_interrupted:
            from .voxcpm_endpoint_protocol import CANCEL_PROTOCOL, _exact_metadata
            wanted = dict(samples=samples, chunks=native_chunks, sha256=native_hash.hexdigest(),
                finish_reason='cancelled', model_eos_verified=False,
                cancel_protocol=CANCEL_PROTOCOL, epoch=epoch, native_closed=True)
            if not allow_interrupted or not _exact_metadata(terminal, wanted):
                raise ValueError('unaccepted interrupted native terminal')
        if not samples and not was_interrupted:
            raise ValueError('stream produced no playable PCM')
        # Close before sealing so a close failure cannot yield a success receipt.
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
        iterator = None
        check()
        if was_interrupted:
            # Never seal an intentionally truncated utterance as a complete WAV.
            # Keep existing PCM immutable and readable for the persistent native
            # tail. Unpublished/zero-prefix sources stay unpublished.
            stream.finish_interrupted(total_samples=samples, sha256=converted_hash.hexdigest())
            return NativePCMReceipt(stream, native_hash.hexdigest(), converted_hash.hexdigest(),
                                    samples, samples, saturated,
                                    completion='interrupted_protocol_END', preroll=decision)
        stream.seal(total_samples=samples, sha256=converted_hash.hexdigest())
        if observe is not None:observe(stream)
        if gate is not None:
            decision = gate.observe(samples, time.monotonic(), sealed=True)
        if not published:
            published = True
            publish(stream)
        return NativePCMReceipt(stream, native_hash.hexdigest(),
                                converted_hash.hexdigest(), samples, samples,
                                saturated, preroll=decision)
    except BaseException:
        # Cleanup must not mask cancellation, END validation, or publisher errors.
        if stream is not None:
            try:
                stream.cancel()
            except BaseException:
                pass
        if iterator is not None:
            try:
                close = getattr(iterator, 'close', None)
                if close is not None:
                    close()
            except BaseException:
                pass
        try:
            tts.close()
        except BaseException:
            pass
        raise
