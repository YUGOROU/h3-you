"""Opt-in ready-TTS sentence owner. No inference startup or audio devices.

Retain at most 30s per ingress; capacity exhaustion is terminal, never a
consumer-credit wait (a short first sentence must not deadlock preroll).
"""
import hashlib
from pathlib import Path
import re
import tempfile
import time

from .continuous_pcm import ContinuousPCM, native_wav

from .streaming_pcm import PCMIngress, PCMChunk

MAX_BYTES = 30 * 48000 * 2


def sentences(reply):
    if (type(reply) is not str or not reply.strip() or len(reply) > 160
            or any(c in reply for c in '{}[]`')):
        raise ValueError('bounded human reply required')
    parts = re.findall(r'[^。！？!?]*[。！？!?]+|[^。！？!?]+$', reply)
    if (not 1 <= len(parts) <= 4 or ''.join(parts) != reply
            or any(not p.strip('。！？!? \t\r\n') for p in parts)):
        raise ValueError('one to four nonempty sentences required')
    return parts


def produce(tts, reply, turn_id, binding, root, deadline, converter,
            validate_pcm, publish, is_current, *, validate_wav=None):
    """Opt-in sentence generation; default full-WAV caller behavior is unchanged.

    Non48k requires receipt-only validate_wav and one continuous resampler.
    publish must enqueue without waiting for playback/preroll acknowledgement.
    First PCM can wait for codec buffering, not necessarily sentence one.
    """
    parts = sentences(reply)
    stream = PCMIngress(binding, max_bytes=MAX_BYTES, max_samples=MAX_BYTES//2,
                        timeout=min(300, max(.001, deadline-time.monotonic())))
    total = index = 0
    resampler = None
    native_format = None
    source_frames = 0
    digest = hashlib.sha256()

    def check():
        if not is_current():
            raise InterruptedError('stale or cancelled sentence synthesis')
        if time.monotonic() >= deadline:
            raise TimeoutError('sentence synthesis deadline')

    def emit(pcm):
        nonlocal total, index
        check()
        if total + len(pcm) > MAX_BYTES:
            raise BufferError('aggregate sentence PCM exceeds 30s')
        for offset in range(0, len(pcm), 64000):
            check()
            body = pcm[offset:offset+64000]
            stream.append(PCMChunk(binding, index, total//2, body))
            total += len(body)
            index += 1
            digest.update(body)
            if index == 1:
                publish(stream)

    try:
        for number, sentence in enumerate(parts):
            check()
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                directory = Path(tmp)/'tts'
                sentence_id = f'{turn_id}-sentence-{number+1}'
                result = tts.synthesize(sentence, sentence_id, directory, deadline)
                check()
                # Both entry paths validate receipts BEFORE any PCM extraction.
                def inspect(path):
                    nonlocal native_format, source_frames
                    fmt, raw = native_wav(path)
                    if native_format is not None and fmt != native_format:
                        raise ValueError('native sentence format drift')
                    native_format = fmt
                    source_frames += len(raw)//fmt[1]
                    if source_frames > fmt[0]*30:
                        raise BufferError('aggregate source exceeds 30s')
                    return fmt, raw

                if validate_wav is not None:
                    path = validate_wav(result, sentence, sentence_id, directory)
                    fmt, raw = inspect(path)
                    if fmt[0] != 48000:
                        if resampler is None:
                            resampler = ContinuousPCM(fmt, root, deadline, check, emit)
                        resampler.feed(fmt, raw)
                        continue
                    pcm = validate_pcm(result, sentence, sentence_id, directory,
                                       converter, deadline)
                else:
                    def native_converter(path, until):
                        fmt, _ = inspect(path)
                        if fmt[0] != 48000:
                            raise ValueError('non48k requires receipt-only validate_wav')
                        return converter(path, until)
                    pcm = validate_pcm(result, sentence, sentence_id, directory,
                                       native_converter, deadline)
            emit(pcm)
            # No consumer ACK: ingress retains a bounded whole utterance.
        if resampler is not None:
            resampler.finish()
        check()
        stream.seal(total_samples=total//2, sha256=digest.hexdigest())
        return stream
    except BaseException:
        stream.cancel()
        raise
    finally:
        if resampler is not None:
            resampler.close()
