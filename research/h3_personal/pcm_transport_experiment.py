"""Offline, bounded PCM16LE lossless transport experiment; not a wire protocol.

Records and decoded bytes belong to the caller. No files, devices, or providers
are accessed. Timings measure local calls, not realtime/speed acceptance.
"""
import hashlib
import struct
import time
import zlib

MAX_PCM_BYTES = 2_880_000
MAX_COMPRESSED_BYTES = MAX_PCM_BYTES + 65_536
_MAGIC = b"H3PX"
_VERSION = 1
# magic, version, codec, level, rate, channels, frames, raw size, payload size, SHA256
_HEADER = struct.Struct(">4sBBB I H I I I 32s")


def _validate(pcm, sample_rate, channels):
    if not isinstance(pcm, bytes):
        raise ValueError("PCM must be bytes")
    if type(sample_rate) is not int or not 1 <= sample_rate <= 384_000:
        raise ValueError("invalid sample rate")
    if type(channels) is not int or not 1 <= channels <= 32:
        raise ValueError("invalid channels")
    if not pcm or len(pcm) > MAX_PCM_BYTES or len(pcm) % (2 * channels):
        raise ValueError("PCM must contain bounded complete PCM16LE frames")


def _encode_record(pcm, sample_rate=48000, channels=1, level=1):
    """Return experimental v1 bytes; level=None selects the raw reference."""
    _validate(pcm, sample_rate, channels)
    if level is not None and (type(level) is not int or level not in (1, 3, 6)):
        raise ValueError("unsupported level")
    payload = pcm if level is None else zlib.compress(pcm, level)
    if len(payload) > MAX_COMPRESSED_BYTES:
        raise ValueError("compressed size cap")
    return _HEADER.pack(
        _MAGIC, _VERSION, int(level is not None), level or 0,
        sample_rate, channels, len(pcm) // (2 * channels), len(pcm),
        len(payload), hashlib.sha256(pcm).digest(),
    ) + payload


def _decode_record(record):
    """Validate one exact record, with bounded inflation; return original PCM."""
    if not isinstance(record, bytes) or not _HEADER.size <= len(record) <= _HEADER.size + MAX_COMPRESSED_BYTES:
        raise ValueError("record size cap or truncated header")
    magic, version, codec, level, rate, channels, frames, size, packed, digest = _HEADER.unpack_from(record)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("unsupported record")
    if (codec, level) not in ((0, 0), (1, 1), (1, 3), (1, 6)):
        raise ValueError("unsupported codec")
    if not 1 <= rate <= 384_000 or not 1 <= channels <= 32:
        raise ValueError("invalid PCM format")
    if not 0 < size <= MAX_PCM_BYTES or size != frames * 2 * channels:
        raise ValueError("invalid PCM size/count")
    if not 0 < packed <= MAX_COMPRESSED_BYTES or len(record) != _HEADER.size + packed:
        raise ValueError("payload size mismatch")
    payload = record[_HEADER.size:]
    if codec == 0:
        pcm = payload
    else:
        inflater = zlib.decompressobj()
        try:
            # Never flush: even a declared tiny size cannot cause unbounded output.
            pcm = inflater.decompress(payload, size + 1)
        except zlib.error as exc:
            raise ValueError("invalid compressed data") from exc
        if len(pcm) > size or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
            raise ValueError("compressed stream exceeds bound, is truncated, or has trailing data")
    if len(pcm) != size or hashlib.sha256(pcm).digest() != digest:
        raise ValueError("PCM integrity mismatch")
    return pcm


def benchmark_pcm(pcm: bytes, sample_rate=48000, channels=1, repeats=1):
    """Return JSON-safe sizes, savings, hashes and mean encode/decode seconds.

    Input is unmodified interleaved PCM16LE; sample_count counts all channel
    samples, frame_count counts simultaneous channel frames. Records include
    their header in record_bytes and savings. No payload is returned.
    """
    _validate(pcm, sample_rate, channels)
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("repeats must be 1..3")
    results = []
    digest = hashlib.sha256(pcm).hexdigest()
    for level in (None, 1, 3, 6):
        encode_seconds = decode_seconds = 0.0
        for _ in range(repeats):
            start = time.perf_counter()
            record = _encode_record(pcm, sample_rate, channels, level)
            encode_seconds += time.perf_counter() - start
            start = time.perf_counter()
            restored = _decode_record(record)
            decode_seconds += time.perf_counter() - start
            if restored != pcm:
                raise ValueError("roundtrip mismatch")
        results.append({
            "codec": "raw" if level is None else "zlib", "level": level,
            "payload_bytes": len(record) - _HEADER.size, "record_bytes": len(record),
            "saved_bytes": len(pcm) - len(record),
            "savings_fraction": 1 - len(record) / len(pcm),
            "encode_seconds_mean": encode_seconds / repeats,
            "decode_seconds_mean": decode_seconds / repeats,
            "decoded_sha256": hashlib.sha256(restored).hexdigest(),
            "decoded_sample_count": len(restored) // 2,
        })
    return {"experiment_version": _VERSION, "pcm_format": "s16le",
            "sample_rate": sample_rate, "channels": channels,
            "pcm_bytes": len(pcm), "sample_count": len(pcm) // 2,
            "frame_count": len(pcm) // (2 * channels), "sha256": digest,
            "repeats": repeats, "results": results}
