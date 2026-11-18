"""Offline PNG candidate audit. No network, provider, or launcher integration."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import tempfile
import zlib

MAX_PIXELS = 10_000_000
MAX_OUTPUT = 40_000_000
MAX_SOURCE = 40_000_000


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _header(data: bytes) -> tuple[int, int, int, int]:
    if len(data) < 33 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[8:16] != b'\x00\x00\x00\rIHDR':
        raise ValueError('PNG IHDR required')
    payload = data[16:29]
    if zlib.crc32(b'IHDR' + payload) != struct.unpack('>I', data[29:33])[0]:
        raise ValueError('IHDR CRC mismatch')
    w, h, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', payload)
    if not w or not h or w * h > MAX_PIXELS or w * h * 4 > MAX_OUTPUT:
        raise ValueError('pixel/output bound exceeded')
    if depth != 8 or color not in (2, 6) or compression or filtering or interlace not in (0, 1):
        raise ValueError('only 8-bit RGB/RGBA PNG supported')
    return w, h, depth, color


def _decode(path: Path, width: int, height: int, ffmpeg: str) -> bytes:
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT, MAX_OUTPUT))
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(path),
                        '-frames:v', '1', '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'],
                       stdout=output, stderr=errors, timeout=20, check=True, preexec_fn=limits)
        if output.tell() != width * height * 4:
            raise ValueError('decoded output size mismatch')
        output.seek(0)
        return output.read(MAX_OUTPUT + 1)


def _rgb_png(width: int, height: int, rgb: bytes) -> bytes:
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    stride = width * 3
    scan = b''.join(b'\0' + rgb[y * stride:(y + 1) * stride] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(scan)) + chunk(b'IEND', b''))


def audit_and_stage(source: Path, destination: Path, *, ffmpeg: str = 'ffmpeg') -> dict:
    """Create a fresh private directory; never flatten transparency or modify source.

    RGB losslessness is relative to FFmpeg's decoded RGB bytes, not embedded
    color-profile interpretation. Original PNG metadata is preserved in source.png.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if source.stat().st_size > MAX_SOURCE:
        raise ValueError('source byte bound exceeded')
    with source.open('rb') as stream:
        data = stream.read(MAX_SOURCE + 1)
    if len(data) > MAX_SOURCE:
        raise ValueError('source byte bound exceeded')
    w, h, depth, color = _header(data)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    # Decode the exact immutable snapshot, not a potentially changing source path.
    snapshot = destination / 'source.png'
    snapshot.write_bytes(data)
    snapshot.chmod(0o400)
    rgba = _decode(snapshot, w, h, ffmpeg)
    alpha = rgba[3::4]
    counts = {'opaque_255': alpha.count(255), 'transparent_0': alpha.count(0)}
    counts['intermediate_1_254'] = w * h - sum(counts.values())
    result = {
        'source_path': str(source), 'source_copy': str(snapshot), 'source_sha256': _digest(data),
        'source_bytes': len(data), 'width': w, 'height': h, 'bit_depth': depth,
        'png_color_type': color, 'mode': 'RGBA' if color == 6 else 'RGB', 'alpha_counts': counts,
        'decoded_rgba_sha256': _digest(rgba), 'operation': 'COPY_NOT_UPLOAD',
        'self_shot_consent': 'USER asserted; not externally verified',
        'same_face_position': 'USER asserted; not independently verified',
        'candidate_quality': 'unverified; current original-SHA launcher rejects substitution',
        'prep_binding': {'short_edge': 1024, 'grid': 32, 'expected_size': [1024, 1824],
                         'actual_native_gpu_validation': 'required later; not run'},
        'candidate_only_prompt': 'Preserve the reference identity, dark suit, white shirt, striped tie, and windmill background. Natural conversational motion.',
    }
    if counts['opaque_255'] == w * h:
        rgb = bytearray(w * h * 3)
        for channel in range(3):
            rgb[channel::3] = rgba[channel::4]
        candidate = destination / 'candidate-rgb.png'
        encoded = _rgb_png(w, h, bytes(rgb))
        candidate.write_bytes(encoded)
        candidate.chmod(0o400)
        if _decode(candidate, w, h, ffmpeg) != rgba:
            raise ValueError('RGB round-trip mismatch')
        result.update(candidate_path=str(candidate), candidate_sha256=_digest(encoded),
                      decoded_rgb_sha256=_digest(bytes(rgb)), rgb_round_trip_verified=True)
    else:
        result['candidate_path'] = None
        result['transparency_policy'] = 'No flattening; explicit compositing decision required'
    metadata = destination / 'metadata.json'
    metadata.write_text(json.dumps(result, indent=2) + '\n')
    metadata.chmod(0o400)
    destination.chmod(0o500)
    return result
