"""Generated-only, silent idle review on a borrowed, already warm runtime.

run_diagnostic(runtime, root, *, deadline) takes an absolute monotonic deadline
and a NEW directory outside the repository. It returns JSON-safe scalar fields;
per-window sample hashes/errors are in samples.json. No runtime start/close,
cloud, sender, playback, or device operations are performed.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import uuid

_LIMIT = 200 * 1024 * 1024


def _check(root, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('idle diagnostic deadline')
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if size > _LIMIT:
        raise ValueError('idle diagnostic exceeds 200 MiB')
    return size


def _exclusive(path, body):
    with path.open('xb') as f:
        f.write(body)


def _atomic_manifest(root, entries):
    target = root / 'native-manifest.json'
    temporary = root / 'native-manifest.json.tmp'
    temporary.write_text(json.dumps(entries, sort_keys=True))
    temporary.replace(target)


def _retain_native(root, deadline, frames, index, entries):
    """Store pre-codec RGB, never latents; verify every decoded byte."""
    import numpy as np
    pixels = np.stack([np.asarray(frame) for frame in frames])
    if (pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3
            or pixels.shape[2] <= 4):
        raise ValueError('unexpected native RGB geometry')
    count, height, width, _ = pixels.shape
    raw = pixels.tobytes(order='C')
    path = root / 'segments' / f'native-{index:03d}.mkv'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    entry = dict(window=index, path=str(path.relative_to(root)),
        shape=list(pixels.shape), crop=dict(left=2, right=2, top=0, bottom=0),
        rgb_sha256=hashlib.sha256(raw).hexdigest(), rgb_bytes=len(raw),
        status='encoding')
    entries.append(entry)
    _atomic_manifest(root, entries)
    remaining = _LIMIT - _check(root, deadline) - 65536
    if remaining <= 0:
        raise ValueError('insufficient bounded artifact space')
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}', '-r', '24',
            '-i', 'pipe:0', '-an', '-c:v', 'ffv1', '-pix_fmt', 'bgr0',
            '-threads', '1', '-fs', str(remaining), str(path)], input=raw,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            timeout=max(.001, deadline-time.monotonic()))
        _check(root, deadline)
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1',
            '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            timeout=max(.001, deadline-time.monotonic())).stdout
        entry['decoded_rgb_sha256'] = hashlib.sha256(decoded).hexdigest()
        entry['decoded_rgb_bytes'] = len(decoded)
        if len(decoded) != len(raw) or entry['decoded_rgb_sha256'] != entry['rgb_sha256']:
            raise ValueError('native FFV1 decoded RGB hash differs (possibly output cap)')
        entry['status'] = 'verified'
    except BaseException:
        entry['status'] = 'failed'
        raise
    finally:
        _atomic_manifest(root, entries)
    _check(root, deadline)


def _capture_export(root, deadline, records):
    # GPU environment dependencies are deliberately lazy, not CPU requirements.
    import numpy as np
    from PIL import Image
    from .live_native_backend import _export

    entries = []

    def exporter(result, path, start, stop, index):
        _check(root, deadline)
        count = stop - start
        if count not in (119, 102, 107) or index >= 14:
            raise ValueError('unexpected native packet geometry')
        if len(result.frames) != count:
            raise ValueError('unexpected native frame count')
        _retain_native(root, deadline, result.frames, index, entries)
        picks = (0, count // 2, count - 1)
        native = []
        for pick in picks:
            frame = np.asarray(result.frames[pick])
            if frame.dtype != np.uint8 or frame.shape != (448, 256, 3):
                raise ValueError('unexpected native sample')
            native.append(frame[:, 2:-2, :].copy())
        # Never replace or synthesize the native result passed to production export.
        if path.exists():
            raise FileExistsError(path)
        video = _export(result, path, start, stop, index)
        _check(root, deadline)
        select = '+'.join(f'eq(n\\,{n})' for n in picks)
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path),
            '-vf', 'select=' + select, '-vsync', '0', '-frames:v', '3',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            timeout=max(.001, deadline-time.monotonic())).stdout
        expected = 3 * 448 * 252 * 3
        if len(decoded) != expected:
            raise ValueError('encoded sample count/geometry differs')
        encoded = np.frombuffer(decoded, dtype=np.uint8).reshape(3, 448, 252, 3)
        sheet = Image.new('RGB', (252 * 3, 448))
        for slot, (pick, a, b) in enumerate(zip(picks, native, encoded)):
            hashes = {}
            for label, pixels in (('native', a), ('encoded', b)):
                target = root / f'{index:02d}-{pick:03d}-{label}.png'
                with target.open('xb') as f:
                    Image.fromarray(pixels).save(f, format='PNG')
                hashes[label + '_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
            delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
            records.append(dict(window=index, local_frame=pick, global_frame=start+pick,
                mae=float(delta.mean()), max_error=int(delta.max()), **hashes))
            sheet.paste(Image.fromarray(a), (slot * 252, 0))
        with (root / f'{index:02d}-native-contact.png').open('xb') as f:
            sheet.save(f, format='PNG')
        _check(root, deadline)
        return video
    return exporter


def run_diagnostic(runtime, root, *, deadline):
    """Generate exactly 14 listening windows, ACK final tail, retire owner.

    deadline is absolute time.monotonic(), finite and at least 300s away.
    Generation/export is bounded by min(caller deadline, now + 300s).
    Artifacts remain on failure for diagnosis; the native owner is abandoned
    (or closed if unused). Cleanup failure is chained, never silently ignored.
    """
    from .live_native_backend import LiveNativeBackend
    from .live_conversation import LiveConversation
    from .paired_av import LiveAVBinding
    from .conversation_conditioning import role_profile_digest
    from .streaming_windows import (ResamplingContext, NATIVE_TORCH,
                                    NATIVE_TORCHAUDIO, REFERENCE_SHA256)

    if not math.isfinite(deadline) or deadline-time.monotonic() < 300:
        raise ValueError('finite deadline with at least 300 seconds required')
    deadline = min(deadline, time.monotonic() + 300)
    root = Path(root).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('diagnostic output must be outside repository')
    root.mkdir(mode=0o700, exist_ok=False)
    records = []
    backend = driver = None
    final = False
    try:
        backend = LiveNativeBackend(runtime, root/'segments',
            exporter=_capture_export(root, deadline, records))
        binding = LiveAVBinding('idle-drift-'+uuid.uuid4().hex,
            role_profile_digest(runtime.quality_profile, 'speech'), 1,
            max_segments=16, max_total_frames=1654)
        context = ResamplingContext('native-48k32k-prefix128', NATIVE_TORCH,
                                    NATIVE_TORCHAUDIO, REFERENCE_SHA256)
        driver = LiveConversation(root/'driver', binding=binding, context=context,
                                  max_idle_windows=15, timeout=min(300, deadline-time.monotonic()))
        for index in range(14):
            _check(root, deadline)
            # Reserve worst-case production segment space before generating.
            if _check(root, deadline) + 16*1024*1024 + 4*1024*1024 > _LIMIT:
                raise ValueError('insufficient bounded artifact space')
            if index == 13:
                driver.request_stop()
            prepared = driver.prepare()
            if prepared is None:
                raise RuntimeError('unexpected listening wait')
            result = backend(prepared.manifest)
            packet = driver.complete_native(prepared, result)
            expected_end = 124 + index*102
            expected_count = 119 if index == 0 else (107 if index == 13 else 102)
            if (result['count'] != expected_count or result['committed_frame'] != expected_end
                    or result['emitted_frame'] != expected_end-(0 if index == 13 else 5)
                    or backend.receipts[-1]['role'] != 'listening'):
                raise ValueError('unexpected listening/final decoder clock')
            del packet  # no sender/pump and no growing PCM/video collection
        final = True
        backend.retire()
        _exclusive(root/'samples.json', json.dumps(records, sort_keys=True).encode())
        _exclusive(root/'REVIEW.txt', b'Generated listening only; silent video, no conversation audio.\n')
        # Separate segments are authoritative; merging would duplicate bounded media.
        return dict(windows=14, native_samples=len(records), encoded_samples=len(records),
            final=final, committed_frames=1450, emitted_frames=1450,
            final_packet_frames=107, mae_mean=sum(r['mae'] for r in records)/len(records),
            max_error=max(r['max_error'] for r in records),
            native_manifest=str(root/'native-manifest.json'),
            sample_manifest_sha256=hashlib.sha256((root/'samples.json').read_bytes()).hexdigest(),
            bytes=_check(root, deadline), root=str(root), merged=False, silence=True)
    except BaseException:
        if backend is not None and not backend._retired:
            if backend._id is None:
                backend.close_unused()
            else:
                backend.abandon(reason='local-sink-failure')
        raise
    finally:
        if driver is not None:
            driver.close()
