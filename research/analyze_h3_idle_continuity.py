"""Offline native14 continuity scalars; numbers are NOT seamless acceptance.

Usage: python -m research.analyze_h3_idle_continuity ROOT [--verify-rgb]
Only local retained files are read. No playback, devices, network or GPU.
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time

COUNTS = (119,) + (102,) * 12 + (107,)
MAX_OUTPUT = 4 * 1024 * 1024


def load_manifest(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('root must be a directory')
    manifest = root / 'native-manifest.json'
    if manifest.is_symlink() or manifest.stat().st_size > MAX_OUTPUT:
        raise ValueError('invalid manifest')
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or len(entries) != 14:
        raise ValueError('expected fourteen windows')
    for i, (entry, count) in enumerate(zip(entries, COUNTS)):
        expected = f'segments/native-{i:03d}.mkv'
        if (not isinstance(entry, dict) or entry.get('window') != i
                or entry.get('path') != expected
                or entry.get('shape') != [count, 448, 256, 3]
                or entry.get('status') != 'verified'
                or entry.get('rgb_bytes') != count * 448 * 256 * 3
                or not re.fullmatch('[0-9a-f]{64}', entry.get('rgb_sha256', ''))):
            raise ValueError(f'invalid window {i}')
        path = root / expected
        if not path.is_file() or path.resolve().parent != root / 'segments' or path.is_symlink():
            raise ValueError('native path escapes root or is not a regular file')
    return root, entries


def _run(args, deadline):
    # Disk spooling avoids unbounded communicate() buffers; both streams share cap.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1'] + args,
                                stdout=out, stderr=err)
        try:
            while True:
                if out.tell() + err.tell() > MAX_OUTPUT:
                    raise ValueError('ffmpeg output exceeds 4 MiB')
                if time.monotonic() >= deadline:
                    raise TimeoutError('ffmpeg deadline (240 seconds)')
                if proc.poll() is not None:
                    break
                time.sleep(.01)
            if out.tell() + err.tell() > MAX_OUTPUT:
                raise ValueError('ffmpeg output exceeds 4 MiB')
            if proc.returncode:
                raise ValueError('ffmpeg failed')
            out.seek(0)
            return out.read(MAX_OUTPUT).decode('utf-8', errors='strict')
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def parse_differences(text):
    values = []
    frame = None
    for line in text.splitlines():
        if line.startswith('frame:'):
            frame = int(line.split()[0].split(':')[1])
        elif line.startswith('lavfi.signalstats.YAVG='):
            value = float(line.split('=', 1)[1])
            if frame != len(values) or not math.isfinite(value) or not 0 <= value <= 255:
                raise ValueError('invalid difference metadata')
            values.append(value)
    if len(values) != sum(COUNTS) - 1:
        raise ValueError('expected 1449 inter-frame differences')
    return values


def summarize(values):
    if len(values) != sum(COUNTS) - 1 or any(
            not math.isfinite(v) or not 0 <= v <= 255 for v in values):
        raise ValueError('invalid differences')
    boundaries = []
    for i in range(1, 14):
        start = 119 + (i - 1) * 102
        # Difference array index f-1 compares global frames f-1 and f.
        early = values[start - 1:start + 23]
        rest = values[start + 23:start + COUNTS[i] - 1]
        boundaries.append(dict(window=i, global_frame=start, refreshed=i in (3, 6, 9, 12),
            boundary_luma_absolute=values[start - 1],
            first24_peak=max(early), first24_mean=sum(early) / len(early),
            remaining_peak=max(rest), remaining_mean=sum(rest) / len(rest)))
    return dict(frames=sum(COUNTS), differences=len(values),
                metric='mean absolute cropped luma difference (ffmpeg YAVG)',
                acceptance='Numeric metrics do not establish seamless continuity.',
                boundaries=boundaries)


def analyze(root, verify_rgb=False):
    root, entries = load_manifest(root)
    deadline = time.monotonic() + 240
    if verify_rgb:
        for entry in entries:
            output = _run(['-i', str(root / entry['path']), '-map', '0:v:0', '-an',
                           '-c:v', 'rawvideo', '-pix_fmt', 'rgb24', '-f', 'hash',
                           '-hash', 'sha256', '-'], deadline)
            if output.strip() != 'SHA256=' + entry['rgb_sha256']:
                raise ValueError('full decoded RGB hash mismatch')
    with tempfile.TemporaryDirectory(prefix='h3-continuity-') as tmp:
        listing = Path(tmp) / 'concat.txt'
        paths = [str(root / e['path']) for e in entries]
        if any('\n' in p or '\r' in p for p in paths):
            raise ValueError('newline in native path')
        listing.write_text(''.join("file '" + p.replace("'", "'\\''") + "'\n" for p in paths))
        output = _run(['-f', 'concat', '-safe', '0', '-protocol_whitelist', 'file',
            '-i', str(listing), '-map', '0:v:0', '-vf',
            'crop=252:448:2:0,tblend=all_mode=difference,signalstats,metadata=print:file=-',
            '-an', '-f', 'null', '-'], deadline)
    result = summarize(parse_differences(output))
    result['full_rgb_verified'] = bool(verify_rgb)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--verify-rgb', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(analyze(args.root, args.verify_rgb), indent=2))
    except (ValueError, OSError, TimeoutError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
