"""Offline stage-1 encode analysis, NOT an OBS or Discord measurement.

Run: python3 -m research.analyze_h3_obs_delivery_quality --source PATH --directory PATH
Source must explicitly name generated review media. Directory must be fresh,
private and outside this repository. No device, UDP, playback or cloud access.
Only the production encoder's local-file equivalent is measured; no stage 2.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from research.h3_personal.obs_av_sink import output_arguments

LIMIT = 64 * 1024 * 1024
TIMEOUT = 120
FRAMES = 2368
FPS = 24
DURATION = FRAMES / FPS
LISTEN_END = 26.4167
SPEECH_END = LISTEN_END + 12.04


def run(command, stdout_limit=4 * 1024 * 1024):
    """Bound wall time and captured output without in-memory unbounded pipes."""
    import resource

    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (LIMIT, LIMIT))

    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        subprocess.run(command, stdout=out, stderr=err, timeout=TIMEOUT,
                       check=True, preexec_fn=limits)
        if out.tell() > stdout_limit:
            raise ValueError('command output exceeds bound')
        out.seek(0)
        return out.read().decode('utf-8')


def probe(path):
    # Select explicit fields: never dump arbitrary metadata, tags or source URLs.
    raw = run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
               '-count_frames', '-count_packets', '-show_entries',
               'stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,sample_rate,channels,duration,nb_read_frames,nb_read_packets,bit_rate:format=duration,size,bit_rate',
               '-of', 'json', str(path)])
    result = json.loads(raw)
    packets = run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
                   '-show_entries', 'packet=stream_index,size', '-of', 'csv=p=0',
                   str(path)])
    totals = {}
    for line in packets.splitlines():
        fields = line.split(',')
        if len(fields) < 2 or not fields[0].isdigit() or not fields[1].isdigit():
            continue  # MPEG-TS side-data may add blank lines/extra fields.
        index, size = int(fields[0]), int(fields[1])
        entry = totals.setdefault(str(index), {'count': 0, 'bytes': 0})
        entry['count'] += 1
        entry['bytes'] += size
    duration = float(result['format']['duration'])
    for entry in totals.values():
        entry['payload_bitrate'] = entry['bytes'] * 8 / duration
    result['packet_sizes'] = totals
    return result


def validate_source(metadata):
    streams = metadata.get('streams', [])
    video = [s for s in streams if s.get('codec_type') == 'video']
    audio = [s for s in streams if s.get('codec_type') == 'audio']
    if len(video) != 1 or len(audio) != 1:
        raise ValueError('exactly one video and audio stream required')
    v = video[0]
    w, h = v.get('width', 0), v.get('height', 0)
    if (type(w) is not int or type(h) is not int or not
            (0 < w <= 4096 and 0 < h <= 4096 and w % 2 == h % 2 == 0)):
        raise ValueError('bounded even source geometry required')
    from fractions import Fraction
    if Fraction(v.get('avg_frame_rate', '0')) != FPS:
        raise ValueError('24 fps source required')
    if int(v.get('nb_read_frames', 0)) != FRAMES:
        raise ValueError('2368 source frames required')
    duration = float(metadata.get('format', {}).get('duration', 0))
    if not math.isfinite(duration) or abs(duration - DURATION) > 0.1:
        raise ValueError('98.667 second source required')
    return w, h


def encode_command(source, destination, duration=DURATION):
    args = output_arguments(48124)
    args[-1] = str(destination)
    return ['ffmpeg', '-nostdin', '-v', 'error', '-n',
            '-protocol_whitelist', 'file,pipe', '-i', str(source),
            '-map', '0:v:0', '-map', '0:a:0', '-t', str(duration),
            '-fs', str(LIMIT - 1024 * 1024)] + args


def ssim_command(source, stage, stats):
    # stats path is generated locally (not caller-controlled filter syntax).
    graph = ('[0:v:0]setpts=PTS-STARTPTS,format=yuv420p[a];'
             '[1:v:0]setpts=PTS-STARTPTS,format=yuv420p[b];'
             '[a][b]ssim=stats_file=' + str(stats) + ':shortest=1')
    return ['ffmpeg', '-nostdin', '-v', 'error', '-protocol_whitelist',
            'file,pipe', '-i', str(source), '-protocol_whitelist', 'file,pipe',
            '-i', str(stage), '-filter_complex', graph, '-an', '-f', 'null', '-']


def parse_ssim(text, expected=FRAMES):
    values = []
    for line in text.splitlines():
        n = re.search(r'(?:^|\s)n:(\d+)(?:\s|$)', line)
        all_value = re.search(r'(?:^|\s)All:([0-9.eE+\-]+)(?:\s|$)', line)
        if not n or not all_value or int(n[1]) != len(values) + 1:
            raise ValueError('missing, duplicate or unordered SSIM frame')
        value = float(all_value[1])
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('invalid SSIM All')
        values.append(value)
    if len(values) != expected:
        raise ValueError('SSIM frame count mismatch')
    return values


def summarize(values):
    def section(start, end):
        selected = [(i, v) for i, v in enumerate(values) if start <= i / FPS < end]
        return dict(start_seconds=start, end_seconds=end, count=len(selected),
                    mean=sum(v for _, v in selected) / len(selected),
                    min=min(v for _, v in selected))
    bins = [section(t, min(t + 5, len(values) / FPS))
            for t in range(0, math.ceil(len(values) / FPS), 5)]
    # Negative slope and late-minus-early indicate decay, not a causal diagnosis.
    mean = sum(values) / len(values)
    mid = (len(values) - 1) / 2
    slope = sum((i - mid) * (v - mean) for i, v in enumerate(values)) / sum(
        (i - mid) ** 2 for i in range(len(values))) * FPS
    return dict(mean=mean, min=min(values), frame_index_range=[0, len(values)-1],
                bins_5s=bins, slope_per_second=slope,
                last_minus_first_bin_mean=bins[-1]['mean']-bins[0]['mean'],
                regions={'listen_first': section(0, LISTEN_END),
                         'speech': section(LISTEN_END, SPEECH_END),
                         'return_listen': section(SPEECH_END, len(values)/FPS)})


def check_bound(directory):
    size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    if size > LIMIT:
        raise ValueError('media/report aggregate exceeds 64 MiB')
    return size


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def analyze(source, directory):
    source = Path(source).resolve(strict=True)
    directory = Path(directory).resolve()
    repo = Path(__file__).resolve().parents[1]
    if not source.is_file() or directory == repo or repo in directory.parents:
        raise ValueError('local source file and output outside repository required')
    # Do not overwrite/reuse any existing directory, even an empty one.
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(directory, 0o700)
    original = probe(source)
    validate_source(original)
    stage = directory / 'stage1.ts'
    run(encode_command(source, stage))
    check_bound(directory)
    encoded = probe(stage)
    validate_source(encoded)
    # Fixed safe relative stats filename avoids filter escaping for private paths.
    with tempfile.TemporaryDirectory(prefix='h3-ssim-') as tmp:
        stats = Path(tmp) / 'ssim.txt'
        run(ssim_command(source, stage, stats))
        if stats.stat().st_size > 1024 * 1024:
            raise ValueError('SSIM statistics exceed bound')
        values = parse_ssim(stats.read_text())
    report = dict(scope='Offline production stage-1 encoding only; NOT actual OBS, browser, camera or Discord output.',
                  source_sha256=digest(source), stage1_sha256=digest(stage),
                  source=original, stage1=encoded, ssim=summarize(values))
    target = directory / 'report.json'
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if check_bound(directory) + len(payload.encode()) > LIMIT:
        raise ValueError('media/report aggregate exceeds 64 MiB')
    target.write_text(payload)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    print(analyze(args.source, args.directory))


if __name__ == '__main__':
    main()
