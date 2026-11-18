"""Local-only, resumable iPhone → H3 raw-manifest preparation (stdlib + FFmpeg)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave

FPS = 24
FRAMES = 124
DURATION = FRAMES / FPS
AUDIO_SAMPLES = round(DURATION * 40) * 800


def dump(path: Path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def probe(path: Path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)
    ]))


def timing(path: Path):
    """Scan packet durations, not a guess based only on average frame rate."""
    process = subprocess.Popen([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=duration_time', '-of', 'csv=p=0', str(path)
    ], stdout=subprocess.PIPE, text=True)
    counts = {}
    for line in process.stdout:
        try:
            value = round(float(line.split(',')[0]), 6)
        except ValueError:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    process.stdout.close()
    if process.wait() != 0:
        raise RuntimeError('packet timing scan failed')
    return {'packet_duration_histogram': counts,
            'variable_packet_durations': len(counts) > 1}


def starts(duration: float):
    """Keep every second, using one overlapping full-length tail, never drop silence."""
    if duration < DURATION:
        return []
    result = [i * DURATION for i in range(math.floor(duration / DURATION))]
    tail = duration - DURATION
    if tail - result[-1] > 1e-6:
        result.append(tail)
    return result


def inventory(source: Path):
    records = []
    for path in sorted(source.glob('*.mov')):
        meta_path = path.with_name(path.name.removeprefix('rejected_')).with_suffix('.json')
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        entry = {'path': str(path.resolve()), 'metadata': meta}
        reasons = []
        try:
            entry['probe'] = info = probe(path)
            video = next(s for s in info['streams'] if s['codec_type'] == 'video')
            audio = [s for s in info['streams'] if s['codec_type'] == 'audio']
            entry['duration_seconds'] = float(video.get('duration', info['format']['duration']))
            if not audio:
                reasons.append('no_audio_track_test_capture')
            entry['timing'] = timing(path)
        except (subprocess.CalledProcessError, StopIteration, KeyError, ValueError) as error:
            reasons.append(f'probe_failed:{error}')
        if path.name.startswith('rejected_'):
            reasons.append('explicitly_rejected_source')
        if meta.get('status') != 'completed':
            reasons.append('missing_or_incomplete_capture_metadata')
        if meta.get('stopReason') or meta.get('error'):
            reasons.append('capture_interruption_or_error')
        entry['exclusion_reasons'] = reasons
        records.append(entry)
    return records


def command(source: str, start: float, video: Path, audio: Path, width: int, height: int):
    # One input/seek origin for both outputs; no independent timestamp reset.
    # FFmpeg autorotates the display matrix before scale. Pad, never crop/stabilize.
    vf = (f'fps={FPS}:start_time=0,scale={width}:{height}:'
          'force_original_aspect_ratio=decrease:force_divisible_by=2,'
          f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1')
    return ['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-y',
            '-ss', f'{start:.9f}', '-t', f'{DURATION:.9f}', '-i', source,
            '-map', '0:v:0', '-an', '-vf', vf, '-frames:v', str(FRAMES),
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '25', '-threads', '2',
            '-pix_fmt', 'yuv420p', '-map_metadata', '-1', '-metadata:s:v:0', 'rotate=0',
            '-movflags', '+faststart', str(video),
            '-map', '0:a:0', '-vn', '-af',
            f'aresample=32000:async=1:first_pts=0,apad,atrim=end_sample={AUDIO_SAMPLES}',
            '-ar', '32000', '-ac', '2', '-c:a', 'pcm_s16le', str(audio)]


def validate(video: Path, audio: Path, width: int, height: int):
    info = probe(video)
    s = next(s for s in info['streams'] if s['codec_type'] == 'video')
    if (s['width'], s['height'], int(s['nb_frames']), s['avg_frame_rate']) != (width, height, FRAMES, '24/1'):
        raise ValueError('output video geometry/clock mismatch')
    if any(d.get('rotation', 0) != 0 for d in s.get('side_data_list', [])):
        raise ValueError('unapplied rotation')
    with wave.open(str(audio)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (32000, 2, 2, AUDIO_SAMPLES):
            raise ValueError('output audio grid mismatch')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(video),
                    '-f', 'null', '-'], check=True, capture_output=True)


def prepare(source: Path, output: Path, limit: int | None, width: int, height: int):
    if width % 32 or height % 32 or min(width, height) <= 0:
        raise ValueError('canvas dimensions must be positive multiples of 32')
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError('outputs must be separate from originals')
    output.mkdir(parents=True, exist_ok=True)
    records = inventory(source)
    dump(output / 'inventory.json', records)
    plan = []
    for rec in records:
        if rec['exclusion_reasons']:
            continue
        for index, start in enumerate(starts(rec['duration_seconds'])):
            ident = f"{Path(rec['path']).stem}_{index:04d}"
            plan.append({'id': ident, 'source': rec['path'], 'start_seconds': start,
                         'end_seconds': start + DURATION,
                         'video': str(output / 'clips' / (ident + '.mp4')),
                         'audio': str(output / 'audio' / (ident + '.wav')),
                         # No fabricated speaking/listening labels or metadata location claims.
                         'caption': 'A selfie video of the same person, with natural facial and body movements and changing surroundings.'})
    settings = {'width': width, 'height': height, 'fps': FPS, 'frames': FRAMES,
                'audio_samples': AUDIO_SAMPLES, 'crf': 25, 'plan': plan,
                'sources': [{'path': r['path'], 'size': Path(r['path']).stat().st_size,
                             'mtime_ns': Path(r['path']).stat().st_mtime_ns} for r in records]}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    config = output / 'config.json'
    if config.exists() and json.loads(config.read_text())['signature'] != signature:
        raise ValueError('source/settings changed; use a new output directory')
    dump(config, {'signature': signature, **settings})
    dump(output / 'plan.json', plan)
    for name in ('clips', 'audio'):
        (output / name).mkdir(exist_ok=True)
    manifest, failures = [], []
    selected = plan if limit is None else plan[:limit]
    for i, item in enumerate(selected):
        video, audio = Path(item['video']), Path(item['audio'])
        temporary_video = video.with_name(video.stem + '.partial.mp4')
        temporary_audio = audio.with_name(audio.stem + '.partial.wav')
        try:
            if not (video.exists() and audio.exists()):
                if shutil.disk_usage(output).free < 1024 ** 3:
                    raise OSError('less than 1 GiB free; stopped without deleting originals')
                subprocess.run(command(item['source'], item['start_seconds'], temporary_video,
                                       temporary_audio, width, height), check=True, capture_output=True)
                validate(temporary_video, temporary_audio, width, height)
                temporary_video.replace(video)
                temporary_audio.replace(audio)
            else:
                validate(video, audio, width, height)
            manifest.append({key: item[key] for key in ('id', 'video', 'audio', 'caption')})
        except OSError:
            raise  # Capacity/tool availability is a blocker, not corrupt training data.
        except (subprocess.CalledProcessError, ValueError) as error:
            failures.append({'id': item['id'], 'reason': str(error),
                             'stderr': getattr(error, 'stderr', b'').decode(errors='replace')[-2000:]})
        finally:
            temporary_video.unlink(missing_ok=True)
            temporary_audio.unlink(missing_ok=True)
        dump(output / 'manifest.json', manifest)
        dump(output / 'failures.json', failures)
        print(f'{i + 1}/{len(selected)} {item["id"]} ok={len(manifest)} failed={len(failures)}', flush=True)
    adopted = sum(r['duration_seconds'] for r in records if not r['exclusion_reasons'])
    summary = {'candidate_source_seconds': adopted, 'planned_clips': len(plan),
               'processed_clips': len(manifest), 'failed_clips': len(failures),
               'excluded_sources': [{'path': r['path'], 'reasons': r['exclusion_reasons']}
                                    for r in records if r['exclusion_reasons']],
               'complete_raw_preprocessing': len(manifest) == len(plan) and not failures and bool(plan),
               'trainer_latent_cache_loader_verified': False,
               'tail_policy': 'overlapping final full clip; no speech filtering',
               'settings_signature': signature}
    dump(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path.home() / 'Downloads/Recordings')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=1344)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    print(json.dumps(prepare(args.source, args.output, args.limit, args.width, args.height), indent=2))


if __name__ == '__main__':
    main()
