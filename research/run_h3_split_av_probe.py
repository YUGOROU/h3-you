"""Offline preparation / explicitly approved playback of a reviewed generated clip.

Not ASR, new TTS, GPU inference, or a live conversation. Playback approval is
an operator assertion of separately obtained device/routing permission.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import uuid
import wave

from research.h3_personal.private_root import base as _private_base, root as _private_root
from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, ReceivedAVLease, encode
from research.h3_personal.paired_av_wire import parse_envelope, envelope_record
from research.h3_personal.conversation_scenario import segment_timing

VIDEO = _private_base() / 'visual-ancestry-comparison-20261026-v1/review/visual-ancestry/arm-1/review-fixed.mov'
AUDIO = _private_root() / 'av-headless-paced-20261024-v1/inputs/artifact-original-pcm.wav'
AUDIO_SHA = 'cf4013b8d2c6570fe81fc4f803b26acee9df7fbebc63b7c9a621b8b5204dc718'
RANGES = ((0, 119), (119, 221), (221, 240))
STRIDE = 252 * 448 * 3


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def private_directory(path, fresh=False):
    path = Path(path).expanduser().absolute()
    repo = Path(__file__).resolve().parents[1]
    resolved = path.resolve()
    if resolved == repo or repo in resolved.parents:
        raise ValueError('private output must be outside repository')
    if fresh:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('real private directory required')
    return path.resolve()


def read_pcm(path, expected_sha=None):
    if expected_sha is not None and file_sha(path) != expected_sha:
        raise ValueError('WAV seal mismatch')
    with wave.open(str(path), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (48000, 1, 2, 'NONE'):
            raise ValueError('original mono S16LE required')
        count = w.getnframes()
        pcm = w.readframes(480000)
    if len(pcm) != 960000:
        raise ValueError('ten seconds required')
    return pcm, count


def validate_video(info):
    streams = info['streams']
    if len(streams) != 1:
        raise ValueError('one selected video required')
    s = streams[0]
    if (s['width'], s['height'], s['r_frame_rate'], s['avg_frame_rate']) != (252, 448, '24/1', '24/1') or int(s['nb_read_frames']) < 874:
        raise ValueError('fixed reviewed geometry/rate/length required')


def prepare(directory):
    # No sink import, constructor, device, or network in this branch.
    from research.run_h3_realtime_first import encode_rgb_segment
    root = private_directory(directory, fresh=True)
    pcm, count = read_pcm(AUDIO, AUDIO_SHA)
    source_sha = file_sha(VIDEO)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_streams', '-of', 'json', str(VIDEO)], timeout=120))
    validate_video(info)
    assets = dict(video_sha256=source_sha, original_wav_sha256=AUDIO_SHA,
                  original_samples=count, source_frame_start=634, source_frame_stop=874,
                  source_cut_seconds=634 / 24, geometry=[252, 448], fps=24,
                  profile_kind='pinned-reviewed-assets-not-model-profile')
    binding = LiveAVBinding('split-av-probe-' + uuid.uuid4().hex, sha(canonical(assets)), 0,
                            max_segments=3, max_total_frames=240)
    with wave.open(str(root / 'speech.wav'), 'wb') as w:
        w.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
        w.writeframes(pcm)
    envelopes = []
    with tempfile.TemporaryFile() as raw:
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(VIDEO), '-map', '0:v:0', '-an', '-vf', 'select=between(n\\,634\\,873),setpts=N/24/TB', '-vsync', '0', '-frames:v', '240', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'], stdout=raw, stderr=subprocess.DEVNULL, check=True, timeout=120)
        if raw.tell() != 240 * STRIDE:
            raise ValueError('exact 240 decoded frames required')
        raw.seek(0)
        for i, (start, stop) in enumerate(RANGES):
            path = root / f'segment-{i}.mp4'
            encode_rgb_segment(raw.read((stop-start)*STRIDE), path, segment_timing(i, start, stop), width=252, height=448)
            part = pcm[start*4000:stop*4000]
            region = AudioRegion('speech', start*2000, stop*2000, 'original-pcm', 0, start*2000, sha(part))
            e = encode(binding, i, start, stop, path.read_bytes(), part, (region,), i == 2)
            envelopes.append(dict(schema='h3-paired-av-wire-v1', envelope=envelope_record(e)))
            path.with_suffix('.stderr').unlink(missing_ok=True)
    if file_sha(VIDEO) != source_sha or file_sha(AUDIO) != AUDIO_SHA:
        raise ValueError('source changed during preparation')
    manifest = dict(schema='h3-split-av-probe-v1', assets=assets, envelopes=envelopes,
                    speech_wav_sha256=file_sha(root / 'speech.wav'), speech_pcm_sha256=sha(pcm),
                    generated_reviewed_av=True, remote_verified=False, live_conversation=False)
    body = canonical(manifest)
    (root / 'manifest.json').write_bytes(body)
    (root / 'manifest.sha256').write_text(sha(body) + '\n')
    return manifest


def inspect(directory):
    root = private_directory(directory)
    def safe(name, limit):
        p = root / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size > limit:
            raise ValueError('missing/unsafe prepared file')
        return p
    body = safe('manifest.json', 65536).read_bytes()
    seal = safe('manifest.sha256', 65).read_text().strip()
    if sha(body) != seal:
        raise ValueError('manifest seal mismatch')
    m = json.loads(body)
    if m['schema'] != 'h3-split-av-probe-v1' or len(m['envelopes']) != 3:
        raise ValueError('fixed probe manifest required')
    assets = m['assets']
    if assets['original_wav_sha256'] != AUDIO_SHA or (assets['source_frame_start'], assets['source_frame_stop'], assets['geometry'], assets['fps']) != (634, 874, [252, 448], 24):
        raise ValueError('asset contract mismatch')
    pcm, count = read_pcm(safe('speech.wav', 961024), m['speech_wav_sha256'])
    if count != 480000 or sha(pcm) != m['speech_pcm_sha256']:
        raise ValueError('prepared PCM mismatch')
    pairs = []
    binding = None
    for i, (start, stop) in enumerate(RANGES):
        e = parse_envelope(m['envelopes'][i])
        b = e.binding
        if not b.session.startswith('split-av-probe-') or b.max_segments != 3 or b.max_total_frames != 240 or b.profile_sha256 != sha(canonical(assets)):
            raise ValueError('binding mismatch')
        binding = binding or b
        video = safe(f'segment-{i}.mp4', 16*1024*1024).read_bytes()
        part = pcm[start*4000:stop*4000]
        region = AudioRegion('speech', start*2000, stop*2000, 'original-pcm', 0, start*2000, sha(part))
        expected = encode(binding, i, start, stop, video, part, (region,), i == 2)
        if e != expected:
            raise ValueError('prepared envelope mismatch')
        pairs.append((root / f'segment-{i}.mp4', ReceivedAVLease(e, video, part)))
    return root, seal, binding, pairs


def run(directory, *, audio_binary, playback_approved=False, port=48124, audio_delay_seconds=.2):
    if playback_approved is not True:
        raise ValueError('separate explicit playback approval required')
    if type(port) is not int or not 1024 <= port <= 65535 or not math.isfinite(audio_delay_seconds) or not 0 <= audio_delay_seconds <= 1:
        raise ValueError('invalid playback configuration')
    root, seal, binding, pairs = inspect(directory)
    binary = Path(audio_binary).expanduser().resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError('executable binary required')
    receipt = dict(manifest_sha256=seal, audio_binary_sha256=file_sha(binary), port=port,
                   audio_delay_seconds=audio_delay_seconds, playback_approved=True,
                   remote_verified=False, live_conversation=False, status='starting')
    # Exclusive claim prevents replay/overwrite, including failed attempts.
    with (root / 'run-seal.json').open('x') as f:
        json.dump(receipt, f, allow_nan=False)
    sink = None
    try:
        from research.h3_personal.split_av_sink import SplitAVSink
        sink = SplitAVSink(root, binding, playback_approved=True, port=port,
                           audio_binary=binary, audio_delay_seconds=audio_delay_seconds,
                           max_total_frames=240)
        for path, lease in pairs:
            sink.accept(path.name, lease)
        sink.finish()
        receipt.update(status='finished', audio_evidence=dict(sink.audio_evidence), timing=dict(sink.timing))
    except Exception as exc:
        from research.run_h3_relay_conversation import _receiver_diagnostic
        receipt.update(status='failed', failure_type=type(exc).__name__,
                       failure_trace=_receiver_diagnostic(exc))
        if sink is not None:
            receipt.update(audio_evidence=dict(sink.audio_evidence),timing=dict(sink.timing))
    finally:
        if sink is not None:
            sink.close()
        (root / 'result.json').write_bytes(canonical(receipt))
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--run', type=Path)
    p.add_argument('--audio-binary', type=Path)
    p.add_argument('--playback-approved', action='store_true')
    p.add_argument('--port', type=int, default=48124)
    p.add_argument('--audio-delay-seconds', type=float, default=.2)
    a = p.parse_args(argv)
    if a.prepare:
        if a.audio_binary or a.playback_approved:
            p.error('playback arguments are run-only')
        prepare(a.prepare)
        return 0
    if not a.audio_binary or not a.playback_approved:
        p.error('--run requires --audio-binary and --playback-approved')
    return 0 if run(a.run, audio_binary=a.audio_binary, playback_approved=True,
                    port=a.port, audio_delay_seconds=a.audio_delay_seconds)['status'] == 'finished' else 1


if __name__ == '__main__':
    raise SystemExit(main())
