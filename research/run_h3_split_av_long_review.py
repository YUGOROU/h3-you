"""Offline preparation and explicitly approved playback of generated reviewed AV.

No capture, upload, fresh inference, or device access during preparation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import uuid
import wave

from research import run_h3_split_av_probe as probe
from research.h3_personal.paired_av import LiveAVBinding, AudioRegion, ReceivedAVLease, encode
from research.h3_personal.paired_av_wire import parse_envelope, envelope_record
from research.h3_personal.conversation_scenario import segment_timing

VIDEO = probe.VIDEO
AUDIO = probe.AUDIO
PCM_SHA = 'a05d3d61febf779b8cfe3f081a1da064e7b315adb99234f90ee94de87bf881bd'
SAMPLES = 4736000
ONSET = 634 * 2000
SPEECH_SAMPLES = 577920
RANGES = tuple(zip([0, 119] + [119 + i * 102 for i in range(1, 22)],
                   [119] + [119 + i * 102 for i in range(1, 22)] + [2368]))
SCHEMA = 'h3-split-av-long-review-v1'


def validate_pcm(pcm, speech):
    if len(pcm) != SAMPLES * 2 or probe.sha(pcm) != PCM_SHA:
        raise ValueError('full reviewed PCM mismatch')
    if len(speech) != SPEECH_SAMPLES * 2 or pcm[ONSET*2:(ONSET+SPEECH_SAMPLES)*2] != speech:
        raise ValueError('original speech mismatch')
    if any(pcm[:ONSET*2]) or any(pcm[(ONSET+SPEECH_SAMPLES)*2:]):
        raise ValueError('reviewed silence mismatch')


def regions(start, stop, pcm):
    a, b = start * 2000, stop * 2000
    points = sorted({a, b, *[x for x in (ONSET, ONSET + SPEECH_SAMPLES) if a < x < b]})
    return tuple(AudioRegion('speech', x, y, 'original-pcm', 0, x-ONSET,
                             probe.sha(pcm[x*2:y*2])) if ONSET <= x < ONSET+SPEECH_SAMPLES
                 else AudioRegion('silence', x, y) for x, y in zip(points, points[1:]))


def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) != (1, 2, 48000, 'NONE'):
            raise ValueError('mono 48k S16LE required')
        return w.readframes(SAMPLES + 1)


def prepare(directory):
    from research.run_h3_realtime_first import encode_rgb_segment
    if probe.file_sha(AUDIO) != probe.AUDIO_SHA:
        raise ValueError('original WAV mismatch')
    speech = read_wav(AUDIO)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_streams', '-of', 'json', str(VIDEO)], timeout=120))
    streams = info['streams']
    if len(streams) != 1 or any(streams[0][k] != v for k, v in dict(width=252, height=448, r_frame_rate='24/1', avg_frame_rate='24/1').items()) or int(streams[0]['nb_read_frames']) != 2368:
        raise ValueError('fixed reviewed video required')
    pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(VIDEO), '-map', '0:a:0', '-f', 's16le', '-acodec', 'pcm_s16le', 'pipe:1'], timeout=120)
    validate_pcm(pcm, speech)
    assets = dict(video_sha256=probe.file_sha(VIDEO), original_wav_sha256=probe.AUDIO_SHA,
                  full_pcm_sha256=PCM_SHA, samples=SAMPLES, speech_samples=SPEECH_SAMPLES,
                  onset=ONSET, geometry=[252, 448], fps=24, frames=2368,
                  profile_kind='pinned-reviewed-assets-not-model-profile')
    root = probe.private_directory(directory, fresh=True)
    root.chmod(0o700)
    binding = LiveAVBinding('split-av-long-review-' + uuid.uuid4().hex, probe.sha(probe.canonical(assets)), 0,
                            max_segments=23, max_total_frames=2368)
    with wave.open(str(root / 'review.wav'), 'wb') as w:
        w.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
        w.writeframes(pcm)
    envelopes = []
    with tempfile.TemporaryFile() as raw:
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(VIDEO), '-map', '0:v:0', '-an', '-frames:v', '2369', '-vsync', '0', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'], stdout=raw, stderr=subprocess.DEVNULL, check=True, timeout=120)
        if raw.tell() != 2368 * probe.STRIDE or raw.tell() > 900*1024*1024:
            raise ValueError('decoded frame size mismatch')
        raw.seek(0)
        for i, (start, stop) in enumerate(RANGES):
            path = root / f'segment-{i}.mp4'
            encode_rgb_segment(raw.read((stop-start)*probe.STRIDE), path, segment_timing(i, start, stop), width=252, height=448)
            e = encode(binding, i, start, stop, path.read_bytes(), pcm[start*4000:stop*4000], regions(start, stop, pcm), i == 22)
            envelopes.append(dict(schema='h3-paired-av-wire-v1', envelope=envelope_record(e)))
            path.with_suffix('.stderr').unlink(missing_ok=True)
    if sum((root / f'segment-{i}.mp4').stat().st_size for i in range(23)) > 32*1024*1024:
        raise ValueError('review fragments exceed media bound')
    if probe.file_sha(VIDEO) != assets['video_sha256'] or probe.file_sha(AUDIO) != probe.AUDIO_SHA:
        raise ValueError('source changed')
    manifest = dict(schema=SCHEMA, assets=assets, envelopes=envelopes,
                    wav_sha256=probe.file_sha(root/'review.wav'), generated_reviewed_av=True,
                    remote_verified=False, live_conversation=False)
    body = probe.canonical(manifest)
    (root/'manifest.json').write_bytes(body)
    (root/'manifest.sha256').write_text(probe.sha(body)+'\n')
    for path in root.iterdir():
        path.chmod(0o400)
    return manifest


def inspect(directory):
    root = probe.private_directory(directory)
    if root.stat().st_mode & 0o077:
        raise ValueError('private directory permissions required')
    def safe(name, limit):
        p = root / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size > limit or p.stat().st_mode & 0o377:
            raise ValueError('unsafe prepared file')
        return p
    body = safe('manifest.json', 262144).read_bytes()
    seal = safe('manifest.sha256', 65).read_text().strip()
    if probe.sha(body) != seal:
        raise ValueError('manifest seal mismatch')
    m = json.loads(body)
    if set(m) != {'schema','assets','envelopes','wav_sha256','generated_reviewed_av','remote_verified','live_conversation'} or m['schema'] != SCHEMA or m['generated_reviewed_av'] is not True or m['remote_verified'] is not False or m['live_conversation'] is not False or len(m['envelopes']) != 23:
        raise ValueError('fixed long review manifest required')
    assets = m['assets']
    expected = dict(video_sha256=assets['video_sha256'], original_wav_sha256=probe.AUDIO_SHA,
                    full_pcm_sha256=PCM_SHA, samples=SAMPLES, speech_samples=SPEECH_SAMPLES,
                    onset=ONSET, geometry=[252,448], fps=24, frames=2368,
                    profile_kind='pinned-reviewed-assets-not-model-profile')
    if assets != expected:
        raise ValueError('asset contract mismatch')
    wav = safe('review.wav', SAMPLES*2+1024)
    if probe.file_sha(wav) != m['wav_sha256']:
        raise ValueError('WAV seal mismatch')
    pcm = read_wav(wav)
    # Re-check the pinned source, not merely a self-consistent prepared seal.
    if probe.file_sha(AUDIO) != probe.AUDIO_SHA:
        raise ValueError('original WAV mismatch')
    validate_pcm(pcm, read_wav(AUDIO))
    binding, pairs = None, []
    for i, (start, stop) in enumerate(RANGES):
        e = parse_envelope(m['envelopes'][i])
        b = e.binding
        prefix = 'split-av-long-review-'
        try:
            valid_session = b.session == prefix + uuid.UUID(b.session.removeprefix(prefix)).hex
        except ValueError:
            valid_session = False
        if not valid_session or b != LiveAVBinding(b.session, probe.sha(probe.canonical(assets)), 0, max_segments=23, max_total_frames=2368):
            raise ValueError('binding mismatch')
        binding = binding or b
        path = safe(f'segment-{i}.mp4', 16*1024*1024)
        video = path.read_bytes()
        part = pcm[start*4000:stop*4000]
        if e != encode(binding, i, start, stop, video, part, regions(start, stop, pcm), i == 22):
            raise ValueError('prepared envelope mismatch')
        pairs.append((path, ReceivedAVLease(e, video, part)))
    if sum(path.stat().st_size for path, _ in pairs) > 32*1024*1024:
        raise ValueError('review fragments exceed media bound')
    return root, seal, binding, pairs


def run(directory, *, audio_binary, playback_approved=False, port=48124, audio_delay_seconds=.2):
    if playback_approved is not True:
        raise ValueError('separate explicit playback approval required')
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('invalid port')
    from research.run_h3_relay_conversation import direct_audio_pin, _receiver_diagnostic
    pin = direct_audio_pin(audio_binary, audio_delay_seconds)
    root, seal, binding, pairs = inspect(directory)
    receipt = dict(manifest_sha256=seal, audio_binary_sha256=pin['binary_sha256'],
                   generated_reviewed_av=True, remote_verified=False, live_conversation=False, status='starting')
    with (root/'run-seal.json').open('x') as f:
        json.dump(receipt, f)
    sink = None
    try:
        from research.h3_personal.split_av_sink import SplitAVSink
        if direct_audio_pin(audio_binary, audio_delay_seconds) != pin:
            raise ValueError('audio pin changed')
        sink = SplitAVSink(root, binding, playback_approved=True, port=port,
                           audio_binary=Path(pin['audio_binary']), audio_delay_seconds=audio_delay_seconds,
                           max_total_frames=binding.max_total_frames)
        for path, lease in pairs:
            sink.accept(path.name, lease)
        sink.finish()
        receipt['status'] = 'finished'
    except Exception as exc:
        receipt.update(status='failed', error='long-review-playback-failed', failure_type=type(exc).__name__, failure_trace=_receiver_diagnostic(exc))
    finally:
        if sink is not None:
            receipt.update(audio_evidence=dict(sink.audio_evidence), timing=dict(sink.timing))
            try:
                sink.close()
            except Exception as exc:
                receipt.update(status='failed', error='long-review-cleanup-failed', failure_trace=_receiver_diagnostic(exc))
        (root/'result.json').write_bytes(probe.canonical(receipt))
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', type=Path)
    modes.add_argument('--run', type=Path)
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
    return 0 if run(a.run, audio_binary=a.audio_binary, playback_approved=True, port=a.port, audio_delay_seconds=a.audio_delay_seconds)['status'] == 'finished' else 1


if __name__ == '__main__':
    raise SystemExit(main())
