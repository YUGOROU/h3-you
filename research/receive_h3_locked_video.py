"""Receive a completed locked-speech deployment; no playback or ledger changes.

Usage: python3 -m research.receive_h3_locked_video --directory DEPLOYMENT_PARENT
Requires huggingface_hub==1.7.2 and ffmpeg/ffprobe. Existing output directories
are refused (including partial attempts); inspect/remove them explicitly to retry.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import time
import wave

from research.bootstrap_h3_fast_ref2va import locked_speech_plan
from research.h3_personal.external_audio import SCHEMA

TERMINAL = {'COMPLETED', 'ERROR', 'CANCELED', 'DELETED'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_profile(profile, spec):
    plan = locked_speech_plan(spec)
    require(plan is not None, 'locked speech spec required')
    count = plan['windows_required']
    require(1 <= count <= 4, 'finite segment count required')
    flags = dict(external_audio_conditioning=True, originals_modified=False,
                 visual_only=True, generated_audio_discarded=True,
                 independent_ref2va_clips=True, shared_head_trim=False)
    for key, value in flags.items():
        require(profile.get(key) is value, 'profile flag differs: ' + key)
    require(profile.get('expected_forwards_per_segment') == 4 and
            profile.get('unique_frames') == 124 * count, 'four-step frame count differs')
    actual_plan = profile['locked_audio_plan']
    for key, value in plan.items():
        if key != 'wav':  # Worker uses an absolute mounted path.
            require(actual_plan.get(key) == value, 'speech plan differs: ' + key)
    receipts = profile['locked_audio_receipts']
    require(len(receipts) == count * 2, 'both-rank locked receipts required')
    seen = set()
    for row in receipts:
        pair = (row['window_index'], row['worker_rank'])
        require(pair not in seen and pair in {(i, r) for i in range(count) for r in (0, 1)},
                'duplicate/unknown locked receipt')
        seen.add(pair)
        window = plan['windows'][pair[0]]
        expected = dict(schema=SCHEMA, wav_sha256=spec['locked_speech']['sha256'],
                        audio_timestep_policy='fixed_clean_1.0', steps_held=4,
                        audio_scheduler_step_skipped=True, audio_velocity_discarded=True,
                        rows_unchanged=True, truncation=False, time_stretch=False,
                        supplied_speech_not_generated=True, generated_audio_decode_skipped=True,
                        latent_start=window['latent_start'], latent_count=window['latent_count'],
                        expected_rows=window['expected_rows'], encoded_rows=window['expected_rows'])
        for key, value in expected.items():
            require(type(row.get(key)) is type(value) and row[key] == value,
                    'locked receipt differs: ' + key)
    coverage = profile['loaded_coverage']
    require(len(coverage) == 2 and {r['worker_rank'] for r in coverage} == {0, 1},
            'both-rank original64 coverage required')
    for row in coverage:
        for key, value in dict(identity_modules=208, total_modules=312, unmatched=0,
                               rank='mixed128-144', partition='transformer_ref').items():
            require(row.get(key) == value, 'original64 coverage differs: ' + key)
    require(len(profile['segments']) == count, 'profile segment count differs')
    return count


def validate_metadata(metadata, frames, *, audio=False):
    streams = metadata['streams']
    videos = [s for s in streams if s['codec_type'] == 'video']
    audios = [s for s in streams if s['codec_type'] == 'audio']
    require(len(videos) == 1 and len(audios) == int(audio) and
            len(streams) == 1 + int(audio), 'unexpected media streams')
    video = videos[0]
    require((video['width'], video['height']) == (252, 448), 'portrait geometry differs')
    require(video['avg_frame_rate'] == '24/1' and video['r_frame_rate'] == '24/1',
            '24fps required')
    require(int(video['nb_read_frames']) == frames, 'decoded frame count differs')


def verify_remote_bytes(data, row):
    require(len(data) == row.size, 'remote size differs')
    if row.lfs:
        expected = row.lfs.sha256
        actual = hashlib.sha256(data).hexdigest()
    else:
        expected = row.blob_id
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    require(actual == expected, 'remote content hash differs')
    return hashlib.sha256(data).hexdigest()


def concat_manifest(segments):
    """Each verified clip is 124/24s, despite absolute-PTS container duration."""
    return ''.join(f"file '{p.name}'\nduration {124 / 24:.12f}\n" for p in segments)


def command(args):
    # Do not leak command stderr (paths or provider responses) to receipts.
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=180).stdout


def probe(path):
    return json.loads(command(['ffprobe', '-v', 'error', '-count_frames',
                               '-show_streams', '-of', 'json', str(path)]))


def decode(path):
    command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
             '-f', 'null', '-'])


def pcm(path):
    return command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
                    '-map', '0:a:0', '-c:a', 'pcm_s32le', '-f', 's32le', '-'])


def receive(directory, wait_seconds):
    require(type(wait_seconds) is int and 0 <= wait_seconds <= 1200, 'invalid wait')
    spec = json.loads((directory / 'deployment/spec.json').read_text())
    submission = json.loads((directory / 'submission.json').read_text())
    plan = locked_speech_plan(spec)
    require(plan is not None, 'locked deployment required')
    timeout = spec.get('config', {}).get('timeout_seconds')
    require(type(timeout) is int and 1 <= timeout <= 1500, 'finite deployment timeout required')
    require(re.fullmatch('[0-9a-f]{40}', submission.get('deployment_revision', '')),
            'immutable deployment submission required')
    require(importlib.metadata.version('huggingface_hub') == '1.7.2', 'SDK 1.7.2 required')
    from huggingface_hub import HfApi, hf_hub_download
    from research.provision_modal_identity_secret import cached_token
    api = HfApi(token=cached_token())  # Never persist raw Job responses.
    repo = submission.get('repo', spec['publication_repo'])
    require(repo == spec['publication_repo'], 'publication repo differs')
    owner = repo.split('/')[0]
    deadline = time.monotonic() + wait_seconds
    while True:
        job = api.inspect_job(job_id=submission['job_id'], namespace=owner)
        status = job.status.stage
        if status in TERMINAL or time.monotonic() >= deadline:
            break
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    if status != 'COMPLETED':
        return {'status': status}
    out = directory / 'verified-locked-video'
    require(not out.exists() and not out.is_symlink(), 'output exists; refusing overwrite')
    info = api.repo_info(repo, repo_type='dataset', files_metadata=True)
    require(info.private is True and re.fullmatch('[0-9a-f]{40}', info.sha),
            'private immutable output revision required')
    revision = info.sha
    rows = {s.rfilename: s for s in info.siblings}
    count = plan['windows_required']
    names = [f'outputs/segment-{i:03d}.mp4' for i in range(count)]
    require(sorted(n for n in rows if n.endswith('.mp4')) == names,
            'exact planned segment allowlist required')
    out.mkdir(mode=0o700)
    files = {}

    def download(repo_id, rev, remote, row, target):
        cached = Path(hf_hub_download(repo_id, remote, repo_type='dataset', revision=rev, token=cached_token()))
        data = cached.read_bytes()
        digest = verify_remote_bytes(data, row)
        with target.open('xb') as stream:
            stream.write(data)
        target.chmod(0o600)
        files[target.name] = digest
        return target

    report_path = download(repo, revision, 'outputs/profile.json', rows['outputs/profile.json'], out / 'profile.json')
    profile = json.loads(report_path.read_text())
    require(validate_profile(profile, spec) == count, 'planned count differs')
    segments = []
    for name in names:
        path = download(repo, revision, name, rows[name], out / Path(name).name)
        validate_metadata(probe(path), 124)
        decode(path)
        segments.append(path)
    speech = spec['locked_speech']
    speech_info = api.repo_info(speech['repo'], repo_type='dataset', revision=speech['revision'], files_metadata=True)
    require(speech_info.private is True and speech_info.sha == speech['revision'], 'private speech pin differs')
    speech_rows = {s.rfilename: s for s in speech_info.siblings}
    wav = download(speech['repo'], speech['revision'], speech['path'], speech_rows[speech['path']], out / 'speech.wav')
    require(files['speech.wav'] == speech['sha256'], 'supplied WAV SHA differs')
    with wave.open(str(wav), 'rb') as source:
        require(source.getcomptype() == 'NONE' and
                (source.getframerate(), source.getnframes()) == (speech['sample_rate'], speech['samples']),
                'supplied WAV metadata differs')
    require(count * 124 / 24 > speech['samples'] / speech['sample_rate'], 'video must cover entire speech')
    decode(wav)
    listing = out / 'concat.txt'
    # Segments carry absolute PTS and their container duration includes that
    # offset. Override concat durations with the verified 124/24 frame span.
    listing.write_text(concat_manifest(segments))
    joined = out / 'joined.mp4'
    command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-n', '-f', 'concat',
             '-safe', '1', '-i', str(listing), '-map', '0:v:0', '-c:v', 'copy', '-an', str(joined)])
    validate_metadata(probe(joined), 124 * count)
    decode(joined)
    review = out / 'review-with-speech.mp4'
    command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-n', '-i', str(joined),
             '-i', str(wav), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
             '-c:a', 'alac', str(review)])
    validate_metadata(probe(review), 124 * count, audio=True)
    decode(review)
    require(pcm(wav) == pcm(review), 'lossless speech PCM differs')
    for path in (joined, review):
        path.chmod(0o600)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dict(status='COMPLETED', job_id=submission['job_id'], output_revision=revision,
                   speech_revision=speech['revision'], files=files, full_decode_verified=True,
                   speech_pcm_exact=True, head_trim=False, autoplay=False,
                   identity_verified=False, lip_sync_verified=False)
    with (out / 'receive-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 1200:
        parser.error('--wait-seconds must be 0..1200')
    try:
        result = receive(args.directory.resolve(), args.wait_seconds)
    except Exception:
        # Provider exceptions can contain credentials; never print them.
        print(json.dumps({'status': 'REFUSED', 'reason': 'validation or retrieval failed'}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
