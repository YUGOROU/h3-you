"""One finite HF conversation scenario: listening -> full speech -> listening.

Same warm 1024 owner, one continuous AV sink. No device playback, no Discord, no
ASR/Pi/TTS, no retries, no silent 512 fallback. Publishes only a private review
video plus sanitized scalar metrics; the original reference and WAV stay private.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import wave

SCHEMA = 'h3-conversation-scenario-v1'
PREFIX = 'conversation/'
MAX_JSON = 1 << 20
MAX_VIDEO = 512 * 1024 * 1024


def validate_spec(spec):
    section = spec.get('conversation_scenario')
    if not isinstance(section, dict) or set(section) != {
            'schema', 'profile', 'listening_seconds', 'expected_frames', 'paced'}:
        raise ValueError('exact conversation scenario section required')
    if section['schema'] != SCHEMA or section['profile'] != 'reference1024':
        raise ValueError('reference1024 conversation profile required')
    seconds = section['listening_seconds']
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 3:
        raise ValueError('bounded explicit listening duration required')
    if section['expected_frames'] != 532 or type(section['expected_frames']) is not int:
        raise ValueError('fixed 532-frame conversation required')
    if section['paced'] is not True:
        raise ValueError('explicit wall-clock paced sink required')
    speech = spec.get('locked_speech')
    if not isinstance(speech, dict) or speech.get('samples') != 577920 or speech.get('sample_rate') != 48000:
        raise ValueError('existing full 12.04s speech binding required')
    if spec.get('attention_mode') != 'fa4-packed' or spec.get('layout') != 'portrait':
        raise ValueError('accepted portrait fa4-packed runtime required')
    if spec.get('runtime_policy') != 'resident-visual-cached':
        raise ValueError('resident visual cached runtime required')
    return section


def _pcm(path, samples, rate):
    with wave.open(str(path), 'rb') as handle:
        if (handle.getframerate(), handle.getnframes(), handle.getcomptype()) != (rate, samples, 'NONE'):
            raise ValueError('locked speech metadata differs')
        pcm = handle.readframes(samples)
    return pcm


def stage_inputs(root, paths, spec, deadline):
    """Create the two explicit encoded-silence WAVs next to the private output."""
    root, deadline = Path(root), deadline
    _remaining = lambda: _check(deadline)
    _remaining()
    speech = paths['speech'] / spec['locked_speech']['path']
    if hashlib.sha256(speech.read_bytes()).hexdigest() != spec['locked_speech']['sha256']:
        raise ValueError('locked speech SHA differs')
    _pcm(speech, spec['locked_speech']['samples'], spec['locked_speech']['sample_rate'])
    section = validate_spec(spec)
    directory = root.with_name(root.name + '-conversation-inputs')
    if directory.exists():
        raise ValueError('fresh private input directory required')
    directory.mkdir(parents=True, mode=0o700)
    samples = int(round(section['listening_seconds'] * 48000))
    if not 0 < samples <= 144000:
        raise ValueError('bounded silence sample count required')
    listening = []
    for name in ('listen-0.wav', 'listen-2.wav'):
        path = directory / name
        with wave.open(str(path), 'wb') as handle:
            handle.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            handle.writeframes(bytes(samples * 2))
        path.chmod(0o400)
        listening.append(str(path))
    _remaining()
    return tuple(listening), speech


def _check(deadline):
    seconds = deadline - time.monotonic()
    if not math.isfinite(seconds) or seconds <= 0:
        raise TimeoutError('conversation scenario deadline exhausted')
    return seconds


def run_session(root, paths, spec, deployment, deadline):
    import sys
    sys.path.insert(0, str(root))
    from huggingface_hub import CommitOperationAdd, HfApi
    section = validate_spec(spec)
    root = Path(root)
    paths = {k: Path(v) for k, v in paths.items()}
    _check(deadline)
    api = HfApi(token=os.environ['HF_TOKEN'])
    if api.repo_info(deployment, repo_type='dataset').private is not True:
        raise ValueError('private deployment dataset required')

    def publish(exports, message):
        _check(deadline)
        api.create_commit(deployment, repo_type='dataset', operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in exports],
            commit_message=message)

    listening, speech = stage_inputs(root, paths, spec, deadline)
    output = root.with_name(root.name + '-conversation-generated')
    if output.exists():
        raise ValueError('fresh private output directory required')
    from research.h3_personal.accepted_profiles import choose_profile
    from research.h3_personal.conversation_conditioning import prompt_sha
    from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
    from research.h3_personal.ref2va_runtime import sha
    identity = paths['personal'] / spec['identity_prefix']
    profile = choose_profile(ordered_source_sha256=(sha(paths['reference'] /
        'production-inputs/references/secondary.jpeg'),), prompt_sha256=prompt_sha('speech'))
    if profile.name != 'reference1024':
        raise ValueError('reference1024 required')
    runtime = LiveVoiceRuntime(model=paths['base'], model_manifest=root / 'model-manifest.json',
        identity=identity / 'adapter_model.safetensors',
        identity_config=identity / 'adapter_config.json',
        smd=paths['smd'] / 'lora/adapter_model.safetensors',
        reference=paths['reference'] / 'production-inputs/references/secondary.jpeg',
        archive=root / 'runtime/FastVideo.tar.gz', output=output, continuation=True,
        quality_profile=profile, conversation_roles=True)
    started = time.monotonic()
    try:
        _check(deadline)
        runtime.start()
        startup = time.monotonic() - started
        remaining = _check(deadline)
        # The sink and adapter enforce a finite <=300s local deadline.
        timeout = int(math.floor(min(remaining - 60, 300)))
        if timeout < 60:
            raise ValueError('insufficient integer conversation timeout')
        begin = time.monotonic()
        receipt = runtime.render_conversation_av(listening, speech,
            output / 'conversation', output / 'conversation.nut',
            timeout=timeout, paced=bool(section['paced']))
        wall = time.monotonic() - begin
        review = output / 'review.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-n',
            '-i', receipt['output'], '-map', '0:v:0', '-map', '0:a:0',
            '-vsync', '0', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-c:a', 'alac', '-movflags', '+faststart',
            str(review)], check=True, timeout=min(300, int(_check(deadline))))
        _check(deadline)
        original = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(review),
            '-map', '0:a:0', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'],
            timeout=min(300, int(_check(deadline))))
        expected = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', receipt['output'],
            '-map', '0:a:0', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'],
            timeout=min(300, int(_check(deadline))))
        if original != expected:
            raise ValueError('review audio differs from the sink PCM')
        report = dict(schema=SCHEMA, completed=True, profile=profile.name,
            profile_sha256=profile.digest, startup_seconds=startup, wall_seconds=wall,
            total_seconds=time.monotonic() - started,
            frames=receipt['frames'], committed_endpoints=receipt['committed_endpoints'],
            emitted_endpoints=receipt['emitted_endpoints'],
            source_onsets=receipt['source_onsets'], roles=receipt['roles'],
            segments=receipt['segments'], av=receipt['av'], sequence=receipt['sequence'],
            generation=receipt['generation'],
            output_sha256=receipt['output_sha256'], review_sha256=sha(review),
            review_pcm_exact=True, listening_seconds=section['listening_seconds'],
            speech_samples=577920, speech_sha256=spec['locked_speech']['sha256'],
            listening_quality_verified=False, lip_sync_verified=False,
            device_playback_verified=False, realtime_complete=False,
            raw_inputs_uploaded=False, same_warm_runtime=True, paced=section['paced'])
        if len(report['segments']) != 5 or report['frames'] != 532:
            raise ValueError('conversation segment coverage differs')
        review_bytes = review.read_bytes()
        if not 0 < len(review_bytes) <= MAX_VIDEO:
            raise ValueError('bounded review video required')
        publish([(PREFIX + 'report.json', (json.dumps(report, sort_keys=True) + '\n').encode()),
                 (PREFIX + 'review.mp4', review_bytes)],
                'Private conversation scenario: listening -> full speech -> listening (one warm 1024 owner)')
        return report
    finally:
        try:
            runtime.close()
        except BaseException:
            pass
