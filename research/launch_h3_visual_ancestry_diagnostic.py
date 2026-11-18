"""Offline-first visual ancestry comparison staging; never launches by default."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from types import FunctionType
import uuid
import wave

from research import launch_h3_idle_diagnostic as idle
from research.launch_h3_conversation_scenario import ROOT, PRIVATE, _bound, closure, rewrite_allowlist

SECTION = {'schema': 'h3-visual-ancestry-comparison-v1',
           'arms': ['epoch-idle-every3-v1', 'conversation-tail-every3-v1'],
           'idle_before': 6, 'idle_after': 14}
WORKER = 'research.hf_h3_visual_ancestry_diagnostic'


def speech_metadata(body):
    """Inspect supplied generated speech locally, without conversion or truncation."""
    try:
        with wave.open(io.BytesIO(body), 'rb') as wav:
            samples = wav.getnframes()
            if (wav.getcomptype() != 'NONE' or wav.getsampwidth() != 2
                    or wav.getnchannels() != 1 or wav.getframerate() != 48000
                    or not 0 < samples <= 30 * 48000):
                raise ValueError('generated speech must be PCM16 mono 48kHz <=30s')
            pcm = wav.readframes(samples)
            if len(pcm) != samples * 2:
                raise ValueError('truncated speech PCM')
    except (wave.Error, EOFError) as error:
        raise ValueError('invalid speech WAV') from error
    return dict(filename='diagnostic-speech.wav',
                wav_sha256=hashlib.sha256(body).hexdigest(),
                pcm_sha256=hashlib.sha256(pcm).hexdigest(), samples=samples,
                sample_rate=48000, channels=1)


def inspect(stage):
    from research.launch_h3_fast_ref2va import inspect as base_inspect
    from research.hf_h3_visual_ancestry_diagnostic import validate_spec
    stage = Path(stage)
    spec, inventory = base_inspect(stage)
    validate_spec(spec)
    if spec.get('visual_ancestry_comparison') != SECTION:
        raise ValueError('fixed visual ancestry section required')
    for key in ('idle_diagnostic', 'conversation_scenario', 'streaming_trial',
                'late_listening_diagnostic', 'live_voice_session', 'live_direct_session',
                'live_relay_session', 'modal_live_approval', 'incremental_transfer',
                'regional_compile', 'warm_stage_diagnostic'):
        if spec.get(key):
            raise ValueError('exclusive visual ancestry diagnostic required')
    if (not re.fullmatch(r'YUGOROU/h3-fast-ref2va-smd4-gpu-ancestry-[a-f0-9]{32}',
                         spec.get('publication_repo', ''))
            or type(spec['config']['timeout_seconds']) is not int
            or spec['config']['timeout_seconds'] != 1200
            or spec.get('locked_speech') is not None
            or spec.get('face_crop_box') is not None
            or spec.get('attention_mode') != 'fa4-packed'
            or spec.get('layout') != 'portrait'
            or spec.get('runtime_policy') != 'resident-visual-cached'):
        raise ValueError('fixed finite ancestry runtime required')
    names = {row['path'] for row in inventory['files']}
    if not {WORKER.replace('.', '/') + '.py', 'diagnostic-speech.wav'} <= names:
        raise ValueError('ancestry worker and speech required')
    if speech_metadata((stage/'diagnostic-speech.wav').read_bytes()) != spec.get('diagnostic_speech'):
        raise ValueError('immutable diagnostic speech binding differs')
    for name in names:
        if name.endswith('.py'):
            compile((stage/name).read_text(), name, 'exec')
    return spec, inventory


def prepare(directory, api, *, speech: Path):
    from research import prepare_h3_fast_ref2va_gpu as base
    body = Path(speech).read_bytes()
    metadata = speech_metadata(body)
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT) or directory.exists():
        raise ValueError('fresh private directory outside repository required')
    directory.mkdir(mode=0o700, parents=True)
    stage = directory/'deployment'
    preparer = FunctionType(base.prepare.__code__, dict(base.prepare.__globals__, budget_bound=_bound),
                            base.prepare.__name__, base.prepare.__defaults__)
    preparer(stage, PRIVATE/'gpu-hf-realtime-first-v2/deployment',
             PRIVATE/'compute-budget.json', api,
             publication_repo='YUGOROU/h3-fast-ref2va-smd4-gpu-ancestry-'+uuid.uuid4().hex,
             timeout_seconds=1200, attention_mode='fa4-packed', layout='portrait',
             runtime_policy='resident-visual-cached', locked_speech=None)
    closure(stage, [WORKER])
    (stage/'diagnostic-speech.wav').write_bytes(body)
    spec = json.loads((stage/'spec.json').read_text())
    spec['visual_ancestry_comparison'] = dict(SECTION)
    spec['diagnostic_speech'] = metadata
    (stage/'spec.json').write_text(json.dumps(spec))
    rewrite_allowlist(stage)
    inspect(stage)
    return stage


def submit(api, stage, token):
    """Reuse the single-POST/no-replay idle submitter without global mutation."""
    bound = FunctionType(idle.submit.__code__, dict(idle.submit.__globals__, inspect=inspect),
                         idle.submit.__name__, idle.submit.__defaults__)
    return bound(api, Path(stage), token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--speech', type=Path)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare != (args.speech is not None):
        parser.error('--prepare requires --speech; --speech requires --prepare')
    if args.prepare or args.submit:
        idle.canonical()
        import httpx
        from huggingface_hub import HfApi, set_client_factory
        from research.provision_modal_identity_secret import cached_token
        set_client_factory(lambda: httpx.Client(timeout=20, trust_env=False,
            follow_redirects=True, event_hooks={'response': [idle.sdk_redirect_guard]}))
        token = cached_token()
        api = HfApi(endpoint='https://huggingface.co', token=token)
    stage = prepare(args.directory, api, speech=args.speech) if args.prepare else args.directory/'deployment'
    if args.submit:
        job, _, _ = submit(api, stage, token)
        print(json.dumps({'job_id': job.id, 'receipt': str(stage.parent/'submission.json')}))
    else:
        spec, _ = inspect(stage)
        print(json.dumps({'offline': True, 'repo': spec['publication_repo'],
                          'timeout_seconds': 1200, 'flavor': 'h200x2'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
