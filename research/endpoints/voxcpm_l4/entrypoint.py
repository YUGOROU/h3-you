"""Finite eager VoxCPM2 endpoint; staging and launch belong to the owner."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time


def emit(row):
    print(json.dumps(row, allow_nan=False, separators=(',', ':')), flush=True)


class LoggedCandidate:
    """Observe native completion only after a full drain, independently of END."""
    def __init__(self, candidate, epoch, *, session_seconds=1200):
        if type(session_seconds) is not int or session_seconds not in (1200, 10800):
            raise ValueError('explicit finite session duration required')
        self.max_receipts = 1000 if session_seconds == 10800 else 2
        self.candidate, self.epoch = candidate, epoch
        self.sample_rate = candidate.sample_rate
        self.receipts = []
        self.runtime = {}

    def export_evidence(self):
        return dict(runtime=dict(self.runtime), completions=list(self.receipts))

    def stream(self, text):
        from research.h3_personal.voxcpm_endpoint_protocol import pcm_bytes
        digest = hashlib.sha256()
        samples = chunks = 0
        epoch = self.epoch()
        iterator = iter(self.candidate.stream(text))
        completed = False
        try:
            for chunk in iterator:
                data = pcm_bytes(chunk)
                digest.update(data)
                samples += len(data)//4
                chunks += 1
                yield chunk
            completion = self.candidate.completion or {}
            # Fixed allowlist, never arbitrary model objects/text in logs.
            facts = {k: completion.get(k) for k in (
                'reason', 'natural_stop', 'stream_exhausted', 'stop_index',
                'effective_max_len', 'started_monotonic_ns',
                'decision_monotonic_ns', 'ended_monotonic_ns')}
            row=dict(type='candidate.completion', epoch=epoch, completion=facts,
                     native_float_sha256=digest.hexdigest(), samples=samples, chunks=chunks)
            if len(self.receipts)>=self.max_receipts:raise ValueError('completion evidence bound')
            self.receipts.append(row)
            emit(row)
            completed = True
        finally:
            if hasattr(iterator, 'close'):
                iterator.close()
            completion = self.candidate.completion or {}
            if not completed and completion.get('reason') == 'cancelled':
                facts = {k: completion.get(k) for k in (
                    'reason','natural_stop','stream_exhausted','cancel_prior_reason',
                    'stop_index','effective_max_len','started_monotonic_ns',
                    'decision_monotonic_ns','ended_monotonic_ns')}
                row = dict(type='candidate.completion',epoch=epoch,completion=facts,
                           native_float_sha256=digest.hexdigest(),samples=samples,chunks=chunks)
                if len(self.receipts)>=self.max_receipts:raise ValueError('completion evidence bound')
                self.receipts.append(row)
                emit(row)


def main():
    os.umask(0o077)
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    spec = json.loads((root/'request.json').read_text())
    expires = spec['expires_at']
    session_seconds = spec.get('session_seconds', 1200)
    duration_marker = os.environ.get('H3_VOX_SESSION_SECONDS')
    if (type(session_seconds) is not int or session_seconds not in (1200, 10800)
            or (session_seconds == 10800 and duration_marker != '10800')
            or (session_seconds == 1200 and duration_marker is not None)
            or (session_seconds == 10800 and os.environ.get('H3_VOX_MEDIA_TRANSPORT') != 'https-native-v1')):
        raise ValueError('explicit matching HTTPS session duration required')
    if type(expires) not in (int, float) or not 0 < expires-time.time() <= session_seconds:
        raise ValueError('invalid lifetime')
    def expired(*_):
        raise TimeoutError('expired')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, expires-time.time()))
    key = os.environ['H3_TTS_KEY']
    session, boot = (os.environ[name] for name in ('H3_TTS_SESSION', 'H3_TTS_BOOT'))
    if len(key) < 32 or any(re.fullmatch('[0-9a-f]{32}', v) is None for v in (session, boot)):
        raise ValueError('invalid owner credentials')
    from research.hf_h3_native_tts_probe import bootstrap, models, validate_profile
    if (spec['candidate'] != 'voxcpm2' or validate_profile(spec)
            or any(spec.get(k, False) for k in ('voxcpm_trace_abba',
                'voxcpm_supply_diagnostic', 'voxcpm_stage_diagnostic', 'voxcpm_lm_compile'))):
        raise ValueError('eager candidate required')
    if len(sys.argv) == 2:
        bootstrap(root, spec, reexec_target=Path(__file__).resolve())
        return
    if sys.argv[2:] != ['--native']:
        raise ValueError('invalid entry mode')
    import torch
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise ValueError('exact one L4 required')
    from research.h3_personal.voxcpm2_streaming_candidate import load, MODEL_ID, MODEL_REVISION
    from research.h3_personal.voxcpm_endpoint_protocol import PINS
    if (spec['model']['repo'] != MODEL_ID or spec['model']['revision'] != MODEL_REVISION
            or spec['reference']['sha256'] != PINS['reference_sha256']):
        raise ValueError('approved pins required')
    model, reference = models(spec)
    candidate = load(model, reference, optimize=False)
    from research.h3_personal.voxcpm_endpoint_service import create_server
    wrapped = LoggedCandidate(candidate, lambda: server.gate.epoch, session_seconds=session_seconds)
    transport=os.environ.get('H3_VOX_MEDIA_TRANSPORT','https-native-v1')
    if transport not in ('https-native-v1','webrtc-exact-v1'):raise ValueError('explicit media transport required')
    private=transport=='webrtc-exact-v1'
    from research.h3_personal.voxcpm_endpoint_protocol import CANCEL_PROTOCOL
    cancel_protocol=os.environ.get('H3_VOX_CANCEL_PROTOCOL')
    if cancel_protocol not in (None,CANCEL_PROTOCOL) or (private and cancel_protocol is not None):
        raise ValueError('explicit HTTPS-only cancel protocol required')
    server = create_server(wrapped, host='127.0.0.1' if private else '0.0.0.0', port=7861 if private else 7860, key=key,
                           session=session, boot=boot, expires_at=expires,
                           max_requests=1000 if session_seconds == 10800 else 2,
                           session_seconds=session_seconds,
                           **({'cancel_protocol':cancel_protocol} if cancel_protocol is not None else {}))
    try:
        wrapped.runtime=dict(type='h3_voxcpm_endpoint_ready', device_name=torch.cuda.get_device_name(0),
                  device_count=torch.cuda.device_count(),
                  total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  torch=str(torch.__version__), cuda=str(torch.version.cuda),
                  optimize=False, sample_rate=candidate.sample_rate)
        emit(wrapped.runtime)
        server.serve_forever(poll_interval=.1)
    finally:
        server.gate.close()
        server.server_close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        emit(dict(type='h3_voxcpm_endpoint_failed', error_type=type(exc).__name__))
        raise SystemExit(1)
