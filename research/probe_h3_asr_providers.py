"""Bounded HF-router ASR probe using PUBLIC JFK audio only, no device/GPU work.

Default is offline plan. --run requires HF_TOKEN in this child environment.
Uses official huggingface.js DeepInfra ASR route and live provider model IDs.
Whole-file upload + stream=true is an OUTPUT-stream probe, NOT proof of live
incremental AUDIO input support. Never classify unknown realtime as unsupported.
"""
import argparse
from decimal import Decimal
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import time
import uuid

MODELS = (
    ('openai/whisper-large-v3', 'openai/whisper-large-v3'),
    ('nvidia/nemotron-3.5-asr-streaming-0.6b', 'nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b'),
    ('Qwen/Qwen3-ASR-1.7B', 'Qwen/Qwen3-ASR-1.7B'),
)
ROUTE = '/deepinfra/v1/openai/audio/transcriptions'
SOURCE = 'https://raw.githubusercontent.com/huggingface/huggingface.js/main/packages/inference/src/providers/deepinfra.ts'
TOGETHER_MODELS = tuple((model, model) for model, _ in MODELS[:2])
TOGETHER_ROUTE = '/together/v1/audio/transcriptions'
CAP = Decimal('216.613337')
ALLOWANCE = Decimal('0.30')  # conservative finite six-call allocation, not a metered invoice


def multipart(audio, model, stream=False):
    boundary = 'h3-asr-' + uuid.uuid4().hex
    fields = dict(model=model, language='en', response_format='json')
    if stream:
        fields['stream'] = 'true'
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="jfk.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
    parts.extend([audio, f'\r\n--{boundary}--\r\n'.encode()])
    return b''.join(parts), 'multipart/form-data; boundary=' + boundary


def reserve(ledger, allowance=ALLOWANCE, max_calls=6):
    key = 'hf-asr-provider-probe-' + uuid.uuid4().hex
    with ledger.with_suffix('.json.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text())
        bound = Decimal(str(data['spent_usd'])) + sum(Decimal(str(r['upper_bound_usd'])) for r in data['reservations'].values())
        if bound + allowance > CAP:
            raise ValueError('existing experiment cap exceeded')
        data['reservations'][key] = dict(upper_bound_usd=float(allowance), scope='finite-public-ASR-router-requests', created_at=time.time(), max_calls=max_calls)
        ledger.write_text(json.dumps(data, indent=2))
    return key


def reconcile(ledger, key, results=()):
    with ledger.with_suffix('.json.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text())
        row = data['reservations'].pop(key)
        router_rejected = bool(results) and all(r.get('http_status') == 402 and 'Pay-as-you go is not enabled for provider' in r.get('body', '') for r in results)
        charge = Decimal('0') if router_rejected else Decimal(str(row['upper_bound_usd']))
        data['spent_usd'] = float(Decimal(str(data['spent_usd'])) + charge)
        basis = 'all requests router-rejected HTTP402; no provider inference executed' if router_rejected else 'entire finite-call allowance retained conservatively; actual API invoice pending'
        data.setdefault('terminal_reconciliations', {})[key] = dict(status='probe-ended', charge_bound_usd=str(charge), invoice_verified=False, basis=basis, at=time.time())
        ledger.write_text(json.dumps(data, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    p.add_argument('--provider', choices=('deepinfra', 'together'), default='deepinfra')
    args = p.parse_args()
    models = MODELS if args.provider == 'deepinfra' else TOGETHER_MODELS
    route = ROUTE if args.provider == 'deepinfra' else TOGETHER_ROUTE
    source = SOURCE.replace('deepinfra.ts', args.provider + '.ts')
    max_calls = len(models) * 2
    allowance = Decimal('0.05') * max_calls
    if not args.run:
        print(json.dumps(dict(host='router.huggingface.co', route=route, provider=args.provider, max_calls=max_calls, public_audio_only=True, allocation_usd=str(allowance), true_input_streaming_tested=False)))
        return
    token = os.environ.get('HF_TOKEN', '')
    if not token or '\n' in token:
        raise ValueError('HF_TOKEN required in child environment')
    root = args.directory.resolve()
    receipt = json.loads((root/'model-sample-receipt.json').read_text())
    expected_url = 'https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/v0.2.0/test_files/asr/wav/test/jfk.wav'
    audio = (root/'jfk.wav').read_bytes()
    if receipt['sample_url'] != expected_url or hashlib.sha256(audio).hexdigest() != receipt['sample_sha256'] or len(audio) > 400000:
        raise ValueError('pinned public sample required')
    # Verify live official model mapping before paid calls (no dependency upgrades).
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for model, provider_id in models:
        mapping = api.model_info(model, expand=['inferenceProviderMapping']).inference_provider_mapping
        if not any(m.provider == args.provider and m.provider_id == provider_id and m.task == 'automatic-speech-recognition' and m.status == 'live' for m in mapping):
            raise ValueError('live official provider mapping differs')
    result_path = root/('paid-hf-asr-results.json' if args.provider == 'deepinfra' else 'paid-hf-asr-results-together.json')
    if result_path.exists():
        raise ValueError('do not replay an already executed paid probe')
    ledger = root.parent/'compute-budget.json'
    key = reserve(ledger, allowance, max_calls)
    results = []
    try:
        for stream in (False, True):
            for model, provider_id in models:
                body, content_type = multipart(audio, provider_id, stream)
                started = time.monotonic()
                conn = http.client.HTTPSConnection('router.huggingface.co', timeout=60)
                row = dict(model=model, provider_id=provider_id, provider=args.provider, route=route, stream_parameter=stream,
                           input_mode='complete_public_wav', true_audio_input_streaming='UNVERIFIED')
                try:
                    conn.request('POST', route, body=body, headers={'Authorization':'Bearer '+token, 'Content-Type':content_type, 'Accept':'text/event-stream' if stream else 'application/json'})
                    response = conn.getresponse()
                    row.update(http_status=response.status, content_type=response.getheader('Content-Type'), headers_seconds=time.monotonic()-started)
                    chunks = []
                    first = None
                    count = 0
                    while True:
                        if time.monotonic() - started > 65:
                            raise TimeoutError('finite response deadline')
                        chunk = response.read1(8192)
                        if not chunk:
                            break
                        if first is None:
                            first = time.monotonic() - started
                        count += len(chunk)
                        if count > 65536:
                            raise ValueError('response size bound')
                        chunks.append(chunk)
                    text = b''.join(chunks).decode(errors='replace').replace(token, '[REDACTED]')
                    row.update(first_body_seconds=first, total_seconds=time.monotonic()-started,
                               body=text, output_stream_observed='text/event-stream' in (row['content_type'] or '') and 'data:' in text)
                except Exception as exc:
                    row.update(error_type=type(exc).__name__, total_seconds=time.monotonic()-started)
                finally:
                    conn.close()
                results.append(row)
                result_path.write_text(json.dumps(dict(reservation=key, official_adapter_source=source, audio_public=True, models=results, invoice_verified=False), indent=2))
                result_path.chmod(0o600)
                # Never print provider payloads, credentials or request headers.
                print(json.dumps({k:v for k,v in row.items() if k != 'body'}), flush=True)
    finally:
        reconcile(ledger, key, results)


if __name__ == '__main__':
    main()
