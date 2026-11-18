"""Bounded ASR RTC acceptance without a microphone or capture binary.

Purpose: prove the single-T4 RTC ASR variant (exact DataChannel + native event
wire) actually transcribes with the real model before any attended capture.

Scope of the evidence, explicitly limited:
  - input is a fixed PUBLIC sample PCM already stored in the parent campaign;
    no microphone, camera, screen, meeting or private voice is touched,
  - no Pi, no Vox/TTS, no H3, no browser, no Discord,
  - one serial turn only; the lease is opened, used once and paused on close.

This is NOT speech E2E, NOT acoustic/device acceptance and NOT a claim that the
Comet fresh-capture path works. Offline by default; `--run` performs one finite
Endpoint deployment (update + resume + pause) as a separate approved operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.h3_personal.resident_endpoint import PRIVATE, ResidentEndpoint  # noqa: E402
from research.h3_personal.asr_rtc_factory import ASRRTCFactory  # noqa: E402
from research.h3_personal.endpoint_asr import finalized_covers, native_completed_text  # noqa: E402

# Content pin of the already-reviewed Japanese PUBLIC sample used by the
# 2026-10-19 public ASR validation. Path is not trusted; bytes are.
PUBLIC_SAMPLE_SHA256 = 'de940ea5e91cb0c94161dfeb6ecda3e27b5dcef102a22f8c9235612f28d4c20d'


def pcm_source(path, max_seconds):
    """Return a callable yielding bounded 16k mono PCM16 from a public sample."""
    with wave.open(str(path)) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError('mono 16k PCM16 sample required')
        raw = wav.readframes(int(min(max_seconds, wav.getnframes()/wav.getframerate())*16000))
    import io
    return lambda: io.BytesIO(raw)


def validate_transcript(transcript):
    if type(transcript) is not dict:
        raise ValueError('transcript required')
    text = transcript.get('text')
    events = transcript.get('events')
    if (type(text) is not str or not 1 <= len(text.strip()) <= 500
            or transcript.get('final') is not True or transcript.get('errors') != []
            or transcript.get('input_committed') is not True
            or transcript.get('completion_policy') != 'input-finalized'
            or type(events) is not list or not events
            or type(transcript.get('audio_sent_seconds')) not in (int, float)
            or transcript['audio_sent_seconds'] <= 0
            or not finalized_covers(transcript.get('finalized_audio_seconds'), transcript['audio_sent_seconds'])):
        raise ValueError('native finalized RTC receipt required')
    finals = [e for e in events if e.get('final') is True]
    if not finals or native_completed_text(events) != text.strip():
        raise ValueError('native completed text required')
    carrier = transcript.get('rtc_carrier') or {}
    if (transcript.get('media_transport') != 'webrtc-exact-v1'
            or carrier.get('transport_complete') is not True
            or carrier.get('native_eos_verified') is not False
            or type(carrier.get('bytes')) is not int or not 0<carrier['bytes']<=4*1024*1024
            or type(carrier.get('sha256')) is not str
            or len(carrier['sha256'])!=64 or any(c not in '0123456789abcdef' for c in carrier['sha256'])):
        raise ValueError('exact RTC carrier completion required')
    return dict(text=text.strip(), events=len(events), carrier=carrier,
                audio_sent_seconds=transcript['audio_sent_seconds'],
                finalized_audio_seconds=transcript['finalized_audio_seconds'])


def run(directory, parent, sample, seconds):
    """One bounded turn against a freshly deployed single-T4 RTC ASR Endpoint."""
    from research.h3_personal.endpoint_asr import transcribe
    root = Path(os.path.abspath(directory))
    if not root.is_relative_to(PRIVATE) or root.exists():
        raise ValueError('fresh private directory required')
    sample = Path(sample).resolve()
    if not sample.is_file() or hashlib.sha256(sample.read_bytes()).hexdigest() != PUBLIC_SAMPLE_SHA256:
        raise ValueError('reviewed public sample bytes required')
    if type(seconds) is not int or not 8 <= seconds <= 20:
        raise ValueError('bounded 8..20 second input window required')
    factory = ASRRTCFactory()
    receipt = dict(schema='h3-asr-rtc-acceptance-v1', scope='public sample, no device/E2E',
                   mic=False, camera=False, screen=False, browser=False, h3=False, tts=False,
                   input=str(sample.name), seconds=seconds, complete=False)
    root.mkdir(mode=0o700)  # Lease dir must be fresh but its campaign parent must exist.
    lease = ResidentEndpoint.open(root/'lease', Path(parent), accounting='operator-managed', rtc_factory=factory)
    turn = None
    try:
        deadline = time.monotonic()+min(90, max(0, lease.deadline-time.time()-90))
        lease._rtc_epoch += 1  # Serial epoch, reserved before the uncertain allocation.
        turn = factory.open_turn(endpoint=lease._url, token=lease._token, key=lease._key,
                                 session=lease.lease_id, epoch=lease._rtc_epoch,
                                 expires_at=lease.deadline, deadline=deadline)
        transcript = transcribe(lease._url, token=lease._token, owner_key=lease._key,
                                pcm=pcm_source(sample, seconds), timeout=90,
                                max_audio_seconds=seconds, completion_policy='input-finalized',
                                _transport=turn.transport)
        # Retain native evidence even if independent carrier completion fails.
        receipt['native_transcript']=transcript
        receipt['rtc_tail_status']=getattr(turn.transport,'tail_status',None)
        turn.close(require_complete=True)
        transcript['media_transport']='webrtc-exact-v1'
        transcript['rtc_carrier']=turn.receipt
        receipt['rtc_carrier']=turn.receipt
        turn = None  # Closed exactly once; the finally path handles failures only.
        receipt.update(transcript=validate_transcript(transcript), complete=True)
        return receipt
    finally:
        if turn is not None:
            try:
                turn.close(require_complete=False)
            except BaseException:
                receipt['carrier_cleanup_unconfirmed'] = True
        lease.close()
        root.mkdir(mode=0o700, exist_ok=True)
        (root/'result.json').write_text(json.dumps(receipt, indent=1, ensure_ascii=False))
        (root/'result.json').chmod(0o600)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path)
    p.add_argument('--parent', type=Path)
    p.add_argument('--sample', type=Path)
    p.add_argument('--seconds', type=int, default=8)
    p.add_argument('--run', action='store_true')
    args = p.parse_args(argv)
    if not args.run:
        print(json.dumps(dict(scope='offline', gpu_started=False, endpoint_updated=False,
                              device_access=False, capture_started=False, browser_opened=False,
                              live_acceptance=False, instance='nvidia-t4 x1', min_max_replica=1,
                              transport='webrtc-exact-v1', input='public sample PCM only')))
        return
    for required in ('directory', 'parent', 'sample'):
        if getattr(args, required) is None:
            raise SystemExit(f'--{required} required for --run')
    print(json.dumps(dict(scope='live', phase='deploying one single-replica RTC ASR Endpoint')), flush=True)
    receipt = run(args.directory, args.parent, args.sample, args.seconds)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
