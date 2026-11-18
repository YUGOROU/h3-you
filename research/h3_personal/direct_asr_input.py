"""Attended input callback for an already-open, caller-owned ASR lease.

Construction is not capture permission. Invoke only after per-operation user
approval. This adapter never opens, resumes, controls or closes the endpoint;
ResidentEndpoint.capture retains its own failure/cleanup semantics and existing
transcript receipts. No additional persistence is performed here.
"""
import math
from pathlib import Path
import re
import time
import uuid

from research.h3_personal.endpoint_asr import (
    FINAL, finalized_covers, native_completed_text,
)
from research.h3_personal.resident_endpoint import TurnRejected
from research.h3_personal.zoom_input import CaptureInterrupted, InputBlocked


def metadata(endpoint):
    """Expected worker contract (not evidence that capture has occurred)."""
    intent = endpoint.intent
    revision = intent['model']['revision']
    name = intent['name']
    if (name != 'h3-asr-t4-7c926aae18' or name != endpoint.name
            or type(revision) is not str
            or re.fullmatch('[0-9a-f]{40}', revision) is None):
        raise ValueError('pinned native endpoint required')
    return dict(policy='endpoint-native-v1', endpoint_name=name,
                revision=revision, endpoint_source='native_completed',
                completion_policy='input-finalized', capture_stopped=True,
                cleanup_complete=True)


def _validate(receipt, *, require_rtc=False):
    text = receipt.get('text')
    events = receipt.get('events')
    if (type(text) is not str or not 1 <= len(text.strip()) <= 500
            or receipt.get('final') is not True or receipt.get('errors') != []
            or receipt.get('capture_stopped') is not True
            or receipt.get('cleanup_complete') is not True
            or receipt.get('input_committed') is not True
            or receipt.get('completion_policy') != 'input-finalized'
            or type(events) is not list or not events
            or type(receipt.get('audio_sent_seconds')) not in (int, float)
            or receipt['audio_sent_seconds'] <= 0
            or not finalized_covers(receipt.get('finalized_audio_seconds'),
                                    receipt['audio_sent_seconds'])):
        raise TurnRejected('native finalized capture receipt required')
    finals = [e for e in events if e.get('final') is True]
    if (not finals or any(e.get('native_event_type') != FINAL for e in finals)
            or native_completed_text(events) != text.strip()):
        raise TurnRejected('native completed text required')
    if require_rtc:
        carrier=receipt.get('rtc_carrier')
        if (receipt.get('media_transport')!='webrtc-exact-v1'
                or type(carrier) is not dict
                or carrier.get('schema')!='h3-asr-rtc-ipc-complete-v1'
                or carrier.get('transport_complete') is not True
                or carrier.get('native_eos_verified') is not False
                or type(carrier.get('bytes')) is not int or not 0<carrier['bytes']<=4*1024*1024
                or type(carrier.get('sha256')) is not str
                or re.fullmatch('[a-f0-9]{64}',carrier['sha256']) is None
                or type(carrier.get('local_eof_already_closed')) is not bool):
            raise TurnRejected('exact RTC carrier receipt required')
    return text.strip()


class DirectASRInput:
    """Callable ``source(context=..., deadline=<monotonic>)``; explicit finite bounds.

    Defaults: two accepted turns and three attempts, including rejected/silent turns.
    Long-session callers may explicitly raise both bounded counters. Exhaustion or
    insufficient time returns None so the conversation can drain normally.
    Unexpected failures propagate without adapter-owned endpoint cleanup.
    """
    def __init__(self, endpoint, *, capture_seconds=45, input_pipeline=None,
                 max_attempts=3, max_turns=2):
        if (type(max_attempts) is not int or not 1 <= max_attempts <= 1000
                or type(max_turns) is not int or not 1 <= max_turns <= max_attempts):
            raise ValueError('explicit finite capture attempt/turn bounds required')
        self.max_attempts, self.max_turns = max_attempts, max_turns
        if type(capture_seconds) is not int or not 8 <= capture_seconds <= 45:
            raise ValueError('capture_seconds must be 8..45')
        self.endpoint = endpoint
        self.capture_seconds = capture_seconds
        self.input_pipeline = input_pipeline
        self.expected_asr = metadata(endpoint)
        self._media_transport=endpoint.intent['model'].get('env',{}).get('H3_ASR_MEDIA_TRANSPORT')
        if self._media_transport not in (None,'webrtc-exact-v1'):
            raise ValueError('unsupported ASR media transport')
        if input_pipeline is not None:
            self.expected_asr['input_source'] = input_pipeline.source_contract.copy()
        self.phase_timing = None  # Optional observer attached after construction.
        self.accepted = self.attempts = 0
        self._prefix = 'direct-asr-' + uuid.uuid4().hex

    def note_reply_published(self):
        if self.input_pipeline is not None:
            self.input_pipeline.note_reply_published()

    def __call__(self, *, context, deadline):
        # Context is supplied by run_h3_relay_conversation; never interpolate
        # its session identifier, transcript, endpoint URL or credentials in logs.
        if not isinstance(context, dict):
            raise ValueError('conversation context required')
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('finite monotonic deadline required')
        while self.accepted < self.max_turns and self.attempts < self.max_attempts:
            endpoint = self.endpoint
            if (endpoint.closed or deadline - time.monotonic() < 80
                    or endpoint.deadline - time.time() < 80):
                return None
            expected = metadata(endpoint)
            if self.input_pipeline is not None:
                expected['input_source'] = self.input_pipeline.source_contract.copy()
            if endpoint.intent['model'].get('env',{}).get('H3_ASR_MEDIA_TRANSPORT')!=self._media_transport:
                raise ValueError('endpoint transport drift')
            if expected != self.expected_asr:
                raise ValueError('endpoint pin drift')
            self.attempts += 1
            root = Path(endpoint.root).resolve()
            directory = root / f'{self._prefix}-{self.attempts}'
            if self.input_pipeline is None:
                print(f'ASR capture starts now (VAD boundary, maximum {self.capture_seconds} seconds).', flush=True)
            else:
                print(f'Awaiting approval for next {self.capture_seconds}s Zoom input window.', flush=True)
            planned_start = None if self.input_pipeline is None else time.time() + 3
            if self.input_pipeline is not None:
                self.input_pipeline.conversation_deadline = deadline
            pipeline_kwargs = {} if self.input_pipeline is None else {'input_pipeline': self.input_pipeline}
            try:
                if self.phase_timing is None:
                    receipt = endpoint.capture(directory, planned_start,
                                               capture_seconds=self.capture_seconds, **pipeline_kwargs)
                else:
                    with self.phase_timing.stage('asr_capture'):
                        receipt = endpoint.capture(
                            directory, planned_start, capture_seconds=self.capture_seconds,
                            on_asr_event=self.phase_timing.on_asr_event, **pipeline_kwargs)
                text = _validate(receipt,require_rtc=self._media_transport=='webrtc-exact-v1')
            except TurnRejected:
                continue  # NoSpeech is a TurnRejected; keep the warm lease.
            except CaptureInterrupted:
                continue  # Transient capture loss keeps the lease; bounded retries.
            except InputBlocked:
                return None  # Attended operator declined/ended input; no replay.
            if time.monotonic() >= deadline or endpoint.closed:
                raise RuntimeError('stale ASR capture completion')
            self.accepted += 1
            asr = self.expected_asr.copy()
            # These are evidence from the actual capture, not invented flags.
            asr.update(capture_stopped=receipt['capture_stopped'],
                       cleanup_complete=receipt['cleanup_complete'])
            if self.input_pipeline is not None:
                from research.h3_personal.zoom_turn_input import validate_zoom_capture
                if receipt.get('input_source') != asr['input_source']:
                    raise ValueError('actual Zoom source drift')
                asr['zoom_capture'] = validate_zoom_capture(receipt.get('zoom_capture'), asr['input_source'])
            return dict(text=text, asr=asr)
        return None
