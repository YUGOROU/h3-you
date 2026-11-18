"""Explicit attended Zoom pipeline. Construction never accesses capture devices.

Parent: build_native(...) only after explicit preflight; pass pipeline to
DirectASRInput(..., capture_seconds=8, input_pipeline=pipeline). Call
note_reply_published() after every send (even an uncertain send).
"""
import hashlib
import math
import select
import struct
import sys
import threading
import time
from pathlib import Path

from research.h3_personal.zoom_input import (CaptureInterrupted, InputBlocked,
                                             ZoomPCMReader, source_contract)


class AttendedGuard:
    """Operator policy, NOT a calibrated EchoGuard or automatic echo detection."""
    def __init__(self, cancel=None):
        self.cancel = cancel
        self.lock = threading.RLock()
        self.epoch = 0
        self.deadline = 0
        self.closed = True

    def block(self):
        with self.lock:
            self.epoch += 1
            self.closed = True

    def arm(self, deadline):
        with self.lock:
            self.epoch += 1
            self.deadline = deadline
            self.closed = False
            return self.epoch

    def check(self, epoch):
        with self.lock:
            if (self.closed or epoch != self.epoch or time.monotonic() >= self.deadline
                    or (self.cancel is not None and self.cancel.is_set())):
                raise InputBlocked('attended input expired or invalidated')

    def dispatch(self, epoch, fn):
        with self.lock:
            self.check(epoch)
            return fn()


def terminal_confirmation(*, deadline, cancel):
    if not sys.stdin.isatty():
        raise InputBlocked('attended terminal or explicit confirmation callback required')
    print('Approve THIS Zoom application-audio capture: ALL participants consent; previous AI reply FULLY heard and ended (or first turn); remote echo absent. Type ZOOM-IDLE:', flush=True)
    while time.monotonic() < deadline and not cancel.is_set():
        if select.select([sys.stdin], [], [], min(.1, max(0, deadline-time.monotonic())))[0]:
            return sys.stdin.readline().strip() == 'ZOOM-IDLE'
    return False


class BoundedRaw:
    def __init__(self, raw, samples):
        self.raw, self.remaining = raw, samples * 2

    def read(self, size):
        if not self.remaining:
            return b''
        data = self.raw.read(min(size, self.remaining))
        if not isinstance(data, bytes) or len(data) % 2 or len(data) > min(size, self.remaining):
            raise ValueError('invalid raw PCM framing')
        if not data:
            # Premature stream end is capture-infrastructure loss (process
            # churn/tap abort/queue full), not an input-eligibility decline.
            raise CaptureInterrupted('capture ended before fixed window; no replay')
        self.remaining -= len(data)
        return data


class ZoomTurnInput:
    capture_argv = ['--stream', '--app', 'zoom']

    def __init__(self, extractor, source, *, confirmation=terminal_confirmation,
                 all_participants_consented=False, cancel=None, tail_validator=None,
                 tail_validation_required=True):
        expected = source_contract(reference_sha256=source['reference_sha256'],
                                   capture_source_sha256=source['capture_source_sha256'])
        if source != expected or not callable(confirmation):
            raise ValueError('exact source and confirmation callback required')
        if type(tail_validation_required) is not bool:
            raise ValueError('explicit tail_validation_required bool required')
        if tail_validator is not None and not callable(tail_validator):
            raise ValueError('tail_validator must be callable when provided')
        self.source_contract = expected
        self.extractor = extractor
        self.confirmation = confirmation
        self.tail_validator = tail_validator
        self.tail_validation_required = tail_validation_required
        self.consented = all_participants_consented is True
        self.cancel = cancel if cancel is not None else threading.Event()
        self.guard = AttendedGuard(self.cancel)
        self.reader = None
        self.epoch = None
        self.conversation_deadline = None

    def note_reply_published(self):
        self.guard.block()

    def prepare_capture(self, *, deadline):
        self.guard.block()
        if not self.consented or not math.isfinite(deadline):
            raise InputBlocked('explicit all-participant consent required')
        if self.conversation_deadline is not None:
            deadline = min(deadline, self.conversation_deadline)
        end = min(deadline, time.monotonic()+45)
        if self.cancel.is_set() or time.monotonic() >= end:
            raise InputBlocked('capture cancelled or deadline expired before validation')
        epoch = self.guard.epoch
        # Documented live gate: measured remote echo-tail validation BEFORE the
        # operator confirmation. Operator judgement alone cannot exclude the
        # network-delayed AI self-echo (TSE preserves clone audio).
        if self.tail_validation_required:
            if not callable(self.tail_validator):
                raise InputBlocked('validated remote echo tail required; '
                                   'no operator-judgement bypass')
            if self.tail_validator(deadline=end, cancel=self.cancel) is not True:
                raise InputBlocked('remote echo tail not validated; no capture')
        # Callback must honor the supplied finite deadline/cancellation event.
        with self.guard.lock:
            if self.cancel.is_set() or time.monotonic() >= end or self.guard.epoch != epoch:
                raise InputBlocked('remote tail validation expired or invalidated')
        confirmed = self.confirmation(deadline=end, cancel=self.cancel)
        # Check and arm atomically relative to note_reply_published(). Never
        # reopen an epoch invalidated while the validator or TTY was running.
        with self.guard.lock:
            if (confirmed is not True or self.cancel.is_set() or time.monotonic() >= end
                    or self.guard.epoch != epoch):
                raise InputBlocked('fresh attended remote-idle confirmation required')
            self.epoch = self.guard.arm(min(deadline, time.monotonic()+65))
            self.reader = None

    def wrap_capture(self, raw, *, capture_seconds):
        if type(capture_seconds) is not int or not 8 <= capture_seconds <= 45:
            raise ValueError('fixed8..45 second capture required')
        if self.reader is not None:
            raise InputBlocked('capture cannot be reused')
        self.reader = ZoomPCMReader(BoundedRaw(raw, capture_seconds*16000),
                                    self.extractor, self.guard, self.epoch)
        return self.reader

    def finish_capture(self, last, transcript, *, capture_seconds):
        reader = self.reader
        if reader is None:
            raise ValueError('missing extracted capture')
        # endpoint sender stops at exactly max_audio_seconds without an EOF read.
        # Drain only the zero-length boundary/flush; never submit a hidden tail.
        if reader.read(5120) != b'':
            raise ValueError('ASR left unsubmitted extracted tail')
        r = reader.receipt()
        evidence = dict(schema='h3-zoom-asr-capture-v1',
            profile_sha256=self.source_contract['profile_sha256'],
            **{k:r[k] for k in ('input_epoch','captured_samples','extracted_samples',
                'submitted_samples','eof','pending_samples','discarded_pending_samples','raw_audio_fallback')},
            capture_frames=last.get('frames'), capture_stopped=last.get('type') == 'stopped',
            capture_cleanup=last.get('cleanup'), capture_app=last.get('app'),
            mute_behavior=last.get('mute_behavior'), operator_confirmed_remote_idle=True)
        validate_zoom_capture(evidence, self.source_contract)
        n = capture_seconds*16000
        if r['failed'] or r['submitted_samples'] != n or abs(transcript['audio_sent_seconds']-n/16000)>1/16000:
            raise ValueError('fixed extracted native ASR coverage required')
        self.guard.block()
        return evidence


def validate_zoom_capture(value, source):
    keys = {'schema','profile_sha256','input_epoch','captured_samples','extracted_samples',
            'submitted_samples','capture_frames','capture_stopped','capture_cleanup','capture_app',
            'mute_behavior','eof','pending_samples','discarded_pending_samples',
            'raw_audio_fallback','operator_confirmed_remote_idle'}
    if type(value) is not dict or set(value) != keys:
        raise ValueError('exact Zoom capture evidence required')
    n = value['captured_samples']
    if (value['schema'] != 'h3-zoom-asr-capture-v1' or value['profile_sha256'] != source['profile_sha256']
            or type(n) is not int or not 0 < n <= 720000
            or any(type(value[k]) is not int or value[k] != n for k in ('extracted_samples','submitted_samples'))
            or type(value['capture_frames']) is not int or not n <= value['capture_frames'] <= 720000
            or type(value['input_epoch']) is not int or value['input_epoch'] <= 0
            or any(value[k] is not True for k in ('capture_stopped','capture_cleanup','eof','operator_confirmed_remote_idle'))
            or any(type(value[k]) is not int or value[k] != 0 for k in ('pending_samples','discarded_pending_samples'))
            or value['raw_audio_fallback'] is not False or value['capture_app'] != 'zoom'
            or value['mute_behavior'] != 'unmuted'):
        raise ValueError('invalid Zoom extracted capture evidence')
    return value.copy()


def build_native(*, model_dir, reference, reference_file_sha256, reference_pcm_sha256,
                 capture_source, capture_source_sha256, reference_offset=0, **kwargs):
    """Explicit native CPU4/batch8 enrollment, once per session; no saved audio."""
    from research.run_h3_zoom_preflight import convert, file_sha
    if not math.isfinite(reference_offset) or not 0 <= reference_offset <= 27:
        raise ValueError('finite reference offset required')
    if file_sha(Path(reference)) != reference_file_sha256 or file_sha(Path(capture_source)) != capture_source_sha256:
        raise ValueError('local reference/capture source hash mismatch')
    pcm = convert(reference, reference_offset, 3, 'f32le')
    if hashlib.sha256(pcm).hexdigest() != reference_pcm_sha256:
        raise ValueError('canonical3sec PCM hash mismatch')
    if not isinstance(pcm, bytes) or len(pcm) != 48000 * 4:
        raise ValueError('canonical3sec PCM must be exactly 48000 little-endian float32 samples')
    from research.h3_personal.voxlock_streaming import load_cpu, VoxLockStream
    enroll, chunk = load_cpu(model_dir, threads=4)
    # stdlib decode so the enrollment gate is testable without numpy; VoxLockStream validates.
    extractor = VoxLockStream(enroll, chunk, list(struct.unpack('<48000f', pcm)), batch_frames=8)
    return ZoomTurnInput(extractor, source_contract(reference_sha256=reference_pcm_sha256,
                         capture_source_sha256=capture_source_sha256), **kwargs)
