"""Bounded, local-clock-only stage evidence; never accepts arbitrary metadata."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time
import threading


class PhaseTiming:
    STAGES = frozenset(('input', 'pi', 'publication', 'asr_capture',
                        'asr_partial', 'asr_final', 'asr_input_finalized',
                        'tts_sentence', 'tts_complete_wav', 'canonical_pcm',
                        'speech_pcm_ready', 'tts_eos', 'h3_window',
                        'first_speech_packet_ready','h3_initial_coverage_ready',
                        'interrupt_received','speech_retired','quarantine_start','paired_cut_ready'))

    def __init__(self, *, clock=time.monotonic, component='mac_controller', emit=False):
        if component not in ('mac_controller', 'gpu_worker') or type(emit) is not bool:
            raise ValueError('known timing component and explicit emit flag required')
        self.component, self.emit = component, emit
        self._lock = threading.Lock()
        self._asr_seen = set()
        self._asr_counts = dict(partial=0, native_completed=0, input_finalized=0)
        self.clock = clock
        self.started = clock()
        self.events = []
        self.dropped = 0
        self.completed = False

    def record(self, stage, *, duration=0, status='completed'):
        if stage not in self.STAGES or status not in ('completed', 'failed'):
            raise ValueError('unknown timing category')
        elapsed = self.clock() - self.started
        if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                   for v in (elapsed, duration)):
            raise ValueError('invalid local duration')
        event = dict(stage=stage, elapsed_seconds=elapsed,
                     duration_seconds=duration, status=status)
        with self._lock:
            if len(self.events) >= 256:
                self.dropped = min(1000000, self.dropped + 1)
                return
            self.events.append(event)
        if self.emit:
            print(json.dumps(dict(type='h3_stage_timing', component=self.component,
                                  clock='local_monotonic',origin_monotonic=self.started, **event), allow_nan=False), flush=True)

    @contextmanager
    def stage(self, name):
        if name == 'asr_capture':
            with self._lock:
                self._asr_seen.clear()
        start = self.clock()
        try:
            yield
        except BaseException:
            self.record(name, duration=self.clock()-start, status='failed')
            raise
        else:
            self.record(name, duration=self.clock()-start)

    def on_asr_event(self, event):
        # Deliberately ignore all text, remote timestamps and identifiers.
        if type(event) is dict:
            name = {'partial': 'asr_partial', 'native_completed': 'asr_final',
                    'input_finalized': 'asr_input_finalized'}.get(event.get('event'))
            if name:
                with self._lock:
                    key = event['event']
                    self._asr_counts[key] = min(1000000, self._asr_counts[key] + 1)
                    if key in self._asr_seen:
                        return
                    self._asr_seen.add(key)
                self.record(name)

    def persist(self, root):
        root = Path(root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or root.stat().st_mode & 0o077:
            raise ValueError('private timing directory required')
        path = root / 'stage-timing.json'
        with self._lock:
            payload = json.dumps(dict(schema='h3-stage-timing-v1', clock='local_monotonic',
                                      component=self.component,origin_monotonic=self.started,
                                      status='completed' if self.completed else 'failed',
                                      events=self.events, dropped=self.dropped,
                                      asr_event_counts=self._asr_counts)).encode()
        # O_EXCL and O_NOFOLLOW: never replay/replace an existing receipt.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
