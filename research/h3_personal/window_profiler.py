"""Best-effort bounded per-window timing; never records payloads or changes math.

One worker-local monotonic origin is shared with its TTS thread. Intervals are
wall durations, not transformer-only/CUDA-event timings. No new CUDA sync.
"""
from contextlib import contextmanager
import json
import math
import sys
import threading
import time


def _stdout(row):
    # One newline-terminated write reduces interleaving with concurrent stages.
    sys.stdout.write(json.dumps(row, allow_nan=False) + '\n')
    sys.stdout.flush()


class WindowProfiler:
    STAGES = frozenset(('driver_total', 'driver_prepare', 'driver_backend', 'driver_complete',
                        'backend_total', 'role_switch', 'manifest_write', 'native_call',
                        'native_generate_sync', 'evidence_read', 'evidence_validate',
                        'ancestry_read', 'video_export'))
    ACTIVITIES = frozenset(('tts_complete_wav', 'canonical_pcm'))

    def __init__(self, *, clock=time.monotonic, emit=None, window_cap=144):
        if type(window_cap) is not int or not 1 <= window_cap <= 2542:
            raise ValueError('bounded profiler window cap required')
        self.clock = clock
        self.origin = clock()
        self._emit = emit if emit is not None else _stdout
        self._lock = threading.RLock()
        self._window_cap = window_cap
        self._rows = {}
        self._activities = []
        self._emitted = set()
        self.dropped = 0

    def _drop(self):
        with self._lock: self.dropped = min(1000000, self.dropped + 1)

    def _now(self):
        try: return self.clock()
        except BaseException:
            self._drop()
            return None

    def interval(self, index, stage, begin, end, *, status='completed'):
        try:
            if (type(index) is not int or not 0 <= index < 2542 or stage not in self.STAGES
                    or status not in ('completed', 'failed') or any(type(t) not in (int, float)
                    or not math.isfinite(t) for t in (begin, end, self.origin))
                    or not 0 <= begin-self.origin <= end-self.origin <= 10800):
                raise ValueError('invalid bounded timing')
            value = dict(start_seconds=begin-self.origin, end_seconds=end-self.origin,
                         duration_seconds=end-begin, status=status)
            with self._lock:
                row = self._rows.setdefault(index, dict(index=index, stages={}))
                if stage in row['stages']:
                    prior = row['stages'][stage]
                    if stage not in ('driver_prepare', 'driver_total') or value['start_seconds'] < prior['end_seconds']:
                        raise ValueError('duplicate stage')
                    value['attempts'] = min(1000000, prior.get('attempts', 1) + 1)
                    value['prior_attempt_seconds'] = prior.get('prior_attempt_seconds', 0) + prior['duration_seconds']
                row['stages'][stage] = value
                while len(self._rows) > self._window_cap:
                    del self._rows[min(self._rows)]
        except BaseException:
            self._drop()

    @contextmanager
    def span(self, index, stage):
        begin = self._now()
        try:
            yield
        except BaseException:
            self.interval(index, stage, begin, self._now(), status='failed')
            raise
        else:
            self.interval(index, stage, begin, self._now())

    def role(self, index, role):
        try:
            if type(index) is not int or not 0 <= index < 2542 or role not in ('speech', 'listening'):
                raise ValueError('bounded enum required')
            with self._lock:
                row = self._rows.setdefault(index, dict(index=index, stages={}))
                row['role'] = role
        except BaseException:
            self._drop()

    def _send(self, value):
        try: self._emit(value)
        except BaseException: self._drop()

    @contextmanager
    def activity(self, name):
        begin = self._now()
        status = 'completed'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            end = self._now()
            try:
                if (name not in self.ACTIVITIES or any(type(t) not in (int, float)
                        or not math.isfinite(t) for t in (begin, end, self.origin))
                        or not 0 <= begin-self.origin <= end-self.origin <= 10800):
                    raise ValueError('invalid activity')
                with self._lock:
                    if len(self._activities) >= 16: raise ValueError('bounded activities')
                    value = dict(type='h3_worker_activity', clock='worker_local_monotonic',
                                 ordinal=len(self._activities), activity=name,
                                 start_seconds=begin-self.origin, end_seconds=end-self.origin,
                                 duration_seconds=end-begin, status=status)
                    self._activities.append(value)
                self._send(dict(value))
            except BaseException:
                self._drop()

    def snapshot(self):
        with self._lock:
            return dict(clock='worker_local_monotonic', dropped=self.dropped,
                        windows=[dict(row, stages={k:dict(v) for k,v in row['stages'].items()})
                                 for _,row in sorted(self._rows.items())],
                        activities=[dict(x) for x in self._activities])

    def emit_window(self, index):
        try:
            with self._lock:
                if index in self._emitted or index not in self._rows: return
                self._emitted.add(index)
                row = self._rows[index]
                value = dict(row, stages={k:dict(v) for k,v in row['stages'].items()})
            self._send(dict(type='h3_window_substages', clock='worker_local_monotonic',
                            origin_monotonic=self.origin, **value))
        except BaseException:
            self._drop()

    def emit_remaining(self):
        try:
            for row in self.snapshot()['windows']: self.emit_window(row['index'])
            self._send(dict(type='h3_window_profile_end', clock='worker_local_monotonic',
                            windows=len(self._rows), dropped=self.dropped))
        except BaseException:
            self._drop()
