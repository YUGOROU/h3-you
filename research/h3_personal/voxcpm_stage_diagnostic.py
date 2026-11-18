"""Opt-in eager CUDA stage attribution; no model transformations or sync calls.

Event spans include dispatch/device idle inside calls and queue semantics, not
isolated kernel utilization. Caller CPU PCM synchronization remains visible in
the measurement and is neither removed nor attributed to a kernel.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time


_TARGETS = (('base_lm', 'forward'), ('base_lm', 'forward_step'),
            ('residual_lm', 'forward'), ('residual_lm', 'forward_step'),
            ('feat_decoder', 'forward'), ('feat_encoder', 'forward'),
            ('audio_vae', 'encode'), ('audio_vae', 'decode'))


class StageRecorder:
    """Bounded instance-only patches. No event is waited on, even after drain."""

    def __init__(self, torch, device, cap=4096):
        if type(cap) is not int or not 1 <= cap <= 4096:
            raise ValueError('cap must be in 1..4096')
        self.torch, self.device, self.cap = torch, device, cap
        self.stream = torch.cuda.current_stream(device)
        self.records = []
        self.counts = {}
        self.dropped = 0

    def check_stream(self):
        if self.torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError('concurrent/changed CUDA streams unsupported')

    @contextmanager
    def patch(self, model):
        saved = []
        try:
            for owner, method in _TARGETS:
                obj = getattr(model, owner)
                original = getattr(obj, method)
                if not callable(original):
                    raise RuntimeError('missing callable stage: ' + owner + '.' + method)
                had = method in vars(obj)
                previous = vars(obj).get(method)
                label = owner + '.' + method

                def wrapped(*args, _fn=original, _label=label, **kwargs):
                    self.check_stream()
                    self.counts[_label] = self.counts.get(_label, 0) + 1
                    if len(self.records) >= self.cap:
                        self.dropped += 1
                        return _fn(*args, **kwargs)
                    start = self.torch.cuda.Event(enable_timing=True)
                    end = self.torch.cuda.Event(enable_timing=True)
                    row = dict(stage=_label, start=start, end=end,
                               host_seconds=None, ended=False)
                    self.records.append(row)  # reserve before nested calls
                    start.record(self.stream)
                    host_start = time.perf_counter()
                    try:
                        return _fn(*args, **kwargs)
                    finally:
                        row['host_seconds'] = time.perf_counter() - host_start
                        self.check_stream()
                        end.record(self.stream)
                        row['ended'] = True

                saved.append((obj, method, had, previous))
                setattr(obj, method, wrapped)
            yield self
        finally:
            for obj, method, had, previous in reversed(saved):
                if had:
                    setattr(obj, method, previous)
                else:
                    delattr(obj, method)

    def summary(self):
        stages = {name: dict(count=count, recorded_count=0,
                            gpu_known_count=0, gpu_unknown_count=0,
                            gpu_event_seconds=0.0, host_seconds=0.0)
                  for name, count in self.counts.items()}
        rows = []
        for row in self.records:
            seconds = None
            if row['ended']:
                # Query both; never force completion or fetch a tensor scalar.
                first = row['start'].query()
                last = row['end'].query()
                if first and last:
                    seconds = row['start'].elapsed_time(row['end']) / 1000.0
            stage = stages[row['stage']]
            stage['recorded_count'] += 1
            stage['host_seconds'] += row['host_seconds'] or 0.0
            stage['gpu_unknown_count' if seconds is None else 'gpu_known_count'] += 1
            if seconds is not None:
                stage['gpu_event_seconds'] += seconds
            rows.append(dict(stage=row['stage'], host_seconds=row['host_seconds'],
                             gpu_event_seconds=seconds))
        return dict(stages=stages, rows=rows, cap=self.cap,
                    dropped_records=self.dropped,
                    semantics='Inclusive spans; nested stages overlap. GPU sums cover only queried-ready events; unknown is not zero. Event spans include idle/dispatch and queue semantics, not isolated kernel utilization. Caller CPU PCM sync is unchanged.')


def _validate(candidate, torch):
    if candidate.observe_stop is not False or candidate._active or sys.gettrace() is not None:
        raise RuntimeError('idle candidate and trace OFF required')
    model = candidate._model.tts_model
    if not torch.cuda.is_available():
        raise RuntimeError('actual CUDA required')
    if any(m.training or hasattr(m, '_orig_mod') or
           getattr(m, '_compiled_call_impl', None) is not None for m in model.modules()):
        raise RuntimeError('eval-only EAGER model required')
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type != 'cuda':
        raise RuntimeError('single CUDA device model required')
    device = next(iter(devices))
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('CUDA graph capture unsupported')
    # Only the default stream is supported. Arbitrary internal/background stream
    # use cannot be discovered by Python wrappers; it is a caller precondition.
    if torch.cuda.current_stream(device) != torch.cuda.default_stream(device):
        raise RuntimeError('default stream only; concurrent streams unsupported')
    return model, device


@contextmanager
def _compiled_lm(model, torch):
    """Compile only LM steps, restoring exact instance state on every exit."""
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError('suppress_errors must be False')
    saved = []
    try:
        for owner in ('base_lm', 'residual_lm'):
            obj = getattr(model, owner)
            original = obj.forward_step
            saved.append((obj, 'forward_step' in vars(obj),
                          vars(obj).get('forward_step')))
            compiled = torch.compile(original, mode='reduce-overhead', fullgraph=True)
            if getattr(compiled, '_torchdynamo_orig_callable', None) is not original:
                raise RuntimeError('LM compile activation verification failed')
            obj.forward_step = compiled
        yield
    finally:
        for obj, had, previous in reversed(saved):
            if had:
                obj.forward_step = previous
            elif 'forward_step' in vars(obj):
                del obj.forward_step


def run(candidate, text, out, torch, measure, lm_compile=False):
    """One temporary warm drain, one baseline, one instrumented greeting drain.

    Caller must ensure exclusive model/device use and no internal side streams.
    measure must fully exhaust the bounded streaming measurement contract.
    """
    if type(lm_compile) is not bool:
        raise TypeError('lm_compile must be an explicit bool')
    if not isinstance(text, str) or not text.strip() or len(text) > 4096:
        raise ValueError('require nonempty text of at most 4096 characters')
    model, device = _validate(candidate, torch)
    import numpy as np
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(experiment='voxcpm-eager-stage-attribution', status='running',
                  production_adoption=False, native_eos='unknown', cases=[],
                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  seed=42, eager=True, trace=False, weights_unchanged=True,
                  stream_contract='exclusive default stream; caller guarantees no internal/concurrent side streams')

    def call(label, destination):
        np.random.seed(42)
        torch.manual_seed(42)
        start = time.perf_counter()
        with torch.inference_mode():
            result = measure(candidate, text, destination, 'float', timing_attribution=True)
        completion = dict(getattr(candidate, 'completion', None) or {})
        known = (candidate.observe_stop and completion.get('stream_exhausted') is True
                 and completion.get('reason') in ('natural_stop', 'max_len'))
        return dict(case=label, wall_seconds=time.perf_counter()-start, measurement=result,
                    observe_stop=candidate.observe_stop, stop_evidence=completion,
                    native_eos='known' if known else 'unknown')

    recorder = None
    try:
        with tempfile.TemporaryDirectory(prefix='voxcpm-stage-warm-') as temp:
            call('warm', Path(temp)/'warm')
        report['cases'].append(call('baseline', directory/'baseline'))
        _validate(candidate, torch)
        recorder = StageRecorder(torch, device)
        with recorder.patch(model):
            report['cases'].append(call('instrumented', directory/'instrumented'))
        recorder.check_stream()
        if lm_compile:
            report.update(compiled_numerics='unaccepted-candidate',
                          compiled_math='native math, same settings; only LM forward_step compiled',
                          compiled_targets=['base_lm.forward_step', 'residual_lm.forward_step'],
                          eager_untouched=['feat_decoder', 'feat_encoder', 'audio_vae'])
            compile_start = time.perf_counter()
            try:
                with _compiled_lm(model, torch):
                    try:
                        with tempfile.TemporaryDirectory(prefix='voxcpm-lm-warm-') as temp:
                            call('compiled-warm', Path(temp)/'warm')
                    finally:
                        report['compilation_and_warm_seconds'] = time.perf_counter()-compile_start
                    for index in range(2):
                        label = 'compiled-untraced-' + str(index + 1)
                        report['cases'].append(call(label, directory/label))
                    candidate.observe_stop = True
                    report['cases'].append(call('compiled-stop-traced', directory/'compiled-stop-traced'))
            finally:
                candidate.observe_stop = False
                report.setdefault('compilation_and_warm_seconds', time.perf_counter()-compile_start)
        report['status'] = 'completed-experiment-only'
        return report
    except BaseException as exc:
        report.update(status='blocked', blocker=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        candidate.observe_stop = False
        if recorder is not None:
            report['attribution'] = recorder.summary()
        (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
