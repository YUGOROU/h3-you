"""Dependency-free stage recorder tests: no CUDA, numpy, or torch import."""
import types
import unittest
import tempfile
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch

from research.h3_personal import voxcpm_stage_diagnostic as diagnostic

from research.h3_personal.voxcpm_stage_diagnostic import StageRecorder, _TARGETS


class Event:
    ready = True

    def __init__(self, enable_timing):
        assert enable_timing
        self.queries = 0

    def record(self, stream):
        self.stream = stream

    def query(self):
        self.queries += 1
        return self.ready

    def elapsed_time(self, end):
        assert self.queries and end.queries
        return 5.0


class Module:
    def forward(self, *args):
        return args

    forward_step = forward
    encode = forward
    decode = forward


def fixture(cap=4096):
    cuda = types.SimpleNamespace(Event=Event, current_stream=lambda device: 'stream')
    recorder = StageRecorder(types.SimpleNamespace(cuda=cuda), 'cuda', cap)
    model = types.SimpleNamespace(**{owner: Module() for owner, _ in _TARGETS})
    return recorder, model


class StageTests(unittest.TestCase):
    def test_restore_inherited_and_instance(self):
        recorder, model = fixture()
        override = lambda: 7
        model.base_lm.forward = override
        with recorder.patch(model):
            self.assertEqual(model.base_lm.forward(), 7)
            self.assertEqual(model.audio_vae.decode(2), (2,))
        self.assertIs(model.base_lm.forward, override)
        self.assertNotIn('decode', vars(model.audio_vae))
        summary = recorder.summary()
        self.assertEqual(summary['stages']['base_lm.forward']['gpu_event_seconds'], .005)

    def test_cap_and_unknown_query(self):
        recorder, model = fixture(2)
        with recorder.patch(model):
            for _ in range(5):
                model.base_lm.forward()
        recorder.records[0]['start'].ready = False
        recorder.records[1]['end'].ready = False
        summary = recorder.summary()
        self.assertEqual(len(summary['rows']), 2)
        self.assertEqual(summary['dropped_records'], 3)
        self.assertEqual(summary['stages']['base_lm.forward']['count'], 5)
        self.assertTrue(all(r['gpu_event_seconds'] is None for r in summary['rows']))
        self.assertTrue(all(r['end'].queries == 1 for r in recorder.records))

    def test_exception_restores_all(self):
        recorder, model = fixture()

        def fail():
            raise ValueError('boom')

        model.feat_decoder.forward = fail
        before = {name: dict(vars(getattr(model, name))) for name, _ in _TARGETS}
        with self.assertRaisesRegex(ValueError, 'boom'):
            with recorder.patch(model):
                model.feat_decoder.forward()
        for name, _ in _TARGETS:
            self.assertEqual(vars(getattr(model, name)), before[name])
        self.assertTrue(recorder.records[0]['ended'])

    def test_partial_patch_failure_restores(self):
        recorder, model = fixture()
        del model.audio_vae
        with self.assertRaises(AttributeError):
            with recorder.patch(model):
                pass
        self.assertEqual(vars(model.base_lm), {})

    def test_nested_calls_reserve_cap(self):
        recorder, model = fixture(1)
        model.base_lm.forward = lambda: model.residual_lm.forward()
        with recorder.patch(model):
            model.base_lm.forward()
        self.assertEqual(len(recorder.records), 1)
        self.assertEqual(recorder.dropped, 1)

    def test_changed_stream_rejected_and_restored(self):
        recorder, model = fixture()
        with self.assertRaises(RuntimeError):
            with recorder.patch(model):
                recorder.torch.cuda.current_stream = lambda device: 'other'
                model.base_lm.forward()
        self.assertEqual(vars(model.base_lm), {})

    def test_invalid_cap(self):
        for cap in (0, 4097, True):
            with self.assertRaises(ValueError):
                fixture(cap)


class CompileTests(unittest.TestCase):
    def setup_run(self, failure=None):
        recorder, model = fixture()
        torch = recorder.torch
        torch._dynamo = types.SimpleNamespace(config=types.SimpleNamespace(suppress_errors=False))
        torch.manual_seed = lambda seed: self.seeds.append(seed)
        torch.inference_mode = nullcontext
        self.seeds, self.compiles, self.calls = [], [], []
        before = {name: dict(vars(getattr(model, name))) for name, _ in _TARGETS}

        def compile(fn, **settings):
            self.compiles.append((fn, settings))
            if len(self.compiles) == 1:
                self.assertEqual(before, {name: dict(vars(getattr(model, name)))
                                          for name, _ in _TARGETS})
            if failure == 'compile' and len(self.compiles) == 2:
                raise RuntimeError('compile failed')
            def compiled(*args, **kwargs):
                return fn(*args, **kwargs)
            compiled._torchdynamo_orig_callable = fn
            return compiled

        torch.compile = compile
        candidate = types.SimpleNamespace(observe_stop=False, completion=None)

        def measure(candidate, text, destination, *args, **kwargs):
            self.calls.append((str(destination), candidate.observe_stop))
            model.base_lm.forward_step()
            if failure == 'measure' and 'compiled-stop' in str(destination):
                raise RuntimeError('measure failed')
            candidate.completion = dict(reason='natural_stop', stream_exhausted=True)
            return {}
        return model, torch, candidate, measure, before

    def exercise(self, failure=None):
        model, torch, candidate, measure, before = self.setup_run(failure)
        numpy = types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda seed: None))
        with tempfile.TemporaryDirectory() as temp, patch.dict('sys.modules', numpy=numpy), patch.object(diagnostic, '_validate', return_value=(model, 'cuda')):
            if failure:
                with self.assertRaisesRegex(RuntimeError, failure + ' failed'):
                    diagnostic.run(candidate, 'hello', Path(temp)/'out', torch, measure, lm_compile=True)
                report = None
            else:
                report = diagnostic.run(candidate, 'hello', Path(temp)/'out', torch, measure, lm_compile=True)
        self.assertFalse(candidate.observe_stop)
        for name, _ in _TARGETS:
            self.assertEqual(vars(getattr(model, name)), before[name])
        for fn, settings in self.compiles:
            self.assertEqual(settings, dict(mode='reduce-overhead', fullgraph=True))
            self.assertIn(fn.__self__, (model.base_lm, model.residual_lm))
        return report

    def test_compile_run(self):
        report = self.exercise()
        self.assertEqual(len(self.calls), 7)
        self.assertEqual(self.seeds, [42]*7)
        self.assertEqual([c['observe_stop'] for c in report['cases']], [False]*4 + [True])
        self.assertEqual([c['native_eos'] for c in report['cases']], ['unknown']*4 + ['known'])
        self.assertGreaterEqual(report['compilation_and_warm_seconds'], 0)

    def test_partial_compile_failure(self):
        self.exercise('compile')

    def test_measure_failure(self):
        self.exercise('measure')

    def test_suppression_and_missing_activation(self):
        model, torch, _, _, before = self.setup_run()
        torch._dynamo.config.suppress_errors = True
        with self.assertRaisesRegex(RuntimeError, 'suppress_errors'):
            with diagnostic._compiled_lm(model, torch):
                pass
        torch._dynamo.config.suppress_errors = False
        torch.compile = lambda fn, **kw: lambda: None
        with self.assertRaisesRegex(RuntimeError, 'verification'):
            with diagnostic._compiled_lm(model, torch):
                pass
        for name, _ in _TARGETS:
            self.assertEqual(vars(getattr(model, name)), before[name])


if __name__ == '__main__':
    unittest.main()
