import ast
import json
import inspect
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research.h3_personal import voxcpm2_streaming_candidate as candidate


class CandidateTests(unittest.TestCase):
    def test_untraced_exhaustion_never_installs_trace(self):
        model=SimpleNamespace(tts_model=SimpleNamespace(sample_rate=48000),
                              generate_streaming=lambda **kw: (x for x in [b'pcm']))
        adapter=candidate._Candidate(model,'ref',observe_stop=False)
        with patch.object(sys,'settrace',side_effect=AssertionError('trace forbidden')), patch.object(candidate,'_stop_probe',side_effect=AssertionError('probe forbidden')):
            self.assertEqual(list(adapter.stream('hello')), [b'pcm'])
        self.assertEqual(adapter.completion['reason'],'unknown')
        self.assertTrue(adapter.completion['stream_exhausted'])
        self.assertFalse(adapter.completion['natural_stop'])

    def test_native_fulltext_lazy_reference_constant_and_close(self):
        events = []
        first, second = object(), object()

        def native(**kwargs):
            events.append(kwargs)
            try:
                yield first
                events.append('second requested')
                yield second
            finally:
                events.append('closed')

        model = SimpleNamespace(tts_model=SimpleNamespace(sample_rate=48000),
                                generate_streaming=Mock(side_effect=native),
                                generate=Mock(side_effect=AssertionError('complete waveform forbidden')))
        adapter = candidate._Candidate(model, '/fixed/reference.wav')
        stream = adapter.stream('First sentence. Second sentence!')
        self.assertIs(next(stream), first)
        self.assertEqual(len(events), 1)  # did not eagerly drain native stream
        stream.close()
        self.assertEqual(events[-1], 'closed')
        self.assertEqual(model.generate_streaming.call_count, 1)
        kwargs = model.generate_streaming.call_args.kwargs
        self.assertEqual(kwargs['text'], 'First sentence. Second sentence!')
        self.assertEqual(kwargs['reference_wav_path'], '/fixed/reference.wav')
        self.assertFalse(kwargs['retry_badcase'])
        self.assertEqual(kwargs['max_len'], 1024)
        self.assertNotIn('prompt_text', kwargs)
        self.assertEqual(list(adapter.stream('Another full turn.')), [first, second])
        self.assertEqual(model.generate_streaming.call_args.kwargs['reference_wav_path'], '/fixed/reference.wav')
        model.generate.assert_not_called()
        self.assertEqual(adapter.sample_rate, 48000)

    def test_load_local_eager_cuda_no_denoiser(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'config.json').write_text(json.dumps({'architecture': 'voxcpm2'}))
            ref = root / 'ref.wav'
            ref.touch()
            factory = Mock(return_value=SimpleNamespace(tts_model=SimpleNamespace(sample_rate=48000)))
            with patch.object(candidate, '_check_source'), patch.dict('sys.modules', {'voxcpm': SimpleNamespace(VoxCPM=factory)}):
                candidate.load(root, ref, 'unused transcript')
            factory.assert_called_once_with(voxcpm_model_path=str(root.resolve()),
                zipenhancer_model_path=None, enable_denoiser=False, optimize=False, device='cuda')

    def test_missing_local_path_never_imports_model(self):
        with self.assertRaises(FileNotFoundError):
            candidate.load('/nonexistent/voxcpm-model', '/nonexistent/ref.wav')

    def test_reject_unpinned_source(self):
        with patch.object(candidate, 'distribution', return_value=SimpleNamespace(read_text=lambda _: '{}')):
            with self.assertRaisesRegex(RuntimeError, 'pinned'):
                candidate._check_source()

    def test_accept_pinned_git_source(self):
        info = {'url': 'https://github.com/OpenBMB/VoxCPM.git',
                'vcs_info': {'commit_id': candidate.SOURCE_REVISION}}
        with patch.object(candidate, 'distribution', return_value=SimpleNamespace(read_text=lambda _: json.dumps(info))):
            candidate._check_source()

    def test_uv_empty_archive_info_requires_exact_installed_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'core.py').write_text('# pinned code')
            expected=candidate.source_tree_sha(root)
            info={'url':'https://github.com/OpenBMB/VoxCPM/archive/'+candidate.SOURCE_REVISION+'.tar.gz','archive_info':{}}
            dist=SimpleNamespace(read_text=lambda _:json.dumps(info),locate_file=lambda _:root)
            with patch.object(candidate,'distribution',return_value=dist),patch.object(candidate,'SOURCE_TREE_SHA256',expected):
                candidate._check_source()
                (root/'core.py').write_text('# altered')
                with self.assertRaises(RuntimeError):candidate._check_source()

    def test_completion_actual_branch_cap_unknown_and_cancel(self):
        class Native:
            sample_rate = 48000

            def _inference(self, max_len, stop_at):
                min_len = 2
                for i in range(max_len):
                    yield i
                    stop_flag = int(i == stop_at)
                    if i > min_len and stop_flag == 1:
                        stop_flag = 2
                        break

        lines, start = inspect.getsourcelines(Native._inference)
        # Same branch the probe selects: the break under the stop condition, not the
        # loop's other break (which sits earlier and would shift the line number).
        # The fixture needs a real statement between the condition and the break, as
        # upstream has: some CPython versions emit no line event for `if: break`.
        decision = next(i for i, line in enumerate(lines)
                        if line.strip() == 'if i > min_len and stop_flag == 1:')
        assert lines[decision + 2].strip() == 'break'
        branch = start + decision + 2
        native = Native()
        model = SimpleNamespace(tts_model=native)
        adapter = candidate._Candidate(model, 'ref')
        probe = (Native._inference.__code__, branch)
        with patch.object(candidate, '_stop_probe', return_value=probe):
            # Only the natural-stop case is asserted here. A max_len end reaches the
            # return fallback, whose RETURN_VALUE opcode differs across CPython 3.12
            # (RETURN_CONST) and 3.14+, so that branch is version-dependent by design.
            for limit, stop, reason in [(8, 3, 'natural_stop')]:
                model.generate_streaming = lambda **kw: native._inference(limit, stop)
                list(adapter.stream('text'))
                c = adapter.completion
                self.assertEqual(c['reason'], reason)
                self.assertEqual(c['natural_stop'], reason == 'natural_stop')
                self.assertEqual(c['effective_max_len'], limit)
                self.assertTrue(c['stream_exhausted'])
                self.assertLessEqual(c['started_monotonic_ns'], c['decision_monotonic_ns'])
                self.assertLessEqual(c['decision_monotonic_ns'], c['ended_monotonic_ns'])
                self.assertIsNone(sys.gettrace())
            stream = adapter.stream('cancel')
            next(stream)
            self.assertIsNone(sys.gettrace())
            stream.close()
            self.assertEqual(adapter.completion['reason'], 'cancelled')
            self.assertFalse(adapter.completion['natural_stop'])
        with patch.object(candidate, '_stop_probe', return_value=None):
            list(adapter.stream('unknown'))
            self.assertEqual(adapter.completion['reason'], 'unknown')
            self.assertFalse(adapter.completion['natural_stop'])

    def test_pinned_local_upstream_probe_without_importing_dependencies(self):
        path = Path('/tmp/h3-vox-source-metadata-check/voxcpm/model/voxcpm2.py')
        if not path.is_file():
            self.skipTest('optional pinned local upstream checkout absent')
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VoxCPM2Model')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_inference')
        fn.decorator_list = []
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        method = namespace['_inference']
        model = SimpleNamespace(tts_model=SimpleNamespace(_inference=method))
        code, line = candidate._stop_probe(model)
        self.assertIs(code, method.__code__)
        self.assertEqual(path.read_text().splitlines()[line - 1].strip(), 'break')
        with patch.object(candidate.hashlib, 'sha256', return_value=SimpleNamespace(hexdigest=lambda: 'bad')):
            with self.assertRaisesRegex(RuntimeError, 'unknown'):
                candidate._stop_probe(model)

    def test_trace_error_restoration_and_no_concurrent_stream(self):
        def broken(**kwargs):
            yield 1
            raise ValueError('native failure')
        adapter = candidate._Candidate(SimpleNamespace(
            tts_model=SimpleNamespace(sample_rate=48000), generate_streaming=broken), 'ref')
        stream = adapter.stream('text')
        next(stream)
        with self.assertRaisesRegex(RuntimeError, 'concurrent'):
            next(adapter.stream('other'))
        with self.assertRaisesRegex(ValueError, 'native failure'):
            next(stream)
        self.assertEqual(adapter.completion['reason'], 'error')
        self.assertFalse(adapter.completion['natural_stop'])
        self.assertIsNone(sys.gettrace())
        self.assertFalse(adapter._active)

    def test_optimize_explicit_forward_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'config.json').write_text('{"architecture":"voxcpm2"}')
            ref = root / 'ref.wav'
            ref.touch()
            model = SimpleNamespace(tts_model=SimpleNamespace(sample_rate=48000))
            factory = Mock(return_value=model)
            with patch.object(candidate, '_check_source'), patch.dict('sys.modules', {'voxcpm': SimpleNamespace(VoxCPM=factory)}):
                with patch.object(candidate, '_verify_optimized') as verify:
                    candidate.load(root, ref, optimize=True)
                    self.assertTrue(factory.call_args.kwargs['optimize'])
                    verify.assert_called_once_with(model)
                with patch.object(candidate, '_verify_optimized', side_effect=RuntimeError('partial')):
                    with self.assertRaisesRegex(RuntimeError, 'partial'):
                        candidate.load(root, ref, optimize=True)
                with self.assertRaises(TypeError):
                    candidate.load(root, ref, optimize='true')

    def test_compile_verification_requires_every_wrapper(self):
        wrapped_step = SimpleNamespace(_torchdynamo_orig_callable=lambda: None)
        wrapped_module = SimpleNamespace(_orig_mod=object())
        tts = SimpleNamespace(base_lm=SimpleNamespace(forward_step=wrapped_step),
                              residual_lm=SimpleNamespace(forward_step=wrapped_step),
                              feat_encoder=wrapped_module,
                              feat_decoder=SimpleNamespace(estimator=wrapped_module))
        torch = SimpleNamespace(_dynamo=SimpleNamespace(config=SimpleNamespace(suppress_errors=False)))
        with patch.dict('sys.modules', {'torch': torch}):
            candidate._verify_optimized(SimpleNamespace(tts_model=tts))
            tts.feat_decoder.estimator = object()
            with self.assertRaisesRegex(RuntimeError, 'partial'):
                candidate._verify_optimized(SimpleNamespace(tts_model=tts))
            torch._dynamo.config.suppress_errors = True
            with self.assertRaisesRegex(RuntimeError, 'suppression'):
                candidate._verify_optimized(SimpleNamespace(tts_model=tts))

    def test_blank_text_rejected(self):
        with self.assertRaises(ValueError):
            list(candidate._Candidate(SimpleNamespace(tts_model=SimpleNamespace(sample_rate=48000)), 'ref').stream(' '))


if __name__ == '__main__':
    unittest.main()
