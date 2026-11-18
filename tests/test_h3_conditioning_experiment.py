"""CPU-only isolated experiment contracts."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import sys

from research.h3_personal import conditioning_experiment as exp
from research.h3_personal.fast_ref2va import condition_key, store_condition, get_condition_cache
from research.h3_personal.fast_continuation import WindowInput, SamplerContract
from research.h3_personal.continuation_plan import plan


def actual_token(index, session='test-session'):
    window = plan(328, 124, 22)[index]
    return WindowInput(session, index, 1, window, window.temporal_offset,
                       0, 0, None, None, None, SamplerContract('SMD4', 4))


class Tensor:
    shape = (491,)
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return Tensor()


class ConditioningExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.policy = Path(self.tmp.name, 'policy.json')
        self.manifest = dict(schema=exp.SCHEMA, short_edges=[1024, 512], active_short_edge=1024)
        self.policy.write_text(json.dumps(self.manifest))
        self.env = patch.dict(os.environ, {exp.EXPERIMENT_ENV: str(self.policy)})
        self.env.start(); self.addCleanup(self.env.stop)
        self.source = Path(self.tmp.name, 'image'); self.source.write_bytes(b'raw')

    def batch(self):
        return NS(prompt='private', height=448, width=256, num_frames=124, fps=24,
                  references=[NS(source=self.source, media_type='image')], extra={}, prompt_embeds=[Tensor()])

    def test_strict_and_off(self):
        self.assertEqual(exp.read_policy(), self.manifest)
        for change in ({'active_short_edge': True}, {'extra': 1}, {'short_edges': [512, 1024]}):
            with self.assertRaises(ValueError): exp.validate_manifest(dict(self.manifest, **change))
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(exp.read_policy())
            self.assertEqual(exp.reference_short_edge(), 2048)

    def test_precompute_two_cpu_entries_original_unchanged(self):
        batch = self.batch(); exp.snapshot_sources(batch)
        original_key = condition_key(batch)
        batch.references = [NS(media_type='image', image=NS(size=(1024, 1824)))]
        batch.extra['minimax_h3_text_token_tags'] = Tensor()
        config = NS(to_dict=lambda: {'merge_size': 2, 'size': {'longest_edge': 123}})
        stage = NS(processor=NS(image_processor=config))
        calls = []
        def prepare(ref, frames, rate):
            self.assertEqual(ref.source, self.source)
            self.assertEqual(rate, 32000)
            edge = exp.reference_short_edge()
            return NS(media_type='image', image=NS(size=(edge, edge * 2)))
        def forward(clone, args):
            calls.append(exp.reference_short_edge())
            clone.prompt_embeds = [Tensor()]
            clone.extra['minimax_h3_text_token_tags'] = Tensor()
            store_condition(stage, clone, args)  # recursion must not encode again
            return clone
        stage.forward = forward
        module = NS(prepare_reference=prepare)
        with patch.object(exp, 'prepare_reference', prepare):
            store_condition(stage, batch, NS())
        self.assertEqual(calls, [512])
        self.assertEqual(len(stage._h3_conditioning_cache), 2)
        self.assertIsNotNone(get_condition_cache(stage, original_key))
        self.assertEqual(batch.references[0].image.size, (1024, 1824))
        self.manifest['active_short_edge'] = 512
        self.policy.write_text(json.dumps(self.manifest))
        self.assertIsNotNone(get_condition_cache(stage, condition_key(batch)))
        batch.prompt = 'different'
        self.assertIsNone(get_condition_cache(stage, condition_key(batch)))
        receipt = exp.export_receipt(stage, batch)
        self.assertNotIn('private', json.dumps(receipt))
        self.assertEqual(receipt['precomputed_edges'], [1024, 512])
        self.assertEqual(receipt['resize_policy'], 'shortedge-lanczos-grid32-v1')
        self.assertEqual(exp.policy_key(), exp.digest(dict(self.manifest, resize_policy=exp.RESIZE_POLICY)))

    def test_native_prepare_source_free_constructor_and_warm_keys(self):
        native = Path(__file__).parent / 'fixtures/minimax_h3_reference.py'
        target = Path(self.tmp.name) / 'fastvideo/pipelines/basic/minimax_h3/reference.py'
        target.parent.mkdir(parents=True)
        target.write_bytes(native.read_bytes())
        tree = ast.parse(target.read_text())
        names = {'resolve_reference_image_size', 'prepare_reference_image', 'prepare_reference'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        calls = []
        class DummyImage:
            def __init__(self, size=(3088, 1737), mode='L'):
                self.size, self.mode = size, mode
            def convert(self, mode):
                calls.append(('convert', mode))
                return DummyImage(self.size, mode)
            def resize(self, size, resample):
                calls.append(('resize', size, resample))
                return DummyImage(size, self.mode)
        def transpose(image):
            calls.append(('exif', image.size))
            return DummyImage(tuple(reversed(image.size)), image.mode)
        namespace = dict(__name__='fastvideo.pipelines.basic.minimax_h3.reference', __file__=str(target), os=os, MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE=2048,
                         MINIMAX_H3_CANVAS_MULTIPLE=16,
                         Image=NS(Image=DummyImage, Resampling=NS(LANCZOS='native-lanczos')),
                         ImageOps=NS(exif_transpose=transpose), load_image=lambda path: DummyImage(),
                         MiniMaxH3PreparedReference=lambda media_type: NS(media_type=media_type))
        # Postponed annotations keep the real native bodies dependency-free on CPU.
        import __future__
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(target), 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)
        original = namespace['prepare_reference']
        original_globals = dict(original.__globals__)
        original_bytes = target.read_bytes()
        def prepare(ref, frames, rate):
            return exp.prepare_reference(ref, frames, rate, native_prepare=original)
        batch = self.batch()
        key = condition_key(batch)  # source snapshot before native input preparation
        batch.references = [prepare(batch.references[0], 124, 32000)]
        exp.capture_prepared_geometry(batch)
        self.assertEqual(batch.extra['_h3_conditioning_prepared_geometry'][0][2], ((1024, 1824),))
        self.assertFalse(hasattr(batch.references[0], 'source'))
        self.assertEqual(condition_key(batch), key)
        batch.extra['minimax_h3_text_token_tags'] = Tensor()
        stage = NS(processor=NS(image_processor=NS(to_dict=lambda: {'native': True})))
        def forward(clone, args):
            clone.prompt_embeds = [Tensor()]
            clone.extra['minimax_h3_text_token_tags'] = Tensor()
            return clone
        stage.forward = forward
        with patch.dict(sys.modules, {'fastvideo.pipelines.basic.minimax_h3.reference': NS(prepare_reference=original)}):
            store_condition(stage, batch)
        self.assertEqual(stage._h3_conditioning_sizes, {1024: [[1024, 1824]], 512: [[512, 896]]})
        self.assertEqual(len(stage._h3_conditioning_cache), 2)
        for edge in (1024, 512):
            self.manifest['active_short_edge'] = edge
            self.policy.write_text(json.dumps(self.manifest))
            fresh = self.batch()
            before = condition_key(fresh)
            fresh.references = [prepare(fresh.references[0], 124, 32000)]
            self.assertEqual(condition_key(fresh), before)
            self.assertIsNotNone(get_condition_cache(stage, before))
            self.assertEqual(fresh.references[0].image.mode, 'RGB')
        with patch.dict(os.environ, {}, clear=True):
            released = prepare(self.batch().references[0], 124, 32000)
            self.assertEqual(released.image.size, (2048, 3648))
        self.assertIn(('resize', (512, 896), 'native-lanczos'), calls)
        self.assertIn(('exif', (3088, 1737)), calls)
        self.assertIn(('convert', 'RGB'), calls)
        self.assertEqual(original.__globals__, original_globals)
        self.assertEqual(target.read_bytes(), original_bytes)
        self.assertNotIn(('resize', (2048, 8192), 'native-lanczos'), calls)

    def test_raw_source_drift_and_policy_cache_isolation(self):
        from research.h3_personal.ref2va_runtime_options import prepare_images_cached
        batch = self.batch(); condition_key(batch)
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'source bytes drifted'):
            condition_key(batch)
        stage = NS(); calls = []
        def prepare(ref, frames, rate):
            calls.append(exp.reference_short_edge())
            return NS(media_type='image', image=NS(size=(exp.reference_short_edge(), 1)))
        with patch.object(exp, 'prepare_reference', lambda r, n, s, native_prepare: native_prepare(r, n, s)):
            prepare_images_cached(stage, batch.references, 124, 24000, prepare)
        with patch.object(exp, 'prepare_reference', lambda r, n, s, native_prepare: native_prepare(r, n, s)):
            prepare_images_cached(stage, batch.references, 124, 24000, prepare)
        self.manifest['active_short_edge'] = 512
        self.policy.write_text(json.dumps(self.manifest))
        with patch.object(exp, 'prepare_reference', lambda r, n, s, native_prepare: native_prepare(r, n, s)):
            prepare_images_cached(stage, batch.references, 124, 24000, prepare)
        self.assertEqual(calls, [1024, 512])

    def test_private_rank_receipt_measured_layout(self):
        batch = self.batch()
        exp.snapshot_sources(batch)
        self.manifest['active_short_edge'] = 512
        self.policy.write_text(json.dumps(self.manifest))
        batch.references = [NS(media_type='image', image=NS(size=(512, 896)))]
        exp.capture_prepared_geometry(batch)
        batch.references = None  # actual native latent-preparation lifecycle
        batch.extra.update(minimax_h3_layout=NS(num_condition_video_rows=448 + 7 * 112,
                                               text_indices=NS(numel=lambda: 491)),
                           minimax_h3_text_token_tags=Tensor(), _h3_conditioning_cache_hit=True)
        stage = NS(_h3_conditioning_sizes={1024: [], 512: []}, _h3_conditioning_processor_digest='digest')
        parameter = object()
        pipeline = NS(_stage_name_mapping={'conditioning_stage': stage}, _denoise_stages_ready=True,
                      get_module=lambda name: NS(parameters=lambda: iter([parameter])))
        batch.extra['_h3_continuation'] = dict(owner=NS(session_id='test-session'),
            token=actual_token(2))
        window = Path(self.tmp.name, 'window.json')
        window.write_text(json.dumps(dict(mode='latent-tail-v1', session_id='test-session', index=2)))
        with patch.dict(sys.modules, {'torch': NS(distributed=NS(get_rank=lambda: 1))}), patch.dict(os.environ, {'H3_REF2VA_COVERAGE_DIR': str(Path(self.tmp.name).resolve()), 'H3_SMD_CONTINUATION_WINDOW_FILE': str(window)}):
            receipt = exp.write_evidence(pipeline, batch)
            self.assertTrue(receipt['parameter_identity_preserved'])
            self.assertEqual(receipt['experimental_cache_hits'], 1)
            self.assertEqual(receipt['session_id'], 'test-session')
            self.assertEqual(receipt['window_index'], 2)
            immutable = Path(self.tmp.name, 'conditioning-experiment-test-session-window-2-rank-1.json')
            self.assertEqual(json.loads(immutable.read_text()), receipt)
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                exp.write_evidence(pipeline, batch)
            batch.extra['_h3_conditioning_cache_hit'] = False
            with self.assertRaisesRegex(RuntimeError, 'pre-DiT'):
                exp.write_evidence(pipeline, batch)
        self.assertEqual((receipt['R'], receipt['T']), (448, 491))
        target = Path(self.tmp.name, 'conditioning-experiment-rank-1.json')
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(target.read_text()), receipt)

    def test_cache_context_override_and_processor_drift(self):
        batch = self.batch()
        key1024 = condition_key(batch)
        token = exp._edge.set(512)
        try:
            key512 = condition_key(batch)
        finally:
            exp._edge.reset(token)
        self.assertNotEqual(key1024, key512)
        config = {'size': 1024}
        stage = NS(processor=NS(image_processor=NS(to_dict=lambda: dict(config))))
        stage._h3_conditioning_processor_digest = exp.processor_digest(stage)
        stage._h3_conditioning_cache = {key1024: ('large',), key512: ('small',)}
        self.assertEqual(get_condition_cache(stage, key1024), ('large',))
        self.assertEqual(get_condition_cache(stage, key512), ('small',))
        config['size'] = 512
        for key in (key1024, key512):
            with self.assertRaisesRegex(RuntimeError, 'processor identity drift'):
                get_condition_cache(stage, key)

    def test_native_latent_clear_geometry_binding(self):
        batch = self.batch()
        exp.snapshot_sources(batch)
        batch.references = [NS(media_type='image', image=NS(size=(1024, 1824)))]
        exp.capture_prepared_geometry(batch)
        snapshot = batch.extra['_h3_conditioning_prepared_geometry']
        batch.references = None  # pinned native latent preparation consumes references
        batch.extra.update(minimax_h3_layout=NS(num_condition_video_rows=1824 + 7 * 112,
                           text_indices=NS(numel=lambda: 1867)),
                           _h3_conditioning_cache_hit=True,
                           _h3_continuation=dict(token=actual_token(1)))
        stage = NS(_h3_conditioning_sizes={1024: [], 512: []},
                   _h3_conditioning_processor_digest='digest')
        parameter = object()
        pipeline = NS(_stage_name_mapping={'conditioning_stage': stage},
                      _denoise_stages_ready=True,
                      get_module=lambda name: NS(parameters=lambda: iter([parameter])))
        with patch.dict(os.environ, {'H3_REF2VA_COVERAGE_DIR': str(Path(self.tmp.name).resolve()),
                                     'H3_SMD_CONTINUATION_WINDOW_FILE': ''}):
            for rank in (0, 1):
                with patch.dict(sys.modules, {'torch': NS(distributed=NS(get_rank=lambda: rank))}):
                    report = exp.write_evidence(pipeline, batch)
                    self.assertEqual(report['prepared_sizes'], [[1024, 1824]])
                    self.assertEqual((report['R'], report['T']), (1824, 1867))
        binding, checksum = snapshot
        for corrupt in ((binding[0], binding[1], ((512, 896),)),
                        ('wrong-policy', binding[1], binding[2]),
                        (binding[0], ('wrong-source',), binding[2])):
            batch.extra['_h3_conditioning_prepared_geometry'] = (corrupt, checksum)
            with self.assertRaisesRegex(RuntimeError, 'binding drift'):
                exp.export_receipt(batch=batch)
        batch.extra['_h3_conditioning_prepared_geometry'] = snapshot
        self.source.write_bytes(b'drift')
        with self.assertRaisesRegex(RuntimeError, 'source binding drift'):
            exp.export_receipt(batch=batch)
        with patch.dict(os.environ, {}, clear=True):
            off = self.batch()
            exp.capture_prepared_geometry(off)
            self.assertEqual(off.extra, {})
        from research.h3_personal import fast_ref2va
        import inspect
        source = inspect.getsource(fast_ref2va._warm_forward)
        preparation = "batch=pipeline._stage_name_mapping['input_preparation_stage'](batch,args)"
        self.assertIn(preparation + '\n    capture_prepared_geometry(batch)\n', source)
        self.assertLess(source.index('capture_prepared_geometry(batch)'),
                        source.index("for name in ('latent_preparation_stage'"))

    def test_atomic_publish_symlink_guard(self):
        folder = Path(self.tmp.name).resolve()
        victim = folder / 'victim'
        victim.write_text('untouched')
        link = folder / 'receipt.json'
        link.symlink_to(victim)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            exp._publish_private(link, {'safe': True})
        self.assertEqual(victim.read_text(), 'untouched')
        link.unlink()
        exp._publish_private(link, {'safe': True})
        self.assertEqual(link.stat().st_mode & 0o777, 0o600)
        # Repeated windows in the same PID replace the mutable rank receipt.
        for window in range(3):
            exp._publish_private(link, {'window': window})
            self.assertEqual(json.loads(link.read_text()), {'window': window})
            self.assertEqual(link.stat().st_mode & 0o777, 0o600)
        self.assertFalse(list(folder.glob('.conditioning-*.tmp')))

    def test_warm_evidence_before_denoise_source(self):
        source = Path('research/h3_personal/fast_ref2va.py').read_text()
        self.assertIn("experimental warm conditioning cache miss; reload forbidden", source)
        self.assertLess(source.index('write_evidence(pipeline, batch)'), source.index('with diagnostic.stage(name), label:'))


if __name__ == '__main__': unittest.main()
