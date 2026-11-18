import tempfile
from pathlib import Path
import unittest
from research.h3_personal.ref2va_helpers import WarmStageProfile,review_groups,aspect_candidate

class RefHelperTests(unittest.TestCase):
    def test_review_separates_late_drift_and_clip_boundaries(self):
        groups=review_groups()
        self.assertEqual(groups['clip-3'],[372,403,434,465,495])
        self.assertEqual(groups['boundary-1'],[122,123,124,125])
        self.assertTrue(all(0<=n<496 for group in groups.values() for n in group))

    def test_aspect_candidate_preserves_pixels_without_claiming_speed_or_identity(self):
        result=aspect_candidate(1737,3088,448,256)
        self.assertAlmostEqual(result['current_aspect_mismatch'],28/9)
        self.assertAlmostEqual(result['swapped_aspect_mismatch'],64/63)
        self.assertEqual((result['candidate_width'],result['candidate_height']),(256,448))
        self.assertFalse(result['automatic_application']);self.assertFalse(result['speedup_demonstrated'])
        with self.assertRaises(ValueError):aspect_candidate(0,3088,448,256)
        with self.assertRaises(ValueError):aspect_candidate(1737,3088,447,256)

    def test_compile_is_explicit_and_leaves_vae_and_default_eager(self):
        from types import SimpleNamespace as NS
        from research.h3_personal.fast_ref2va import generator_config
        names=('GeneratorConfig','PipelineSelection','ComponentConfig','EngineConfig','ParallelismConfig','OffloadConfig','CompileConfig')
        api=NS(**{name:NS for name in names})
        base=generator_config(api,'base','adapter')
        compiled=generator_config(api,'base','adapter',regional_compile=True)
        self.assertFalse(base.engine.compile.enabled)
        self.assertTrue(compiled.engine.compile.enabled)
        self.assertIsNone(compiled.engine.compile.mode)
        self.assertTrue(compiled.engine.compile.fullgraph)
        self.assertEqual(compiled.engine.compile.extras,{'options':{'triton.cudagraphs':True}})
        self.assertFalse(compiled.engine.compile.vae_enabled)
        self.assertFalse(compiled.engine.compile.audio_vae_enabled)
        self.assertEqual(compiled.pipeline.components.lora_strength,1.0)

    def test_regional_proof_matches_pinned_forward_wrapper_not_module_type(self):
        import ast,os,tarfile
        from types import SimpleNamespace as NS
        from research.h3_personal.fast_ref2va import regional_compile_receipt
        source=Path('/tmp/h3-ref2va-fastvideo-audit/FastVideo-ba371e33ed080e51f1884ae166baf7c4447a66fe/fastvideo/pipelines/composed_pipeline_base.py')
        if not source.exists():self.skipTest('optional pinned source fixture')
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ComposedPipelineBase')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_compile_with_conditions')
        method.decorator_list=[]
        def compile_fn(original,**kwargs):
            def wrapped(*args,**kw):return original(*args,**kw)
            wrapped._torchdynamo_orig_callable=original
            return wrapped
        namespace={'torch':NS(compile=compile_fn)}
        exec('from __future__ import annotations\n'+ast.unparse(method),namespace)
        rows=[(f'transformer_blocks.{i}',NS(forward=lambda:None)) for i in range(50)]
        rows += [(f'token_refiner.refiner_blocks.{i}',NS(forward=lambda:None)) for i in range(2)]
        model=NS(named_modules=lambda:rows,_compile_conditions=[lambda name,module:True])
        self.assertEqual(namespace['_compile_with_conditions'](model,{}),52)
        self.assertEqual(regional_compile_receipt(model)['wrapped_regions'],52)
        rows[0][1].forward=lambda:None
        with self.assertRaises(RuntimeError):regional_compile_receipt(model)
        loader=source.parents[1]/'models/loader/fsdp_load.py'
        tree=ast.parse(loader.read_text())
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_compile_model_regions')
        namespace.update(logger=NS(info=lambda *args:None))
        exec('from __future__ import annotations\n'+ast.unparse(method),namespace)
        for kwargs in ({'fullgraph':False},{'mode':'reduce-overhead'}):
            with self.assertRaises(ValueError):namespace['_compile_model_regions'](model,kwargs)
        self.assertEqual(namespace['_compile_model_regions'](model,{'fullgraph':True,'dynamic':False,'options':{'triton.cudagraphs':True}}),52)
        self.assertEqual(regional_compile_receipt(model)['wrapped_regions'],52)

    def test_static_merge_guard_rejects_dynamic_apply(self):
        from types import SimpleNamespace
        from research.h3_personal.fast_ref2va import static_merge_receipt
        layers={str(i):SimpleNamespace(disable_lora=False,merged=True) for i in range(312)}
        group=SimpleNamespace(lora_layers_by_block=lambda:[(None,layers)])
        pipeline=SimpleNamespace(lora_layers={'transformer':group})
        self.assertEqual(static_merge_receipt(pipeline),dict(active=312,merged=312,dynamic=0))
        layers['1'].merged=False
        with self.assertRaises(RuntimeError):static_merge_receipt(pipeline)

    def test_disabled_profile_never_touches_cuda(self):
        profiler=WarmStageProfile(cuda=object())
        with profiler.stage('denoising'):pass
        self.assertIsNone(profiler.finish(0,1));self.assertEqual(profiler.records,[])

    def test_profile_records_four_intervals_with_one_final_sync(self):
        events=[]
        class Event:
            def record(self):events.append('record')
            def synchronize(self):events.append('sync')
            def elapsed_time(self,other):return 12.5
        class CUDA:
            def Event(self,**kw):
                if kw!={'enable_timing':True}:raise AssertionError(kw)
                return Event()
        with tempfile.TemporaryDirectory() as directory:
            p=WarmStageProfile(directory,cuda=CUDA())
            for name in ('latent','denoise','video','audio'):
                with p.stage(name):pass
            report=p.finish(1,3)
            self.assertEqual(events.count('sync'),1);self.assertEqual(events.count('record'),8)
            self.assertFalse(report['benchmark_comparable'])
            self.assertTrue((Path(directory)/'rank-1-warm-3.json').is_file())
            with self.assertRaises(FileExistsError):p.finish(1,3)
            paired=WarmStageProfile(directory,cuda=CUDA(),max_hits=6)
            for name in ('latent','denoise','video','audio'):
                with paired.stage(name):pass
            self.assertEqual(paired.finish(0,6)['cache_hit'],6)
            with self.assertRaises(ValueError):paired.finish(0,7)
            with self.assertRaises(ValueError):WarmStageProfile(directory,max_hits=7)

    def test_profile_refuses_partial_stages_and_unbounded_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            p=WarmStageProfile(directory)
            with self.assertRaises(ValueError):p.finish(0,1)
            with self.assertRaises(ValueError):p.finish(0,4)
