import contextlib
import io
import json
import unittest
from research.h3_personal.acceleration_experiment import validate,selected,SCHEMA
from research.build_h3_acceleration_experiment import experiment,PRESETS,main,build
from research.prepare_h3_fp8_runtime import main as install


class AccelerationExperimentTests(unittest.TestCase):
    def test_separate_presets_are_explicit_and_keep_valid_grid(self):
        from research.h3_personal.continuation_plan import plan
        for name in PRESETS:
            config=experiment(name);self.assertEqual(validate(config),config)
            windows=plan(685,config['window_frames'],22)
            self.assertEqual(windows[-1].end,685)
            self.assertTrue(all(w.frames>=124 for w in windows))
        self.assertEqual(len(plan(685,158,22)),5)
        self.assertEqual(len(plan(685,277,22)),3)

    def test_no_unknown_or_implicit_precision_recipe(self):
        c=experiment('fp8-ffn')
        for key,value in [('quantization','int4'),('window_frames',90),('fast_accum',1),('schema','baseline')]:
            with self.assertRaises(ValueError):validate(dict(c,**{key:value}))
        with self.assertRaises(ValueError):validate(dict(c,fuse_lora=True))

    def test_only_requested_base_block_linears_selected(self):
        self.assertTrue(selected('transformer_blocks.0.ff.net.0.proj.base_layer','fp8-ffn'))
        self.assertTrue(selected('transformer_blocks.0.attn.to_q.base_layer','fp8-blocks'))
        self.assertFalse(selected('transformer_blocks.0.attn.to_q.base_layer','fp8-ffn'))
        for name in ('transformer_blocks.0.ff.net.0.proj.lora_A.default',
                     'transformer_blocks.0.attn.to_q.lora_B.h3_ref2va_turbo8',
                     'token_refiner.transformer_blocks.0.ff.net.0.proj',
                     'video_proj','transformer_blocks.0.norm1.linear'):
            self.assertFalse(selected(name,'fp8-blocks'),name)
        self.assertFalse(selected('token_refiner.blocks.0.ff.net.0.proj','fp8-blocks'))

    def test_builder_default_is_plan_only_no_launch(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):self.assertEqual(main([]),0)
        value=json.loads(output.getvalue());self.assertTrue(value['plan_only']);self.assertFalse(value['gpu_submitted'])
        self.assertEqual(value['configuration']['quantization'],'fp8-blocks')
        self.assertTrue(value['configuration']['incremental_decode'])

    def test_finite_job_builder_preserves_baseline_and_uses_requested_timeout(self):
        import copy
        from unittest.mock import patch
        from test_h3_compute_price_approval import proof
        from test_h3_native_worker import contract
        p=proof();p['scope']='h3-corrected-native-continuation-v1'
        old={'pipeline':{'preview_only':True,'inference':{'native_continuation':contract()},
             'hf_price_precision_approval':p,'territory_approval':{'provider':'huggingface_jobs'}},
             'runtime_bootstrap':{}}
        manifest={'bootstrap_files':{n:'a'*64 for n in ('research/prepare_h3_fp8_runtime.py',
            'research/h3_personal/acceleration_experiment.py','research/h3_personal/incremental_decode.py')}}
        for seconds in (2400,3600,4800):
            before=copy.deepcopy(old)
            with patch('research.build_h3_acceleration_experiment.wire.load',side_effect=[old,manifest]), \
                 patch('research.build_h3_acceleration_experiment.candidate.build',return_value=copy.deepcopy(old)):
                value=build('synthetic',runtime_mount='unused',runtime_repo='owner/runtime',runtime_revision='a'*40,
                    preset='fp8-short-incremental',spent_and_reserved_usd=0,total_seconds=seconds)
            self.assertEqual(old,before)
            self.assertEqual(value['pipeline']['overall_seconds']+value['runtime_bootstrap']['timeout_seconds'],seconds)
            self.assertEqual(value['pipeline']['hf_price_precision_approval']['scope'],SCHEMA)
            self.assertGreaterEqual(value['pipeline']['overall_seconds'],1201)

    def test_price_scope_does_not_silently_reuse_baseline_approval(self):
        from test_h3_compute_price_approval import proof
        from test_h3_native_worker import contract
        from research.h3_personal.compute_price_approval import request_limit,RATE
        p=proof();p['scope']='h3-corrected-native-continuation-v1'
        r=dict(hf_price_precision_approval=p,rate_usd_hour=RATE,preview_only=True,overall_seconds=3600,
            territory_approval={'provider':'huggingface_jobs'},
            inference={'native_continuation':contract(),'acceleration_experiment':experiment('fp8-blocks')})
        with self.assertRaises(ValueError):request_limit(r)
        p['scope']=SCHEMA;self.assertEqual(request_limit(r),RATE)
        with self.assertRaises(ValueError):request_limit(dict(r,inference={'native_continuation':contract()}))

    def test_dependency_installer_does_not_install_by_default(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):self.assertEqual(install([]),0)
        value=json.loads(output.getvalue());self.assertTrue(value['plan_only'])
        self.assertFalse(value['torch_upgrade']);self.assertEqual(value['version'],'0.13.0')
