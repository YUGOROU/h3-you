import ast
from pathlib import Path
import unittest
from research.h3_personal.ref2va_attention import configure_environment


class AttentionTests(unittest.TestCase):
    def test_default_route_resets_inherited_precision_and_packing_flags(self):
        env={'FASTVIDEO_FA4':'1','FASTVIDEO_NVFP4_FA4':'1','FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN':'1'}
        configure_environment('fa2',env)
        self.assertEqual(env['FASTVIDEO_FA4'],'0')
        self.assertEqual(env['FASTVIDEO_NVFP4_FA4'],'0')
        self.assertEqual(env['FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN'],'0')

    def test_explicit_fa4_never_enables_packed_or_quantized_attention(self):
        env=configure_environment('fa4',{})
        self.assertEqual(env['FASTVIDEO_FA4'],'1')
        self.assertEqual(env['H3_REF2VA_ATTENTION_MODE'],'fa4')
        self.assertEqual(env['FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN'],'0')
        self.assertEqual(env['FASTVIDEO_NVFP4_FA4'],'0')
        with self.assertRaises(ValueError):configure_environment('unknown',env)

    def test_packed_is_explicit_and_cold_counter_is_removed(self):
        from types import SimpleNamespace as NS
        from research.h3_personal.ref2va_attention import start_packed_counter,finish_packed_counter
        env=configure_environment('fa4-packed',{})
        self.assertEqual(env['FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN'],'1')
        self.assertEqual(env['FASTVIDEO_NVFP4_FA4'],'0')
        original=lambda x:x+1
        backend=NS(fa_version='4',flash_attn_varlen_func_compilable=original)
        start_packed_counter(backend)
        self.assertEqual(backend.flash_attn_varlen_func_compilable(4),5)
        receipt=finish_packed_counter(backend)
        self.assertEqual(receipt['cold_packed_calls'],1)
        self.assertTrue(receipt['warm_counter_removed'])
        self.assertIs(backend.flash_attn_varlen_func_compilable,original)
        backend.flash_attn_varlen_func_compilable(8)
        self.assertEqual(finish_packed_counter(backend)['cold_packed_calls'],1)

    def test_configuration_precedes_worker_exec_and_fastvideo_import(self):
        text=Path('research/run_h3_ref2va.py').read_text()
        self.assertLess(text.index('configure_environment(a.attention_mode,env)'),text.index('os.execvpe'))
        self.assertLess(text.index('os.execvpe'),text.index('    import fastvideo\n'))
        ast.parse(text)

    def test_cold_guard_rejects_wrong_dispatch_and_hardware_before_any_kernel(self):
        from unittest.mock import patch
        from types import SimpleNamespace as NS
        from research.h3_personal.ref2va_attention import cold_receipt
        import sys
        default=NS(fa_version='2',flash_attn_func_compilable=lambda *a:None)
        modules={'torch':NS(cuda=NS(current_device=lambda:0,get_device_capability=lambda device:(8,9))),
                 'fastvideo.attention.utils.flash_attn_default':default,
                 'fastvideo.attention.utils.flash_attn_no_pad':NS(_FA_VARLEN_VERSION='4')}
        with patch.dict(sys.modules,modules):
            with self.assertRaisesRegex(RuntimeError,'dispatcher'):cold_receipt()
            default.fa_version='4'
            with self.assertRaisesRegex(RuntimeError,'SM90'):cold_receipt()

    def test_staging_includes_new_helper_and_bootstrap_forwards_selector(self):
        prepare=Path('research/prepare_h3_fast_ref2va_gpu.py').read_text()
        boot=Path('research/bootstrap_h3_fast_ref2va.py').read_text()
        self.assertIn('research/h3_personal/ref2va_attention.py',prepare)
        self.assertIn("configure_environment(spec.get('attention_mode','fa2'))",boot)
        self.assertIn("command.extend(['--attention-mode'",boot)


if __name__=='__main__':unittest.main()
