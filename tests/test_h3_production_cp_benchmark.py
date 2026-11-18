import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace, ModuleType
from research.benchmark_h3_context_parallel import main, pinned_json


class ProductionCPTests(unittest.TestCase):
    def test_plan_only_has_no_gpu_or_file_access(self):
        args=[]
        for name in ('request','checkpoint','packed','metadata','source-root','source-receipts','output'):
            args.extend(['--'+name,'/not/read'])
        args+=['--metadata-sha256','none','--source-receipts-sha256','none','--full-trajectory']
        output=io.StringIO()
        with contextlib.redirect_stdout(output):self.assertEqual(main(args),0)
        result=json.loads(output.getvalue())
        self.assertTrue(result['plan_only']);self.assertTrue(result['full_trajectory'])
        self.assertEqual(result['denoiser_forwards'],49)
        self.assertFalse(result['quality_verified'])

    def test_pinned_metadata_rejects_drift(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'pin.json';b=b'{"ok":true}';p.write_bytes(b)
            self.assertTrue(pinned_json(p,hashlib.sha256(b).hexdigest())['ok'])
            with self.assertRaises(ValueError):pinned_json(p,'0'*64)

    def test_installer_refiner_exclusion_and_shape_guards(self):
        from research.h3_personal import context_parallel_runtime as runtime
        hooks=[[],[]]
        blocks=[SimpleNamespace(register_forward_hook=lambda hook,i=i: hooks[i].append(hook)) for i in range(2)]
        processor=SimpleNamespace(_parallel_config='replicated-must-clear')
        model=SimpleNamespace(config=SimpleNamespace(num_layers=2,num_attention_heads=4,
                              num_refiner_layers=1,hidden_size=128),transformer_blocks=blocks,
                              token_refiner=SimpleNamespace(modules=lambda:[SimpleNamespace(processor=processor)]),
                              set_attention_backend=lambda backend:None,
                              enable_parallelism=lambda **kwargs:None)
        parallel=ModuleType('diffusers.models._modeling_parallel')
        parallel.ContextParallelConfig=lambda **kwargs:kwargs
        torch=ModuleType('torch');dist=ModuleType('torch.distributed')
        dist.is_initialized=lambda:True;dist.get_world_size=lambda:2
        torch.distributed=dist
        with patch.dict('sys.modules',{'torch':torch,'torch.distributed':dist,
                                      'diffusers.models._modeling_parallel':parallel}), \
             patch.object(runtime,'verify_source_files'),patch.object(runtime,'materialize_plan',side_effect=lambda x:x):
            result=runtime.install(model,sequence_rows=32,source_root='unused',source_receipts=[{}],allowed_sequence_rows=(32,48))
            self.assertIsNone(processor._parallel_config)
            hooks[0][0](None,None,SimpleNamespace(shape=(1,16,128)))
            with self.assertRaises(ValueError):result['set_sequence_rows'](48)
            hooks[1][0](None,None,SimpleNamespace(shape=(1,32,128)))
            self.assertEqual(result['observed_blocks'],[0,1])
            result['set_sequence_rows'](48)
            hooks[0][0](None,None,SimpleNamespace(shape=(1,24,128)))
            hooks[1][0](None,None,SimpleNamespace(shape=(1,48,128)))
            with self.assertRaises(ValueError):result['set_sequence_rows'](64)
            with self.assertRaises(ValueError):hooks[0][0](None,None,SimpleNamespace(shape=(1,32,128)))
            with self.assertRaises(ValueError):runtime.install(model,sequence_rows=32,source_root='unused',source_receipts=[{}])

    def test_pack_is_local_private_only(self):
        source=Path('research/hf_identity_preview.py').read_text()
        block=source.split('    if packed_only:',1)[1].split('    start = time.monotonic()',1)[0]
        self.assertIn('private_conditioning_do_not_upload',block)
        self.assertIn("case['schedule_points'] != 50",block)
        self.assertNotIn('create_commit',block)
