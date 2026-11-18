"""CPU-only cache/provenance contracts; no model, torch, network or raw media."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from research.h3_personal import condition_cache as c
from research.hf_identity_preview import conditioning_identity


class ConditionCacheTests(unittest.TestCase):
    def test_exact_hit_and_key_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'private';identity={'prompt_sha256':'a'*64,'geometry':[768,1344]}
            self.assertIsNone(c.load(root,identity,Path.read_bytes))
            key=c.store(root,identity,lambda p:p.write_bytes(b'exact-bf16-test-payload'))
            self.assertEqual(c.load(root,identity,Path.read_bytes),b'exact-bf16-test-payload')
            self.assertIsNone(c.load(root,dict(identity,geometry=[512,896]),Path.read_bytes))
            self.assertEqual((root/key).stat().st_mode & 0o077,0)
            with self.assertRaises(ValueError):
                c.store(root,identity,lambda p:p.write_bytes(b'overwrite'))

    def test_corruption_partial_and_symlinks_fail_closed(self):
        for kind in ('payload','marker','partial','payload-link','entry-link'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'private';identity={'key':1}
                key=c.store(root,identity,lambda p:p.write_bytes(b'known'))
                entry=root/key;payload=entry/'condition.safetensors';marker=entry/'VERIFIED.json'
                if kind=='payload':payload.write_bytes(b'changed')
                if kind=='marker':
                    value=json.loads(marker.read_text());value['key_sha256']='0'*64
                    marker.write_text(json.dumps(value))
                if kind=='partial':marker.unlink()
                if kind=='payload-link':
                    other=Path(tmp)/'outside';other.write_bytes(b'known');payload.unlink();payload.symlink_to(other)
                if kind=='entry-link':
                    target=root/'original';entry.rename(target);entry.symlink_to(target,target_is_directory=True)
                with self.assertRaises(ValueError):c.load(root,identity,Path.read_bytes)

    def test_failed_writer_never_publishes_and_nonprivate_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'private'
            def fail(p):
                p.write_bytes(b'partial');raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):c.store(root,{'key':1},fail)
            self.assertEqual(list(root.iterdir()),[])
            root.chmod(0o755)
            with self.assertRaises(ValueError):c.load(root,{'key':1},Path.read_bytes)

    def test_prompt_tensor_contract_checked_without_models(self):
        from types import SimpleNamespace as N
        from unittest.mock import patch
        from research.hf_identity_preview import validate_prompt_tensors
        torch=N(bfloat16='bf16',long='int64',isfinite=lambda t:N(all=lambda:t.finite))
        def tensors(dtype='bf16',tag_dtype='int64',rows=3,tag_rows=3,finite=True):
            return {'prompt_embeds':N(ndim=2,shape=(rows,8),dtype=dtype,finite=finite),
                    'text_token_tags':N(ndim=1,shape=(tag_rows,),dtype=tag_dtype)}
        with patch.dict('sys.modules',torch=torch):
            good=tensors();self.assertIs(validate_prompt_tensors(good),good)
            for bad in (tensors(dtype='fp32'),tensors(tag_dtype='fp32'),
                        tensors(tag_rows=2),tensors(rows=0),tensors(finite=False),{}):
                with self.assertRaises(ValueError):validate_prompt_tensors(bad)

    def test_identity_binds_all_conditioning_but_not_audio_seed_adapter(self):
        request={'pins':{'model':'42ed'},'model_inventory':{'model':'a'*64},
                 'runtime_overlay':{'stage_sha256':'b'*64,'stage':'/first'}}
        case={'prompt':'hello','reference_image':{'sha256':'c'*64},'seed':42,'audio_sha256':'d'*64}
        reference={'sha256':'e'*64};geometry=[768,1344]
        identity=conditioning_identity(request,case,reference,geometry)
        variant=dict(case,seed=314159,audio_sha256='f'*64)
        self.assertEqual(identity,conditioning_identity(request,variant,reference,geometry))
        for what in ('prompt','image','reference','geometry','model','source','overlay'):
            r=copy.deepcopy(request);cs=copy.deepcopy(case);ref=copy.deepcopy(reference);g=geometry
            if what=='prompt':cs['prompt']='another'
            if what=='image':cs['reference_image']['sha256']='f'*64
            if what=='reference':ref['sha256']='f'*64
            if what=='geometry':g=[512,896]
            if what=='model':r['model_inventory']['model']='f'*64
            if what=='source':r['pins']['model']='another'
            if what=='overlay':r['runtime_overlay']['stage_sha256']='f'*64
            self.assertNotEqual(c.address(identity),c.address(conditioning_identity(r,cs,ref,g)))
        self.assertNotIn('hello',json.dumps(identity))
        self.assertNotIn('/first',json.dumps(identity))
        with self.assertRaises(ValueError):c.address({'not_finite':float('nan')})


if __name__=='__main__':unittest.main()
