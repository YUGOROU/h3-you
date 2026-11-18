import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research import hf_identity_pipeline as pipe


class RepreviewTests(unittest.TestCase):
    def test_real_cli_accepts_handoff_and_dispatches(self):
        import os
        import signal
        old_mask=os.umask(0o077);os.umask(old_mask)
        handler=signal.getsignal(signal.SIGTERM)
        try:
            with patch.object(pipe,'pinned'), patch.object(pipe,'validate',return_value={}), \
                 patch.object(pipe,'approvals'), patch.object(pipe,'internal') as internal:
                result=pipe.main(['--request','synthetic.json','--request-sha256','0'*64,
                                  '--internal-stage','handoff','--seconds','100'])
                self.assertEqual(result,0)
                internal.assert_called_once_with('handoff',Path('synthetic.json'),'0'*64,100)
        finally:
            signal.signal(signal.SIGTERM,handler);os.umask(old_mask)

    def test_preview_only_never_invokes_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            request={'output':tmp,'preview_only':True}
            source=root/'input-checkpoint';source.mkdir()
            (source/'adapter_model.safetensors').write_bytes(b'synthetic')
            metadata=root/'HANDOFF.json';metadata.write_text(json.dumps({'parent_step':64}))
            value={'work':str(root/'training'), 'warm_start':{'checkpoint':str(source),'metadata':str(metadata)}}
            with patch.object(pipe,'validate',return_value=request), patch.object(pipe,'pinned'), \
                 patch.object(pipe,'approvals'), patch.object(pipe,'train_request',return_value=value), \
                 patch.object(pipe.trainer,'materialize_runtime'), patch.object(pipe.trainer,'main') as main, \
                 patch.dict('os.environ',{},clear=False):
                (root/'train-request.json').write_text('{}')
                pipe.internal('handoff','unused','unused',100)
                main.assert_not_called()
                saved=json.loads((root/'training/training-snapshot.json').read_text())
                self.assertEqual(saved['status'],'inference_only')
                checkpoint=Path(saved['checkpoint'])
                self.assertEqual(checkpoint.parent.name,'checkpoint-0000064')
                self.assertEqual((checkpoint/'adapter_model.safetensors').read_bytes(),b'synthetic')
                self.assertTrue(source.is_dir())
                self.assertNotIn('actual_updates',saved)
                # Exercise the exact uploader that failed on the fetched escrow path.
                from types import SimpleNamespace as NS
                from unittest.mock import Mock
                api=Mock();api.repo_info.return_value=NS(private=True)
                hub=NS(HfApi=Mock(return_value=api),CommitOperationAdd=lambda **kw:kw)
                with patch.dict('sys.modules',{'huggingface_hub':hub}), \
                     patch.object(pipe.trainer,'helper',return_value=NS(inspect_verified=Mock())), \
                     patch.object(pipe.trainer,'upload_run_id',return_value='synthetic-new-run'), \
                     patch.object(pipe.trainer,'CANONICAL',('adapter_model.safetensors',)):
                    pipe.trainer.upload_validated({'pins':{},'output_repo':'YUGOROU/synthetic'},checkpoint)
                operation=api.create_commit.call_args.kwargs['operations'][0]
                self.assertIn('/training/checkpoint-0000064/',operation['path_in_repo'])

    def test_carrier_progress_is_strictly_sanitized(self):
        from research.modal_identity_worker import safe_progress
        self.assertEqual(safe_progress('PREVIEW_DENOISE 4 / 50\n'),'PREVIEW_DENOISE 4 / 50')
        self.assertEqual(safe_progress('H3_PHASE handoff\n'),'H3_PHASE handoff')
        for line in ('token=synthetic-secret', 'H3_PHASE inputs synthetic-secret',
                     'PREVIEW_DENOISE 4 / 50 synthetic-secret', 'x'*5000):
            self.assertIsNone(safe_progress(line))

    def test_operator_managed_stages_have_no_phase_deadlines(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out'
            r={'output':str(out),'overall_seconds':2300,'stage_seconds':1,
               'preview_only':True,'operator_managed_stages':True,'cache_reuse':{},
               'inference':{'seconds':1800}}
            runner=Mock()
            with patch.object(pipe,'approvals'), \
                 patch.object(pipe.encoder,'private_root',side_effect=lambda p:p.mkdir()), \
                 patch.object(pipe.trainer,'read_json',return_value={'uploaded':True,
                    'status':'final_strict_reload_verified_quality_unverified'}):
                pipe.run(r,Path('unused'),'0'*64,runner=runner)
            self.assertEqual(runner.call_count,5)
            self.assertTrue(all(c.kwargs['deadline'] is None for c in runner.call_args_list))
            self.assertTrue(all(c.args[0][-1]=='2300' for c in runner.call_args_list))

    def test_unbounded_wait_retains_failure_and_cancellation_handling(self):
        from unittest.mock import Mock
        process=Mock();process.wait.return_value=0
        with patch.object(pipe.trainer.subprocess,'Popen',return_value=process):
            pipe.trainer.bounded_process(['synthetic'],deadline=None)
        process.wait.assert_called_once_with(timeout=None)
        with self.assertRaisesRegex(ValueError,'cannot train'):
            pipe.trainer.train_stage({'operator_managed_stages':True})

    def test_invalid_preview_only_fails_before_artifacts(self):
        for flag in ('true',1,True):
            with self.assertRaises(ValueError):
                pipe.validate({'schema':'h3-identity-pipeline-v1','preview_only':flag})


if __name__=='__main__':unittest.main()
