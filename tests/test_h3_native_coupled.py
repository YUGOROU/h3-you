import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from research import run_h3_native_coupled as owner
from research.launch_h3_direct_conversation import collect_native_worker_receipts


class Coupled(unittest.TestCase):
    def test_staged_import_closure_no_model_or_runtime_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp)/'deployment';owner.copy_runtime(stage,{'candidate':'voxcpm2'})
            result=subprocess.run([sys.executable,'-c',
                'import research.h3_personal.voxcpm_endpoint_service;import entrypoint;print("ok")'],
                cwd=stage,capture_output=True,text=True,timeout=5,env={'PATH':'/usr/bin:/bin'})
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'ok')
            self.assertFalse((stage/'reference.wav').exists())
            spec=owner.deployment('YUGOROU/fresh','a'*40,'fresh','b'*32,'c'*32,1500)
            self.assertEqual(spec['model']['command'],['python','/repository/entrypoint.py','/repository'])
            self.assertEqual(spec['compute']['scaling'],{'minReplica':1,'maxReplica':1})
            self.assertNotIn('secrets',spec['model'])

    def test_diagnostic_failure_still_confirms_job_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'h3').mkdir();(root/'h3/submission.json').write_text(json.dumps({'job_id':'owned'}))
            api=Mock();api.fetch_job_logs.side_effect=OSError('logs unavailable')
            api.inspect_job.side_effect=[SimpleNamespace(status=SimpleNamespace(stage=s)) for s in ('RUNNING','CANCELED')]
            api.list_jobs.return_value=[]
            with patch.object(owner.time,'sleep'),self.assertRaises(OSError):owner.collect_h3(api,root)
            api.cancel_job.assert_called_once_with(job_id='owned',namespace='YUGOROU')
            self.assertEqual(json.loads((root/'h3-terminal.json').read_text())['status'],'CANCELED')

    def test_receipts_collected_before_job_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            row={'type':'h3_native_tts_receipts','receipts':[{'epoch':1},{'epoch':2}]}
            api=Mock();api.fetch_job_logs.return_value=['unrelated',json.dumps(row)]
            found=collect_native_worker_receipts(api,SimpleNamespace(id='owned'),tmp,time.monotonic()+1)
            self.assertEqual(found,row)
            self.assertEqual(json.loads((Path(tmp)/'native-worker-evidence.json').read_text()),row)
            api.cancel_job.assert_not_called()

    def test_bad_receipt_epochs_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            api=Mock();api.fetch_job_logs.return_value=[json.dumps({'type':'h3_native_tts_receipts','receipts':[{'epoch':2}]})]
            with self.assertRaises(ValueError):collect_native_worker_receipts(api,SimpleNamespace(id='owned'),tmp,time.monotonic()+1)
            self.assertFalse((Path(tmp)/'native-worker-evidence.json').exists())

    def evidence_fixture(self,root):
        (root/'h3').mkdir();(root/'packets').mkdir()
        config=dict(pins=dict(owner.PINS),session='a'*32,boot='b'*32,expires_at=time.time()+300)
        receipts=[];completions=[];offset=0
        for epoch in (1,2):
            pcm=epoch.to_bytes(2,'little')*3
            (root/f'packets/{epoch:03d}.pcm').write_bytes(pcm)
            row=dict(sample_start=offset,conversation_final=epoch==2,
                regions=[dict(kind='speech',source_epoch=epoch,sample_start=offset,sample_stop=offset+3)])
            (root/f'packets/{epoch:03d}.json').write_text(json.dumps(row));offset+=3
            receipts.append(dict(epoch=epoch,pins=dict(owner.PINS),native_sha256=str(epoch)*64,
                native_samples=3,converted_samples=3,converted_sha256=hashlib.sha256(pcm).hexdigest()))
            completions.append(dict(epoch=epoch,native_float_sha256=str(epoch)*64,samples=3,
                completion=dict(reason='natural_stop',natural_stop=True,stream_exhausted=True,
                    stop_index=3,effective_max_len=10,decision_monotonic_ns=123)))
        (root/'h3/native-worker-evidence.json').write_text(json.dumps({'receipts':receipts}))
        evidence=dict(binding=dict(owner.PINS,session=config['session'],boot=config['boot'],expires_at=config['expires_at'],epoch=0),
            evidence=dict(runtime=dict(device_name='NVIDIA L4',device_count=1,optimize=False,sample_rate=48000,torch='2.8.0+cu128'),completions=completions))
        return config,evidence,dict(complete=True,final=True)

    def test_exact_native_and_sink_two_turn_crosscheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,evidence,result=self.evidence_fixture(root)
            self.assertTrue(owner.validate_evidence(root,config,evidence,result)['exact_two_turn_pcm'])
            (root/'packets/002.pcm').write_bytes(b'wrong')
            with self.assertRaises(ValueError):owner.validate_evidence(root,config,evidence,result)

    def test_protocol_end_is_not_native_eos_or_second_turn(self):
        for change in ('one_turn','capped','hash'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);config,evidence,result=self.evidence_fixture(root)
                rows=evidence['evidence']['completions']
                if change=='one_turn':rows.pop()
                elif change=='capped':rows[0]['completion']['reason']='max_len'
                else:rows[0]['native_float_sha256']='f'*64
                with self.assertRaises(ValueError):owner.validate_evidence(root,config,evidence,result)

    def test_offline_does_not_create_or_query(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(owner,'run') as run:
            self.assertEqual(owner.main(['--directory',str(Path(tmp)/'absent')]),0)
            run.assert_not_called()
            self.assertFalse((Path(tmp)/'absent').exists())


if __name__=='__main__':unittest.main()
