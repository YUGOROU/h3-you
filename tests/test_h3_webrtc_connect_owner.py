import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import zipfile

from research import run_h3_webrtc_connect as owner
from research.h3_personal.webrtc_connect_client import candidate_summary


class ConnectOwner(unittest.TestCase):
    def test_offline_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(owner,'run') as run,contextlib.redirect_stdout(io.StringIO()):
            p=Path(tmp)/'fresh'
            self.assertEqual(owner.main(['--directory',str(p)]),0)
            self.assertFalse(p.exists());run.assert_not_called()

    def test_archive_only_rtc_code_lock_and_synthetic_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            # Public pinned bootstrap spec only; never touch private media.
            source=root/'tts-voxcpm2-streaming-20261028-v3';source.mkdir()
            (source/'request.json').write_text(json.dumps({'uv':{'url':'https://example.invalid/uv.whl',
                'sha256':'fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946'}}))
            with patch.object(owner,'PRIVATE',root):archive=owner.stage(root,1500,'a'*32)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()),set(owner.FILES)|{'research/__init__.py','research/h3_personal/__init__.py','request.json'})
                spec=json.loads(z.read('request.json'))
                self.assertEqual(spec['session'],'a'*32)
                self.assertNotIn('model',spec);self.assertNotIn('reference',spec)
                self.assertNotIn('HF_TOKEN',spec)

    def test_cancellation_reconciles_before_clearing_fence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fence=root/'fence.json';fence.write_text(json.dumps({'job_id':'owned'}))
            api=Mock();api.inspect_job.side_effect=[SimpleNamespace(status=SimpleNamespace(stage=s)) for s in ('RUNNING','CANCELED')]
            api.list_jobs.return_value=[]
            with patch.object(owner,'FENCE',fence),patch.object(owner.time,'sleep'):owner.cleanup(api,'owned',root)
            api.cancel_job.assert_called_once_with(job_id='owned',namespace='YUGOROU')
            self.assertFalse(fence.exists())
            self.assertEqual(json.loads((root/'provider-terminal.json').read_text())['status'],'CANCELED')

    def test_unrelated_cpu_is_not_cancelled_or_global_blocker(self):
        api=Mock();other=SimpleNamespace(id='other',flavor='cpu-basic',status=SimpleNamespace(stage='RUNNING'))
        api.list_jobs.return_value=[other]
        self.assertEqual(owner.preexisting_cpu_jobs(api),[dict(id='other',status='RUNNING',flavor='cpu-basic')])
        api.cancel_job.assert_not_called()
        other.flavor='h200x2'
        with self.assertRaises(ValueError):owner.preexisting_cpu_jobs(api)
        api.cancel_job.assert_not_called()

    def test_candidate_summary_never_returns_addresses(self):
        value={'sdp':'a=candidate:1 1 udp 123 192.168.10.2 1234 typ host\r\na=candidate:2 1 udp 123 8.8.8.8 5678 typ srflx\r\n'}
        row=candidate_summary(value)
        self.assertEqual(row,dict(count=2,global_host=0,private_host=1,srflx=1,relay=0))
        self.assertNotIn('192.168',json.dumps(row));self.assertNotIn('8.8.8.8',json.dumps(row))


if __name__=='__main__':unittest.main()
