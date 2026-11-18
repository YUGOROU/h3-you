"""Deferred adoption contracts; no provider/network/model/device operations."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from research.h3_personal.h3_first_owner import DeferredH3Owner,H3_IMAGE,ROOT
from research.h3_personal.voxcpm_endpoint_protocol import PINS

class FirstOwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.api=Mock();self.client=Mock()
        self.owner=DeferredH3Owner(self.api,self.client,'token',self.root,'h3-group')
        self.owner.job=SimpleNamespace(id='b'*24);self.owner.command=['python','fixed-wrapper']
        self.owner.expires_at=time.time()+600
        self.owner.ready={'boot':'c'*32}
        self.config=dict(schema='h3-native-voxcpm-network-v2',connection=dict(group='h3-group',alias='voxcpm',port=17861,
            certificate_sha256='a'*64,server_job_id='a'*24),session='d'*32,boot='e'*32,expires_at=time.time()+1000,
            pins=dict(PINS),preroll=dict(sample_rate=48000,reserve_samples=48000,max_reserve_samples=96000,max_wait_seconds=2.0))
        self.spec={'publication_repo':'YUGOROU/h3-fresh','live_direct_session':{'tts_mode':'native_voxcpm','native_tts':self.config}}

    def test_launch_adopts_same_job_no_second_post_and_clamps_expiry(self):
        stage=self.root/'stage';(stage/'research').mkdir(parents=True)
        (stage/'research/bootstrap_h3_fast_ref2va.py').write_bytes((ROOT/'research/bootstrap_h3_fast_ref2va.py').read_bytes())
        (stage/'spec.json').write_text(json.dumps(self.spec))
        self.client.request.return_value=SimpleNamespace(status_code=202)
        payload={'dockerImage':H3_IMAGE,'secrets':{'H3_DIRECT_PRODUCER_KEY':'p'*40,'H3_DIRECT_CONSUMER_KEY':'c'*40,'H3_NATIVE_TTS_KEY':'n'*40}}
        with patch.object(self.owner,'admit') as admit:
            job,image,command=self.owner.launch(self.api,stage,'token',payload,'f'*40,'a'*64)
            with self.assertRaises(ValueError):self.owner.launch(self.api,stage,'token',payload,'f'*40,'a'*64)
        self.assertIs(job,self.owner.job);self.assertEqual(image,H3_IMAGE)
        self.assertEqual(command,self.owner.command);self.client.post.assert_not_called()
        self.client.request.assert_called_once()
        capsule=self.client.request.call_args.kwargs['json']
        self.assertEqual(capsule['expires_at'],self.owner.expires_at)
        saved=json.loads((self.root/'h3-first-launch.json').read_text())
        for key in ('producer_key','consumer_key','native_key'):self.assertNotIn(key,saved)
        self.assertEqual(saved['spec_sha256'],hashlib.sha256((stage/'spec.json').read_bytes()).hexdigest())

    def test_admit_rejects_changed_source_command(self):
        from research.h3_personal.director_tts_job import IMAGE
        self.owner.bind_source('a'*24,['python','original'],'f'*40)
        self.api.list_jobs.return_value=[SimpleNamespace(id=i,status=SimpleNamespace(stage='RUNNING')) for i in ('a'*24,'b'*24)]
        h3=dict(id='b'*24,owner={'name':'YUGOROU'},dockerImage=H3_IMAGE,command=self.owner.command,flavor='h200x2',
            network={'group':'h3-group'},retry=0,expose={'ports':[7860]},labels={'h3campaign':self.owner.campaign},status={'stage':'RUNNING'})
        source=dict(id='a'*24,owner={'name':'YUGOROU'},dockerImage=IMAGE,command=['python','changed'],flavor='l4x1',
            network={'group':'h3-group','aliases':['voxcpm']},retry=0,labels={'h3session':'d'*32,'h3boot':'e'*32,'deployrev':'f'*40},status={'stage':'RUNNING'})
        with patch.object(self.owner,'_raw',return_value=h3):
            self.client.get.return_value=SimpleNamespace(status_code=200,json=lambda:source)
            with self.assertRaisesRegex(ValueError,'command/publication drift'):self.owner.admit(self.api,self.spec,'token')
        with self.assertRaises(ValueError):self.owner.bind_source('a'*24,['python'],'f'*40)

    def test_foreign_h3_or_public_port_refused(self):
        with self.assertRaises(ValueError):self.owner._identity({'id':'foreign'})

if __name__=='__main__':unittest.main()
