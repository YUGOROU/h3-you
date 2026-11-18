"""Offline contract tests. No allocation, capture, model inference or upload."""
import contextlib
import copy
from decimal import Decimal
import io
import json
import os
import time
from email.message import Message
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from research import run_h3_asr_endpoint as endpoint


class EndpointTests(unittest.TestCase):
    def test_offline_is_default(self):
        out = io.StringIO()
        with patch.object(sys,'argv',['probe','--directory','/missing']), contextlib.redirect_stdout(out):
            endpoint.main()
        x = json.loads(out.getvalue())
        self.assertEqual((x['min_replica'],x['max_replica']), (1,3))
        self.assertTrue(x['public_samples_only'])

    def test_pinned_t4_one_endpoint_three_replicas(self):
        spec = endpoint.deployment('owner/repo','a'*40,'test',2000000000)
        self.assertEqual(spec['compute']['instanceType'],'nvidia-t4')
        self.assertEqual(spec['compute']['instanceSize'],'x1')
        self.assertEqual(spec['compute']['scaling'],dict(minReplica=1,maxReplica=3))
        self.assertEqual(spec['type'],'authenticated')
        self.assertIn('@sha256:',spec['model']['image']['custom']['url'])
        self.assertEqual(spec['model']['revision'],'a'*40)
        self.assertNotIn('secrets',spec['model'])

    def test_budget_existing_exposure_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'ledger.json'
            x = dict(spent_usd=210,reservations={'old':dict(upper_bound_usd=1)})
            p.write_text(json.dumps(x))
            endpoint.reserve(p,'new')
            self.assertEqual(json.loads(p.read_text())['reservations']['old'],x['reservations']['old'])
            with self.assertRaises(ValueError):
                endpoint.reserve(p,'new')
            x['spent_usd']=231
            p.write_text(json.dumps(x))
            with self.assertRaises(ValueError):
                endpoint.reserve(p,'too-much')
        self.assertEqual(endpoint.RESERVATION,Decimal('0.75'))

    def test_pause_attempted_before_lookup_and_retried(self):
        with patch.object(endpoint,'control',side_effect=[TimeoutError(),{'status':{'state':'paused'}}]) as api, patch.object(endpoint.time,'sleep'):
            self.assertEqual(endpoint.pause_owned('secret','/owned'),'paused')
            self.assertEqual(api.call_count,2)
            for call in api.call_args_list:
                self.assertEqual(call.args[1],'/owned/pause')
        with patch.object(endpoint,'control',side_effect=TimeoutError()), patch.object(endpoint.time,'sleep'):
            with self.assertRaises(RuntimeError):
                endpoint.pause_owned('secret','/owned')

    def test_resume_contract_rejects_modified_provider_runtime_or_schedule(self):
        expected=endpoint.deployment('owner/repo','a'*40,'test',2000000000)
        current=copy.deepcopy(expected)
        current['model']['args']=None
        current['compute']['scaling']['measure']={'hardwareUsage':80}
        endpoint.assert_deployment(current,expected)
        for section,field,value in [('provider','region','eu-west-1'),('compute','instanceType','nvidia-h200'),
                                    ('model','repository','other/repo'),('model','env',{}),('model','image',{})]:
            changed=copy.deepcopy(current);changed[section][field]=value
            with self.assertRaises(ValueError):endpoint.assert_deployment(changed,expected)
        current['compute']['scaling']['schedule']=[{'maxReplica':100}]
        with self.assertRaises(ValueError):endpoint.assert_deployment(current,expected)

    def test_japanese_reference_metric(self):
        from research.validate_h3_asr_endpoint_japanese import cer
        self.assertEqual(cer('これは、テストです。','これはテストです'),0)
        self.assertAlmostEqual(cer('あいう','あえう'),1/3)

    def test_owner_gate_rejects_without_reaching_native(self):
        from research.endpoints.nemotron_t4 import gate
        handler = object.__new__(gate.Gate)
        handler.path = '/v1/audio/transcriptions/realtime'
        handler.headers = Message()
        handler.headers['Upgrade'] = 'websocket'
        handler.send_error = MagicMock()
        handler.relay = MagicMock()
        with patch.dict(os.environ, H3_ASR_EXPIRES_AT=str(int(time.time())+60),H3_ASR_KEY='a'*40):
            handler.do_GET()
            handler.send_error.assert_called_once_with(403)
            handler.relay.assert_not_called()
            handler.headers['X-H3-ASR-Key']='a'*40
            handler.do_GET()
            handler.relay.assert_called_once()

    def test_runtime_fail_closed_and_gate(self):
        root = Path('research/endpoints/nemotron_t4')
        entry = (root/'entrypoint.py').read_text()
        gate = (root/'gate.py').read_text()
        for value in ('H3_ASR_EXPIRES_AT','H3_ASR_KEY',"'cuda'",'timeout=remaining',"filter='data'",endpoint.RUNTIME_SHA,endpoint.MODEL_SHA):
            self.assertIn(value,entry)
        for value in ('hmac.compare_digest','X-H3-ASR-Key',"'authorization'",'4*1024*1024','end = min'):
            self.assertIn(value,gate)
        for value in ('HF_TOKEN', 'print(self.headers', 'requestAccess', 'microphone'):
            self.assertNotIn(value,gate)


if __name__=='__main__':
    unittest.main()
