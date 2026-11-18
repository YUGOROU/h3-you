"""Version/config/READY/CLI cancellation binding; offline only."""
import copy
import io
import os
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from test_h3_native_tts_backend import config
from test_h3_direct_conversation_spec import spec
from research.h3_personal.native_tts_backend import (CANCEL_SCHEMA, config_sha256,
    validate_native_config, prepare_direct_tts)
from research.h3_personal.voxcpm_endpoint_protocol import CANCEL_PROTOCOL
from research.hf_h3_direct_conversation import validate_spec
from research.launch_h3_direct_conversation import validate_ready, main


def cancelled_config():
    return dict(config(), schema=CANCEL_SCHEMA, cancel_protocol=CANCEL_PROTOCOL)


class Tests(unittest.TestCase):
    def test_versioned_config_cannot_sneak_into_v1(self):
        value=cancelled_config()
        self.assertEqual(validate_native_config(value),value)
        self.assertNotEqual(config_sha256(value),config_sha256(config()))
        wrong=dict(value,schema='h3-native-voxcpm-backend-v1')
        with self.assertRaises(ValueError):validate_native_config(wrong)
        for bad in (None,True,'unknown'):
            with self.assertRaises(ValueError):validate_native_config(dict(value,cancel_protocol=bad))

    def test_factory_requires_exact_optin_and_consumes_secrets_without_local_loading(self):
        row=dict(tts_mode='native_voxcpm',native_tts=cancelled_config(),native_tts_cancel=True)
        with patch.dict(os.environ,{'H3_NATIVE_TTS_KEY':'test-key','H3_NATIVE_TTS_HF_TOKEN':'test-token'}):
            submit=Mock()
            client=prepare_direct_tts(None,dict(live_direct_session=row),submit).result()
            self.assertEqual(client.cancel_protocol,CANCEL_PROTOCOL)
            self.assertFalse(submit.called)
            self.assertNotIn('H3_NATIVE_TTS_KEY',os.environ)
            client.close()
        for flag in (False,1,None):
            with self.assertRaises(ValueError):
                prepare_direct_tts(None,dict(live_direct_session=dict(row,native_tts_cancel=flag)),Mock())

    def test_immutable_worker_spec_requires_schema_flag_and_https_interruption(self):
        value=spec(); row=value['live_direct_session']
        row.update(tts_mode='native_voxcpm',native_tts=cancelled_config(),
                   native_tts_cancel=True,turn_interruptions=True,continuous_av=True)
        self.assertEqual(validate_spec(value),row)
        for mutate in (
            lambda r:r.pop('native_tts_cancel'),
            lambda r:r.update(native_tts_cancel=1),
            lambda r:r.update(turn_interruptions=False),
            lambda r:r.update(native_tts=config()),
            lambda r:r.update(continuous_av=False)):
            bad=copy.deepcopy(value);mutate(bad['live_direct_session'])
            with self.assertRaises(ValueError):validate_spec(bad)
        bad=copy.deepcopy(value);bad['native_tts_cancel']=True
        with self.assertRaises(ValueError):validate_spec(bad)

    def test_ready_rejects_downgrade_and_undeclared_cancel(self):
        value=cancelled_config()
        ready=dict(type='h3_direct_ready',status='ready',session='a'*32,profile_sha256='b'*64,
            epoch=1,max_segments=16,max_total_frames=1654,port=7860,expires_at=time.time()+300,
            tts_mode='native_voxcpm',native_tts_config_sha256=config_sha256(value),
            continuous_av=True,turn_interruptions=True,native_tts_cancel=True)
        kw=dict(tts_mode='native_voxcpm',native_tts=value,continuous_av=True,
                turn_interruptions=True,native_tts_cancel=True)
        self.assertEqual(validate_ready(ready,'a'*32,**kw),ready)
        for change in ({'native_tts_cancel':False},{'native_tts_cancel':1},
                       {'native_tts_config_sha256':config_sha256(config())}):
            with self.assertRaises(ValueError):validate_ready(dict(ready,**change),'a'*32,**kw)
        with self.assertRaises(ValueError):validate_ready(ready,'a'*32,**dict(kw,native_tts_cancel=False))

    def test_native_acceptance_rejects_post_eos_cancel_and_hash_or_clock_drift(self):
        from research.h3_personal.voxcpm_endpoint_protocol import PINS
        from research.launch_h3_direct_conversation import validate_native_cancel_evidence
        worker=dict(receipts=[dict(epoch=i,native_samples=100,native_sha256=str(i)*64,
            completion='interrupted_protocol_END' if i==1 else 'protocol_END',pins=dict(PINS)) for i in (1,2)])
        control=dict(receipts=[dict(epoch=1,status='requested',ack=dict(cancel_monotonic_ns=100))])
        endpoint=dict(evidence=dict(runtime=dict(device_count=1,device_name='NVIDIA L4',optimize=False),
            completions=[dict(epoch=1,samples=100,native_float_sha256='1'*64,completion=dict(
                reason='cancelled',cancel_prior_reason='unknown',natural_stop=False,
                stream_exhausted=False,ended_monotonic_ns=200)),
                dict(epoch=2,samples=100,native_float_sha256='2'*64,completion=dict(
                reason='natural_stop',natural_stop=True,stream_exhausted=True,
                stop_index=10,effective_max_len=100,started_monotonic_ns=300))]))
        self.assertTrue(validate_native_cancel_evidence(worker,control,endpoint)['native_cancel_exercised'])
        for mutate in (lambda e:e['evidence']['completions'][0]['completion'].update(cancel_prior_reason='natural_stop'),
                       lambda e:e['evidence']['completions'][1].update(native_float_sha256='0'*64),
                       lambda e:e['evidence']['completions'][1]['completion'].update(started_monotonic_ns=50),
                       lambda e:e['evidence']['completions'][1]['completion'].update(natural_stop=False)):
            bad=copy.deepcopy(endpoint);mutate(bad)
            with self.assertRaises(ValueError):validate_native_cancel_evidence(worker,control,bad)

    def test_native_trigger_is_immutable_optin_not_new_default(self):
        value=spec(); row=value['live_direct_session']
        row.update(tts_mode='native_voxcpm',native_tts=cancelled_config(),native_tts_cancel=True,
                   turn_interruptions=True,continuous_av=True,native_cancel_trigger='native-preroll-v1')
        self.assertEqual(validate_spec(value),row)
        for change in ({'native_tts_cancel':False},{'native_cancel_trigger':'sleep-2s'},
                       {'native_cancel_trigger':True}):
            bad=copy.deepcopy(value);bad['live_direct_session'].update(change)
            with self.assertRaises(ValueError):validate_spec(bad)
        ready=dict(type='h3_direct_ready',status='ready',session='a'*32,profile_sha256='b'*64,
            epoch=1,max_segments=16,max_total_frames=1654,port=7860,expires_at=time.time()+300,
            tts_mode='native_voxcpm',native_tts_config_sha256=config_sha256(row['native_tts']),
            continuous_av=True,turn_interruptions=True,native_tts_cancel=True,
            native_cancel_trigger='native-preroll-v1')
        kw=dict(tts_mode='native_voxcpm',native_tts=row['native_tts'],continuous_av=True,
                turn_interruptions=True,native_tts_cancel=True,native_cancel_trigger='native-preroll-v1')
        self.assertEqual(validate_ready(ready,'a'*32,**kw),ready)
        bad=dict(ready);del bad['native_cancel_trigger']
        with self.assertRaises(ValueError):validate_ready(bad,'a'*32,**kw)
        with self.assertRaises(ValueError):validate_ready(ready,'a'*32,**dict(kw,native_cancel_trigger=None))

    def test_cli_offline_and_optin_guard(self):
        with self.assertRaises(SystemExit):main(['--directory','/tmp/h3-never-created','--native-tts-cancel'])
        output=io.StringIO()
        with redirect_stdout(output):
            main(['--directory','/tmp/h3-never-created','--native-tts-cancel',
                  '--headless-interrupt','--vox-parent','/tmp/not-contacted'])
        self.assertIn('Native cancel v2',output.getvalue())
        self.assertFalse(os.path.exists('/tmp/h3-never-created'))


if __name__=='__main__':unittest.main()
