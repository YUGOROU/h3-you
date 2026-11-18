import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from research import prepare_h3_voxcpm_rtc as prep
from research.h3_personal.voxcpm_endpoint_protocol import PINS
from research.h3_personal.native_tts_backend import validate_native_config

class DeploymentTests(unittest.TestCase):
    def test_explicit_variant_preserves_legacy(self):
        from research.run_h3_native_coupled import deployment
        args=('YUGOROU/h3-private','a'*40,'h3-voxcpm-l4-a474fd64f740','b'*32,'c'*32,int(time.time())+1000)
        old=deployment(*args);rtc=prep.deployment(*args)
        self.assertNotIn('H3_VOX_MEDIA_TRANSPORT',old['model']['env'])
        self.assertEqual(rtc['compute'],old['compute']);self.assertEqual(rtc['provider'],old['provider'])
        self.assertEqual(rtc['model']['image'],old['model']['image'])
        self.assertIn('entrypoint_rtc.py',rtc['model']['command'][1])
        self.assertEqual(rtc['model']['env']['H3_VOX_MEDIA_TRANSPORT'],'webrtc-exact-v1')

    def test_stage_first_party_only_and_locked(self):
        with tempfile.TemporaryDirectory() as d:
            root=prep.stage(Path(d)/'runtime',{'expires_at':int(time.time())+1000})
            seal=json.loads((root/'vox-rtc.json').read_text())
            self.assertEqual(hashlib.sha256((root/'research/webrtc_experiment/uv.lock').read_bytes()).hexdigest(),seal['lock_sha256'])
            self.assertTrue((root/'entrypoint_rtc.py').is_file())
            self.assertFalse(any('.venv' in p.parts or '__pycache__' in p.parts for p in root.rglob('*')))

    def test_rtc_config_selector_is_explicit(self):
        config=dict(schema='h3-native-voxcpm-rtc-v3',url='https://test.endpoints.huggingface.cloud',session='b'*32,
            boot='c'*32,expires_at=time.time()+1000,pins=dict(PINS),preroll=dict(sample_rate=48000,
            reserve_samples=48000,max_reserve_samples=96000,max_wait_seconds=2.0))
        self.assertEqual(validate_native_config(config)['schema'],config['schema'])
        with self.assertRaises(ValueError):validate_native_config(dict(config,schema='fallback'))

    def test_native_completion_not_inferred_from_carrier(self):
        import copy
        from research.run_h3_voxcpm_rtc_acceptance import validate_source
        c={'pins':dict(PINS),'session':'a'*32,'boot':'b'*32,'expires_at':time.time()+600}
        turns=[dict(epoch=i,samples=100,sha256='a'*64,carrier={'transport_complete':True}) for i in (1,2)]
        e=dict(binding=dict(PINS,session=c['session'],boot=c['boot'],expires_at=c['expires_at'],epoch=0),
            evidence=dict(runtime=dict(device_name='NVIDIA L4',device_count=1,optimize=False,sample_rate=48000,torch='2.8.0+cu128'),
            completions=[dict(epoch=i,samples=100,native_float_sha256='a'*64,completion=dict(reason='natural_stop',natural_stop=True,
                stream_exhausted=True,stop_index=2,effective_max_len=10,decision_monotonic_ns=3)) for i in (1,2)]))
        validate_source(c,e,turns)
        bad=copy.deepcopy(e);bad['evidence']['completions'][0]['completion']['natural_stop']=False
        with self.assertRaises(ValueError):validate_source(c,bad,turns)
        bad=copy.deepcopy(e);bad['evidence']['runtime']['device_name']='other'
        with self.assertRaises(ValueError):validate_source(c,bad,turns)

    def test_supervisor_stops_descendant_groups_even_after_leader_exit(self):
        from research.endpoints.voxcpm_l4.entrypoint_rtc import stop_children
        from unittest.mock import Mock
        children=[Mock(pid=12345),Mock(pid=12346)]
        with patch('research.endpoints.voxcpm_l4.entrypoint_rtc.os.killpg') as kill:
            stop_children(children)
        import signal
        self.assertEqual(kill.call_args_list[0].args,(12346,signal.SIGTERM))
        self.assertEqual(kill.call_args_list[-1].args,(12345,signal.SIGKILL))
        for child in children:self.assertEqual(child.wait.call_count,2)

    def test_offline_acceptance_never_opens_lease(self):
        from research import run_h3_voxcpm_rtc_acceptance as harness
        with patch.object(harness,'lease',side_effect=AssertionError('GPU forbidden')),patch('builtins.print') as out:
            harness.main([])
        self.assertFalse(json.loads(out.call_args.args[0])['gpu_started'])

if __name__=='__main__':unittest.main()
