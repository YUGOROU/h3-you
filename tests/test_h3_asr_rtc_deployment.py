"""Offline RTC ASR deployment/capture guards; no Hub, device or model calls."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from research.run_h3_asr_endpoint import deployment,assert_deployment
from research.prepare_h3_asr_rtc import stage,ROOT
from research.endpoints.nemotron_t4.entrypoint_rtc import native_command,RUNTIME_SHA,MODEL_SHA
from research.h3_personal import resident_endpoint as resident

class DeploymentTests(unittest.TestCase):
    def test_one_replica_rtc_explicit_legacy_preserved(self):
        old=deployment('owner/model','a'*40,'test',100)
        rtc=deployment('owner/model','a'*40,'test',100,rtc=True)
        self.assertEqual(old['compute']['scaling'],dict(minReplica=1,maxReplica=3))
        self.assertEqual(rtc['compute']['scaling'],dict(minReplica=1,maxReplica=1))
        self.assertEqual(rtc['model']['env']['H3_ASR_MEDIA_TRANSPORT'],'webrtc-exact-v1')
        assert_deployment(rtc,rtc)
        with self.assertRaises(ValueError):assert_deployment(old,rtc)

    def test_original_native_pins_and_argv_unchanged(self):
        tree=ast.parse((ROOT/'research/endpoints/nemotron_t4/entrypoint.py').read_text())
        constants={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('RUNTIME_SHA','MODEL_SHA')}
        self.assertEqual(constants,dict(RUNTIME_SHA=RUNTIME_SHA,MODEL_SHA=MODEL_SHA))
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        command=next(node for node in main.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in node.targets))
        namespace=dict(binary=Path('/native'),root=Path('/model-root'))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[command],type_ignores=[])),'<native argv>','exec'),namespace)
        self.assertEqual(native_command('/native','/model-root/model.gguf'),namespace['command'])

    def test_stage_is_first_party_only_sealed_and_no_model_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';manifest=stage(root)
            self.assertIn('research/h3_personal/asr_rtc_gateway.py',manifest)
            self.assertFalse(any('.venv' in key or '__pycache__' in key for key in manifest))
            self.assertFalse((root/'model.gguf').exists());self.assertFalse((root/'runtime.tar.gz').exists())
            seal=json.loads((root/'asr-rtc.json').read_text())
            self.assertEqual(seal['lock_sha256'],hashlib.sha256((root/'research/webrtc_experiment/uv.lock').read_bytes()).hexdigest())
            self.assertNotIn('def relay', (root/'gate.py').read_text())

    def test_parent_review_accepts_explicit_rtc_but_not_extra_replicas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=deployment('owner/model','a'*40,'test',1)
            rtc=deployment('owner/model','b'*40,'test',1,rtc=True)
            (root/'intent.json').write_text(json.dumps(rtc));(root/'source-pins.json').write_text('{}')
            (root/'result.json').write_text('{"terminal_state":"paused"}')
            self.assertEqual(resident._parent(original,root,SimpleNamespace(private=True,sha='b'*40)),rtc)
            rtc['compute']['scaling']['maxReplica']=3;(root/'intent.json').write_text(json.dumps(rtc))
            with self.assertRaises(ValueError):resident._parent(original,root,SimpleNamespace(private=True,sha='b'*40))

    def test_factory_and_operator_accounting_validated_before_any_operation(self):
        with self.assertRaises(ValueError):resident.ResidentEndpoint.open(None,None,rtc_factory=object())
        factory=Mock(media_transport='webrtc-exact-v1')
        with self.assertRaises(ValueError):resident.ResidentEndpoint.open(None,None,rtc_factory=factory)

    def test_rtc_failure_cannot_start_capture_or_legacy_transcribe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();lease=resident.ResidentEndpoint()
            lease.root=root;lease.closed=False;lease.owned=True;lease._key='k'*40;lease._token='token';lease._path='/owned'
            lease._intent={};lease._url='https://test.endpoints.huggingface.cloud';lease.lease_id='fresh'
            lease.deadline=time.time()+1800;lease.receipt={};lease._binary=root/'fake';lease._pins={'capture_binary':'pin'}
            lease._rtc_epoch=0;lease._rtc_factory=Mock();lease._rtc_factory.open_turn.side_effect=RuntimeError('RTC unavailable')
            with patch.object(resident,'control',return_value={'status':{'state':'running'}}),patch.object(resident,'assert_deployment'),\
                 patch.object(resident,'sha',return_value='pin'),patch.object(resident,'wait_for_capture',return_value={}),\
                 patch.object(resident.subprocess,'Popen',side_effect=AssertionError('capture forbidden')) as capture,\
                 patch.object(resident,'transcribe',side_effect=AssertionError('legacy forbidden')) as transcribe,\
                 patch.object(resident,'stop_capture'),patch.object(resident,'pause_owned',return_value='paused'):
                with self.assertRaises(RuntimeError):lease.capture(root/'turn',time.time()+3)
            capture.assert_not_called();transcribe.assert_not_called();self.assertEqual(lease._rtc_epoch,1)

if __name__=='__main__':unittest.main()
