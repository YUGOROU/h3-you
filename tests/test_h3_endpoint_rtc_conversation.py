import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from research import run_h3_endpoint_rtc_conversation as owner

class OwnerTests(unittest.TestCase):
    def exercise(self,unknown=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        base=Path(tmp.name);root=base/'run';fence=base/'fence'
        @contextmanager
        def lease(path,**kwargs):
            self.assertEqual(kwargs['min_remaining'],600)
            path.mkdir()
            try:yield dict(config={'schema':'h3-native-voxcpm-rtc-v3','url':'https://test.endpoints.huggingface.cloud',
                'session':'a'*32,'boot':'b'*32,'expires_at':9999999999},token='token',key='k'*40)
            finally:(path/'terminal.json').write_text(json.dumps({'independent_confirmation':True}))
        def direct(path,turns,**kwargs):
            self.assertEqual(kwargs['native_tts']['schema'],'h3-native-voxcpm-rtc-v3')
            self.assertEqual(kwargs['media_transport'],'webrtc-rtp-v1')
            self.assertFalse(kwargs['capture_comet']);self.assertNotIn('asr_rtc_factory',kwargs)
            path.mkdir();(path/'submission_guard.json').write_text('{}')
            if unknown:raise RuntimeError('unknown create')
            (path/'submission.json').write_text(json.dumps({'job_id':'a'*24}))
            (path/'rtc-job-terminal.json').write_text(json.dumps({'job_id':'a'*24,'status':'CANCELED'}))
            return {'media_complete':False,'source_complete':True,'turns':2}
        control=Mock();control._request.return_value={}
        with patch.object(owner,'PRIVATE',base),patch.object(owner,'FENCE',fence),patch.object(owner,'lease',side_effect=lease),\
             patch('research.launch_h3_direct_conversation.run',side_effect=direct),\
             patch('research.run_h3_rtc_conversation.validate_evidence',return_value={}),\
             patch('research.h3_personal.webrtc_upstream_adapter.RemoteVoxCPMRTC',return_value=control):
            if unknown:
                with self.assertRaises(RuntimeError):owner.run(root,on_browser=lambda _:None,turns=['one','two'])
            else:owner.run(root,on_browser=lambda _:None,turns=['one','two'])
        return fence,json.loads((root/'result.json').read_text())

    def test_composition_endpoint_not_group_and_dual_cleanup(self):
        fence,result=self.exercise()
        self.assertFalse(fence.exists());self.assertTrue(result['cleanup_confirmed'])
        self.assertFalse(result['speech_input']);self.assertFalse(result['complete'])

    def test_unknown_h3_keeps_fence_even_when_endpoint_paused(self):
        fence,result=self.exercise(True)
        self.assertTrue(fence.exists());self.assertFalse(result['cleanup_confirmed'])

    def test_bad_input_no_resource(self):
        with patch.object(owner,'lease',side_effect=AssertionError('resource forbidden')):
            with self.assertRaises(ValueError):owner.run('/tmp/no',on_browser=None,turns=['a','b'])
            with self.assertRaises(ValueError):owner.run('/tmp/no',on_browser=lambda _:None,turns=None)

if __name__=='__main__':unittest.main()
