"""Offline composition; no deployment/output/capture from these tests."""
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
from research.run_h3_hybrid_conversation import run

class HybridTests(unittest.TestCase):
    def exercise(self,capture):
        config={'schema':'h3-native-voxcpm-network-v2'};factory=Mock(media_transport='webrtc-exact-v1')
        def pair(root,*,conversation,source_placement_gate):
            self.assertIs(source_placement_gate,placement)
            return conversation(Path(root),config)
        placement=Mock(return_value=True);browser=Mock()
        with patch('research.run_h3_network_pair.run',side_effect=pair),\
             patch('research.launch_h3_direct_conversation.run',return_value='done') as direct,\
             patch('research.h3_personal.asr_rtc_factory.ASRRTCFactory',return_value=factory) as constructor:
            result=run('/never-created',on_browser=browser,source_placement_gate=placement,
                **(dict(capture_comet=True,asr_parent=Path('/approved-parent')) if capture else dict(turns=['fresh1','fresh2'])))
        self.assertEqual(result,'done');self.assertIs(direct.call_args.kwargs['native_tts'],config)
        self.assertEqual(direct.call_args.kwargs['media_transport'],'webrtc-rtp-v1')
        self.assertFalse(direct.call_args.kwargs.get('continuous_av',False))
        if capture:
            constructor.assert_called_once();self.assertIs(direct.call_args.kwargs['asr_rtc_factory'],factory)
            self.assertIsNone(direct.call_args.args[1])
        else:constructor.assert_not_called();self.assertNotIn('asr_rtc_factory',direct.call_args.kwargs)
        browser.assert_not_called();placement.assert_not_called()

    def test_fresh_speech_uses_rtc_factory_not_legacy_wss(self):self.exercise(True)
    def test_typed_is_explicit_not_speech_e2e(self):self.exercise(False)
    def test_missing_gates_or_mixed_replay_rejected_before_resources(self):
        with patch('research.run_h3_network_pair.run',side_effect=AssertionError):
            with self.assertRaises(ValueError):run('/unused',on_browser=None,source_placement_gate=Mock(),turns=['1','2'])
            with self.assertRaises(ValueError):run('/unused',on_browser=Mock(),source_placement_gate=Mock(),capture_comet=True,
                                                 asr_parent=Path('/parent'),turns=['old','replay'])

if __name__=='__main__':unittest.main()
