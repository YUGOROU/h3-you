import unittest

from research.h3_personal.conversation_route import HTTPSConversationRoute
from research.h3_personal.voxcpm_endpoint_protocol import PINS


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.route = HTTPSConversationRoute('a'*40, 'b'*40, True)
        ready = dict(PINS, session='session', boot='boot', epoch=0, expires_at=123)
        self.args = dict(vox_revision='a'*40, asr_revision='b'*40,
                         native_tts={'schema': 'h3-native-voxcpm-backend-v1'},
                         vox_ready=ready, expected_vox_ready=dict(ready),
                         output_transport='http-paired-v1', continuous_av=True)

    def test_reuses_existing_adapter_without_constructing(self):
        from research.h3_personal.voxcpm_endpoint_client import RemoteVoxCPM
        self.assertIs(self.route.admit(**self.args), RemoteVoxCPM)
        self.assertFalse(self.route.inspect()['asr_https_body_streaming'])
        self.assertFalse(self.route.inspect()['rtc_fallback'])

    def test_asr_exception_closed_by_default(self):
        with self.assertRaisesRegex(ValueError, 'user confirmation'):
            HTTPSConversationRoute('a'*40, 'b'*40).admit(**self.args)

    def test_revision_backend_output_drift(self):
        for key, value in (
            ('vox_revision', 'c'*40), ('asr_revision', 'c'*40),
            ('native_tts', {'schema': 'h3-native-voxcpm-rtc-v3'}),
            ('native_tts', {'schema': 'h3-native-voxcpm-network-v2'}),
            ('output_transport', 'webrtc-rtp-v1'), ('continuous_av', False)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.route.admit(**dict(self.args, **{key: value}))

    def test_ready_mismatch_and_rtc_extension(self):
        for key, value in (('boot', 'old'), ('epoch', False),
                           ('media_transport', 'webrtc-exact-v1'),
                           ('model_revision', 'wrong')):
            ready = dict(self.args['vox_ready'], **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.route.admit(**dict(self.args, vox_ready=ready))

    def evidence(self):
        return {k: v for k, v in self.args.items()
                if k not in ('native_tts', 'output_transport', 'continuous_av')}

    def test_launch_composes_existing_https_flags(self):
        from unittest.mock import Mock
        runner = Mock(return_value='owned-result')
        result = self.route.launch('/unused', turns=['fresh text'],
                                  native_tts=self.args['native_tts'],
                                  evidence=self.evidence(), runner=runner)
        self.assertEqual(result, 'owned-result')
        kwargs = runner.call_args.kwargs
        for flag in ('continuous_av', 'http_keepalive', 'overlap_send',
                     'prefetch_receive', 'conversation_anchor'):
            self.assertIs(kwargs[flag], True)
        self.assertEqual(kwargs['media_transport'], 'http-paired-v1')
        self.assertFalse(kwargs['capture_comet'])
        self.assertNotIn('asr_rtc_factory', kwargs)

    def test_bad_evidence_never_reaches_runner(self):
        from unittest.mock import Mock
        runner = Mock()
        evidence = dict(self.evidence(), vox_revision='c'*40)
        with self.assertRaises(ValueError):
            self.route.launch('/unused', turns=['text'], evidence=evidence,
                              native_tts=self.args['native_tts'], runner=runner)
        runner.assert_not_called()

    def test_capture_rejects_rtc_parent_before_runner(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'intent.json'
            for revision, env in (
                ('b'*40, {'H3_ASR_MEDIA_TRANSPORT': 'webrtc-exact-v1'}),
                ('c'*40, {})):
                path.write_text(json.dumps({'model': {'revision': revision, 'env': env}}))
                runner = Mock()
                with self.assertRaises(ValueError):
                    self.route.launch('/unused', capture_comet=True, asr_parent=directory,
                                      evidence=self.evidence(),
                                      native_tts=self.args['native_tts'], runner=runner)
                runner.assert_not_called()

    def test_immutable_revision_required(self):
        for revision in ('main', '', 'A'*40, None):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                HTTPSConversationRoute(revision, 'b'*40)


if __name__ == '__main__':
    unittest.main()
