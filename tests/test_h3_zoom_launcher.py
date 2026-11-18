import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

from research import launch_h3_zoom_poc as z
from research.h3_personal.zoom_input import source_contract
from research.h3_personal.zoom_turn_input import ZoomTurnInput

class LauncherTests(unittest.TestCase):
    def test_offline_no_devices_models_or_network(self):
        with patch.object(z,'build_input',side_effect=AssertionError('no model load')):
            out=io.StringIO()
            with contextlib.redirect_stdout(out):self.assertEqual(z.main([]),0)
        row=json.loads(out.getvalue());self.assertTrue(row['offline']);self.assertFalse(row['devices_accessed'])
        self.assertIn('HTTP',row['output']);self.assertIn('BlackHole2ch',row['output'])
    def test_missing_approval_fails_before_native_or_provider(self):
        with patch.object(z,'build_input') as build:
            with self.assertRaises(PermissionError):
                z.run(directory='/tmp/new',asr_parent='/tmp',model_dir='/tmp')
        build.assert_not_called()
    def test_unattended_refuses_even_if_flags_set(self):
        with patch('sys.stdin.isatty',return_value=False),patch.object(z,'build_input') as build:
            with self.assertRaises(PermissionError):
                z.run(directory='/tmp/new',asr_parent='/tmp',model_dir='/tmp',all_participants_consented=True,
                      capture_approved=True,allow_obs=True,obs_port=48124)
        build.assert_not_called()
    def test_missing_tail_validator_refuses_before_native_or_allocations(self):
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'must-not-exist'
            with patch('sys.stdin.isatty',return_value=True),patch.object(z,'build_input') as build:
                with patch('research.launch_h3_direct_conversation.run') as direct:
                    with self.assertRaisesRegex(PermissionError, 'echo-tail validator required before allocation'):
                        z.run(directory=output,asr_parent=root,model_dir=root,
                              all_participants_consented=True,capture_approved=True,
                              allow_obs=True,obs_port=48124)
            build.assert_not_called()
            direct.assert_not_called()
            self.assertFalse(output.exists())
    def test_check_local_is_not_capture_permission(self):
        p=Mock(source_contract={'schema':'fixture'})
        with patch.object(z,'build_input',return_value=p) as build:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(z.main(['--check-local','--model-dir','/tmp']),0)
        row=json.loads(out.getvalue());self.assertFalse(row['capture_started']);self.assertFalse(row['provider_started'])
        self.assertNotIn('all_participants_consented',build.call_args.kwargs)
    def test_original_launcher_conflicting_sources_reject_preflight(self):
        from research.launch_h3_direct_conversation import run
        p=ZoomTurnInput(object(),source_contract(reference_sha256='a'*64,capture_source_sha256='b'*64))
        with self.assertRaises(ValueError):run('/tmp/new',capture_comet=True,zoom_input=p)
