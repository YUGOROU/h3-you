import contextlib
import io
import fcntl
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.run_h3_irodori_endpoint import deployment, pause_owned, ControlError, main
from research.h3_personal.director_tts_job import IMAGE


class EndpointOwnerTests(unittest.TestCase):
    def test_l4_auth_image_pin_and_replica_bound(self):
        r=deployment('YUGOROU/test','a'*40,'owned','session',1234)
        self.assertEqual(r['provider'],dict(vendor='aws',region='us-east-1'))
        self.assertEqual(r['compute']['instanceType'],'nvidia-l4')
        self.assertEqual(r['compute']['scaling'],dict(minReplica=1,maxReplica=1))
        self.assertEqual(r['model']['image']['custom']['url'],IMAGE)
        self.assertEqual(r['type'],'authenticated')
        self.assertNotIn('secrets',r['model'])

    def test_pause_first_even_if_first_operation_fails(self):
        with patch('research.run_h3_irodori_endpoint.control',side_effect=[OSError(),{'status':{'state':'paused'}}]) as call:
            with patch('research.run_h3_irodori_endpoint.time.sleep'):
                self.assertEqual(pause_owned('TOKEN','/owned'),'paused')
        self.assertEqual(call.call_count,2)
        self.assertEqual(call.call_args_list[0].args[1],'/owned/pause')
        self.assertEqual(call.call_args_list[1].args[1],'/owned')

    def test_pause_verifies_get(self):
        with patch('research.run_h3_irodori_endpoint.control',side_effect=[{'status':{'state':'pending'}},{'status':{'state':'paused'}}]) as call:
            self.assertEqual(pause_owned('TOKEN','/owned'),'paused')
        self.assertEqual(call.call_args_list[0].args[1],'/owned/pause')
        self.assertEqual(call.call_args_list[1].args[1],'/owned')

    def test_uncertain_create_404_is_unconfirmed_not_success(self):
        with patch('research.run_h3_irodori_endpoint.control',side_effect=ControlError(404)) as call:
            with patch('research.run_h3_irodori_endpoint.time.sleep'):
                with self.assertRaises(RuntimeError):pause_owned('TOKEN','/owned')
        self.assertEqual(call.call_count,6)

    def test_unconfirmed_pause_is_error_not_success(self):
        with patch('research.run_h3_irodori_endpoint.control',side_effect=OSError()) as call:
            with patch('research.run_h3_irodori_endpoint.time.sleep'):
                with self.assertRaises(RuntimeError):pause_owned('TOKEN','/owned')
        self.assertEqual(call.call_count,6)

    def test_owner_lock_prevents_parallel_launch(self):
        from research.run_h3_irodori_endpoint import run
        with tempfile.TemporaryDirectory() as tmp:
            lock=Path(tmp)/'owner.lock'
            fd=os.open(lock,os.O_CREAT|os.O_WRONLY,0o600)
            try:
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('research.run_h3_irodori_endpoint.OWNER_LOCK',lock):
                    with patch('research.run_h3_irodori_endpoint._run_owned') as owned:
                        with self.assertRaises(BlockingIOError):run(Path(tmp)/'attempt')
                        owned.assert_not_called()
            finally:os.close(fd)

    def test_unresolved_guard_blocks_new_launch(self):
        from research.run_h3_irodori_endpoint import run
        with tempfile.TemporaryDirectory() as tmp:
            guard=Path(tmp)/'unresolved.json';guard.write_text('{}')
            with patch('research.run_h3_irodori_endpoint.OWNER_LOCK',Path(tmp)/'owner.lock'):
                with patch('research.run_h3_irodori_endpoint.OWNER_GUARD',guard):
                    with patch('research.run_h3_irodori_endpoint._run_owned') as owned:
                        with self.assertRaises(RuntimeError):run(Path(tmp)/'attempt')
                        owned.assert_not_called()

    def test_staging_materializes_namespace_packages_without_root_init(self):
        from research.run_h3_irodori_endpoint import FILES,copy_runtime
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';artifact=Path(tmp)/'artifact';artifact.mkdir()
            for name in (*FILES,'research/endpoints/irodori_l4/entrypoint.py'):
                if name.endswith('/__init__.py'):continue
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# source')
            copy_runtime(artifact,root)
            self.assertEqual((artifact/'research/__init__.py').read_bytes(),b'')
            self.assertEqual((artifact/'research/h3_personal/__init__.py').read_bytes(),b'')
            self.assertTrue((artifact/'entrypoint.py').is_file())
            self.assertFalse((root/'research/__init__.py').exists())

    def test_offline_default_does_not_call_provider(self):
        with patch('research.run_h3_irodori_endpoint.control') as call:
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main([]),0)
        call.assert_not_called()
