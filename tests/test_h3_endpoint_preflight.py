import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

from research import run_h3_endpoint_preflight as m

# macOS /var is a symlink; use a canonical parent for fresh-root tests.
TEMP_PARENT = str(Path(tempfile.gettempdir()).resolve())


def row(**changes):
    value = dict(vendor='aws', region='ap-northeast-1', accelerator='gpu',
                 instance_type='nvidia-h200', instance_size='x2', num_accelerators=2,
                 status='available', used_accelerators=0, max_accelerators=8,
                 price_per_hour=999, headers={'Authorization': 'secret'})
    value.update(changes)
    return value


class EndpointPreflightTests(unittest.TestCase):
    def test_offline_no_side_effects(self):
        def forbidden(*a, **kw):
            self.fail('offline side effect')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main([], runner=forbidden, token_getter=forbidden), 0)
        self.assertFalse(m.preflight()['queried'])

    def test_region_status_quota_and_exact_pair(self):
        self.assertEqual(len(m.sanitize([row()])), 1)
        for vendor, regions in m.REGIONS.items():
            for region in regions:
                self.assertEqual(len(m.sanitize([row(vendor=vendor, region=region)])), 1)
        for changes in [dict(region='us-west-2'), dict(region='us-south1'),
                        dict(region='ap-northeast-2'), dict(region='eu-west-1'),
                        dict(region='jp'), dict(vendor='unknown'), dict(status='deprecated'),
                        dict(status='unavailable'), dict(used_accelerators=7),
                        dict(max_accelerators=None), dict(used_accelerators=True),
                        dict(instance_size='x1'), dict(instance_size='x4'),
                        dict(num_accelerators=1), dict(instance_type='nvidia-h100')]:
            with self.subTest(changes=changes):
                self.assertEqual(m.sanitize([row(**changes)]), [])
        self.assertEqual(len(m.sanitize([row(used_accelerators=6)])), 1)

    def test_query_private_sanitized_and_env_only(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as parent:
            root = Path(parent) / 'fresh'
            original = os.environ.get('HF_TOKEN')
            calls = []
            def runner(command, *, env, timeout):
                calls.append(command)
                self.assertEqual(command, m.COMMAND)
                self.assertNotIn('hf_private_test', command)
                self.assertEqual(env['HF_TOKEN'], 'hf_private_test')
                self.assertEqual(timeout, 30)
                return subprocess.CompletedProcess(command, 0, json.dumps([row()]), 'raw secret')
            result = m.preflight(query=True, directory=root, runner=runner,
                                 token_getter=lambda: 'hf_private_test')
            self.assertTrue(result['queried'])
            self.assertEqual(len(calls), 1)
            snapshot = root / 'endpoint-catalog.json'
            text = snapshot.read_text()
            for secret in ('price', 'headers', 'Authorization', 'secret', 'hf_private_test'):
                self.assertNotIn(secret, text)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(snapshot.stat().st_mode), 0o600)
            self.assertEqual(os.environ.get('HF_TOKEN'), original)
            with self.assertRaises(m.PreflightError):
                m.preflight(query=True, directory=root, runner=runner)

    def test_directory_guards(self):
        def forbidden():
            self.fail('token accessed before path validation')
        with tempfile.TemporaryDirectory() as parent:
            link = Path(parent) / 'link'
            link.symlink_to(Path(parent) / 'missing')
            for directory in (None, parent, link, Path(__file__).resolve().parents[1] / 'never-create-endpoint'):
                with self.subTest(directory=directory), self.assertRaises(m.PreflightError):
                    m.preflight(query=True, directory=directory, token_getter=forbidden)

    def test_failure_does_not_persist_raw_data(self):
        for response in [subprocess.CompletedProcess([], 1, 'hf_secret', 'headers raw error'),
                         subprocess.CompletedProcess([], 0, 'not json hf_secret'),
                         subprocess.CompletedProcess([], 0, 'x' * (m.MAX_BYTES + 1)),
                         subprocess.CompletedProcess([], 0, '{}')]:
            with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as parent:
                root = Path(parent) / 'fresh'
                with self.assertRaises(m.PreflightError) as caught:
                    m.preflight(query=True, directory=root, token_getter=lambda: 'hf_secret',
                                runner=lambda *a, **kw: response)
                self.assertNotIn('hf_secret', str(caught.exception))
                self.assertEqual(list(root.iterdir()), [])
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as parent:
            def broken(*a, **kw):
                raise RuntimeError('hf_secret raw headers')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = m.main(['--query', '--directory', str(Path(parent) / 'fresh')],
                              runner=broken, token_getter=lambda: 'hf_secret')
            self.assertEqual(code, 1)
            self.assertNotIn('hf_secret', output.getvalue())

    def test_real_runner_local_timeout_and_output_bound(self):
        with self.assertRaises(m.PreflightError):
            m.bounded_runner([sys.executable, '-c', 'import time; time.sleep(5)'],
                             env=dict(os.environ), timeout=0.1)
        with self.assertRaises(m.PreflightError):
            m.bounded_runner([sys.executable, '-c',
                              'import sys; sys.stderr.write("x" * 2000000)'],
                             env=dict(os.environ), timeout=5)


if __name__ == '__main__':
    unittest.main()
