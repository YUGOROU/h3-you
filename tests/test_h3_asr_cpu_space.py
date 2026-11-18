"""Offline Space staging and CPU-only security contract; no deployment or devices."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from research.launch_h3_asr_cpu_space import stage, SOURCE


class CPUSpaceTests(unittest.TestCase):
    def test_private_free_finite_immutable_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'new'
            plan, deployment = stage(root)
            self.assertIs(plan['private'], True)
            self.assertEqual(plan['hardware'], 'cpu-basic')
            self.assertEqual(plan['hourly_price_usd'], '0')
            self.assertEqual(plan['max_server_seconds'], 1800)
            self.assertEqual(stage(root)[0], plan)
            (deployment/'Dockerfile').write_text('changed')
            with self.assertRaises(ValueError):
                stage(root)

    def test_visibility_or_hardware_changes_rejected(self):
        for changes in (dict(private=False), dict(hardware='cpu-upgrade')):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'new'
                plan, _ = stage(root)
                plan.update(changes)
                (root/'plan.json').write_text(json.dumps(plan))
                with self.assertRaises(ValueError):
                    stage(root)

    def test_no_gpu_tokens_capture_or_unbounded_restart(self):
        docker = (SOURCE/'Dockerfile').read_text()
        self.assertIn('@sha256:', docker)
        self.assertEqual(docker.count('sha256sum -c -'), 2)
        self.assertNotIn('HF_TOKEN', docker)
        self.assertIn('USER app', docker)
        script = (SOURCE/'start.sh').read_text()
        for required in ('H3_ASR_EXPIRES_AT', '1800', 'timeout --signal=TERM',
                         '--device cpu', '--no-ui', '--max-upload-mb 2'):
            self.assertIn(required, script)
        self.assertIn('--asr.batching.enabled=false', script)
        self.assertIn('--asr.endpointing.enable=true', script)
        self.assertNotIn('--asr.batching.enabled false', script)
        self.assertNotIn('--asr.endpointing.enable true', script)
        for forbidden in ('--live', '--access-log', '--open', '--api-key', 'while true'):
            self.assertNotIn(forbidden, script)
        subprocess.run(['sh', '-n', str(SOURCE/'start.sh')], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
