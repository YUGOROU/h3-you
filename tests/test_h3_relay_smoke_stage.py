import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from research.run_h3_relay_smoke import prepare,EXPECTED_FILES,IMAGE


class StageTests(unittest.TestCase):
    def test_fresh_private_code_only_no_credentials_or_models(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'stage';plan=prepare(root)
            self.assertEqual(plan['hardware'],'cpu-basic');self.assertTrue(plan['private'])
            self.assertFalse(plan['gpu']);self.assertFalse(plan['model_loaded'])
            self.assertEqual({r['path'] for r in plan['files']},EXPECTED_FILES)
            for row in plan['files']:
                body=(root/'deployment'/row['path']).read_bytes()
                self.assertEqual(hashlib.sha256(body).hexdigest(),row['sha256'])
                self.assertEqual((root/'deployment'/row['path']).stat().st_mode & 0o777,0o400)
            docker=(root/'deployment/Dockerfile').read_text()
            self.assertIn('@sha256:',IMAGE);self.assertNotIn('pip install',docker)
            self.assertNotIn('apt-get',docker)
            self.assertNotIn('producer_key',json.dumps(plan))
            with self.assertRaises(FileExistsError):prepare(root)
