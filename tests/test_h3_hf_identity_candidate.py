import json
from pathlib import Path
import tempfile
import unittest
from research.build_hf_identity_candidate import build


class CandidateTests(unittest.TestCase):
    def test_preserves_hf_approval_objective_preview_and_enables_escrow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'lock.json').write_text('{}')
            (root / 'runtime-manifest.json').write_text(json.dumps(dict(
                schema='h3-identity-gpu-runtime-bootstrap-v1', image='pinned-image',
                bootstrap_files={}, artifact_files={}, documents={'runtime_lock': {'path': 'lock.json'}},
                module='research.hf_restore_identity_runtime', entrypoint={'path': 'entry.py', 'sha256': 'a'*64})))
            old = dict(schema='h3-identity-launch-spec-v1', image='pinned-image', documents={},
                runtime_bootstrap={}, pipeline=dict(territory_approval={'provider': 'huggingface_jobs'},
                runtime_provenance={}, inference={'sample_ids': ['synthetic'], 'seconds': 2700},
                objective_marker='unchanged', max_updates=2000))
            path = root / 'old.json'
            path.write_text(json.dumps(old))
            spec = build(path, runtime_mount=root, runtime_repo='test/private', runtime_revision='b'*40,
                         rate_usd_hour=4.99998)
            self.assertEqual(spec['pipeline']['territory_approval'], old['pipeline']['territory_approval'])
            self.assertEqual(spec['pipeline']['objective_marker'], 'unchanged')
            self.assertEqual(spec['pipeline']['inference'], old['pipeline']['inference'])
            self.assertTrue(spec['pipeline']['checkpoint_escrow'])
            self.assertEqual(spec['pipeline']['checkpoint_interval'], 10)
            self.assertEqual(spec['pipeline']['overall_seconds'], 16200)
            with self.assertRaises(ValueError):
                build(path, runtime_mount=root, runtime_repo='test/private', runtime_revision='b'*40,
                      rate_usd_hour=6)


if __name__ == '__main__': unittest.main()
