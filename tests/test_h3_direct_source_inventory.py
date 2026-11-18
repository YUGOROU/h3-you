from pathlib import Path
import tempfile
import unittest
from research.launch_h3_direct_conversation import first_party_sources


class SourceInventoryTests(unittest.TestCase):
    def test_nested_uv_and_installed_sources_never_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ('research/main.py', 'research/h3_personal/probe.py',
                     'research/webrtc_experiment/.venv/lib/private.py',
                     'research/.cache/secret.py', 'research/venv/pkg.py',
                     'research/site-packages/pkg.py', 'research/__pycache__/cached.py')
            for name in names:
                p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('')
            self.assertEqual({str(p.relative_to(root)) for p in first_party_sources(root)},
                             {'research/main.py','research/h3_personal/probe.py'})

    def test_first_party_symlink_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'research').mkdir()
            (root/'outside.py').write_text('')
            (root/'research/linked.py').symlink_to(root/'outside.py')
            with self.assertRaises(ValueError): list(first_party_sources(root))
