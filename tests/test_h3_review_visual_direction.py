import hashlib
import json
from pathlib import Path
import stat
import tempfile
import time
import unittest

from research.h3_personal.headless_av_sink import ReviewArchive
from research.h3_personal.conversation_conditioning import VISUAL_INTENT_SCHEMA


class Tests(unittest.TestCase):
    def test_private_text_and_exact_role_hashes_continue_without_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = ReviewArchive(Path(tmp).resolve()/'review', time.monotonic()+20)
            active = dict(schema=VISUAL_INTENT_SCHEMA,
                          instruction='Return to the same person holding the phone, with their face visible again.')
            first = archive.capture_visual_direction(proposed=active, active=active,
                                                     intent_epoch=1, target_epoch=1)
            continued = archive.capture_visual_direction(
                proposed=dict(schema=VISUAL_INTENT_SCHEMA, instruction=None), active=active,
                intent_epoch=1, target_epoch=2)
            self.assertIsNone(continued['metadata']['proposed']['instruction'])
            for entry in (first, continued):
                self.assertEqual(entry['kind'], 'private-visual-direction')
                self.assertIn('not_native', entry['metadata']['stage'])
                for role, data in entry['metadata']['roles'].items():
                    self.assertEqual(data['binding']['role'], role)
                    self.assertEqual(data['binding']['prompt_sha256'],
                                     hashlib.sha256(data['prompt'].encode()).hexdigest())
                    self.assertIn(active['instruction'], data['prompt'])
                for file in entry['files']:
                    self.assertEqual(stat.S_IMODE((archive.root/file['name']).stat().st_mode), 0o600)
            self.assertEqual(len(archive.snapshot()['entries']), 2)
            self.assertNotIn(active['instruction'], json.dumps(archive.snapshot()))
            saved = json.loads((archive.root/'review-index.json').read_text())
            self.assertEqual(saved['entries'][0]['metadata']['roles']['speech']['binding']['visual_intent'], active)
            with self.assertRaises(ValueError):
                archive.capture_visual_direction(proposed=active, active=active,
                                                 intent_epoch=1, target_epoch=2)
            self.assertEqual(len(archive.snapshot()['entries']), 2)
            unicode_archive = ReviewArchive(Path(tmp).resolve()/'unicode-review', time.monotonic()+20)
            unicode_active = dict(schema=VISUAL_INTENT_SCHEMA, instruction='🙂'*256)
            unicode_archive.capture_visual_direction(proposed=unicode_active, active=unicode_active,
                                                    intent_epoch=1, target_epoch=1)
            self.assertEqual(len(unicode_archive.snapshot()['entries']), 1)

    def test_shared_cap_rollback_no_partial_private_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = ReviewArchive(Path(tmp).resolve()/'review', time.monotonic()+20, max_bytes=1024)
            active = dict(schema=VISUAL_INTENT_SCHEMA, instruction='Turn back to the same person.')
            with self.assertRaisesRegex(ValueError, 'byte cap'):
                archive.capture_visual_direction(proposed=active, active=active,
                                                 intent_epoch=1, target_epoch=1)
            self.assertEqual(archive.snapshot()['entries'], [])
            self.assertEqual([x.name for x in archive.root.iterdir()], ['review-index.json'])
            self.assertEqual(json.loads((archive.root/'review-index.json').read_text())['entries'], [])


if __name__ == '__main__': unittest.main()
