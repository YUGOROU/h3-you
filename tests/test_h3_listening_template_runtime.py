"""CPU contract tests, not native runtime/visual acceptance."""
import copy
import unittest
from research.h3_personal import listening_templates as t
from research.h3_personal.conversation_conditioning import roles_manifest, LISTENING_PROMPT, SPEECH_PROMPT


class ListeningTemplateContractTests(unittest.TestCase):
    def setUp(self):
        self.policy = t.ListeningTemplatePolicy()

    def selection(self, sequence=0, role='listening'):
        frame = 0 if sequence == 0 else 124 + (sequence - 1) * 102
        return t.select(self.policy, role, frame, sequence)

    def test_default_manifest_unchanged(self):
        self.assertEqual(set(roles_manifest()), {'schema', 'prompt_sha256'})
        self.assertEqual(t.REGISTRY[0], ('baseline', LISTENING_PROMPT))
        self.assertEqual(len(t.REGISTRY), 3)
        self.assertIsNone(t.validate_authority(None, None, None))

    def test_paired_authority(self):
        declaration = self.policy.declaration()
        self.assertIs(t.validate_authority(self.policy, declaration, t.POLICY), self.policy)
        for args in ((None, None, t.POLICY), (self.policy, None, t.POLICY),
                     (self.policy, declaration, None), (None, declaration, None)):
            with self.assertRaises(ValueError):
                t.validate_authority(*args)
        with self.assertRaises(ValueError):
            t.ListeningTemplatePolicy(registry_sha256='0' * 64)

    def test_absolute_clock_speech_does_not_reset_phase(self):
        self.assertEqual(self.selection(3)['template_id'], 'baseline')
        self.assertEqual(self.selection(4)['template_id'], 'soft-attention')
        self.assertEqual(self.selection(4, 'speech')['prompt_sha256'], t.prompt_digest(SPEECH_PROMPT))
        self.assertEqual(self.selection(5)['template_id'], 'soft-attention')
        self.assertEqual(self.selection(8)['template_id'], 'gentle-attention')
        for frame, seq in ((0, 1), (124, 0), (True, 0)):
            with self.assertRaises(ValueError):
                t.select(self.policy, 'listening', frame, seq)

    def test_each_rank_actual_prompt_cache_and_selection_fail_closed(self):
        selection = self.selection(4)
        rows = [dict(rank=r, selection=dict(selection), actual_prompt_sha256=selection['prompt_sha256'],
                     cache_hit=True) for r in (0, 1)]
        kwargs = dict(policy=self.policy, role='listening', committed_frame=430, sequence=4)
        self.assertEqual(t.verify_rank_evidence(selection, rows, **kwargs), selection)
        for rank in (0, 1):
            for key, value in (('actual_prompt_sha256', '0' * 64), ('cache_hit', False),
                               ('rank', 1-rank), ('selection', self.selection(3))):
                bad = copy.deepcopy(rows)
                bad[rank][key] = value
                with self.assertRaises(ValueError):
                    t.verify_rank_evidence(selection, bad, **kwargs)
        for bad in ([], rows[:1], rows[::-1]):
            with self.assertRaises(ValueError):
                t.verify_rank_evidence(selection, bad, **kwargs)
        for key, value in (('template_id', 'unknown'), ('sequence', 3), ('registry_sha256', '0'*64)):
            bad = dict(selection, **{key: value})
            with self.assertRaises(ValueError):
                t.verify_rank_evidence(bad, rows, **kwargs)


if __name__ == '__main__':
    unittest.main()
