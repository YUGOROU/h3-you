"""CPU policy tests only; not native visual-quality acceptance."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from research.h3_personal.live_native_backend import LiveNativeBackend
from research.run_h3_relay_conversation import proposal_reply, MODEL


def proposal(instruction=None):
    return dict(type='turn', turn_id='one', epoch=1, model=MODEL, baseline={},
                schema='h3-director-visual-turn-v1', reply='Hello',
                visual_intent=dict(schema='h3-visual-intent-v1', instruction=instruction),
                status='proposal_only', effects_executed=False,
                renderer_attached=False, audio_attached=False)


class StablePolicyTests(unittest.TestCase):
    def test_null_reply_allowed_but_movement_rejected(self):
        self.assertEqual(proposal_reply(proposal(), visual_intents=True, stable_selfie=True), 'Hello')
        for instruction in ('Turn right once and hold.', 'Return to the selfie view.'):
            with self.assertRaisesRegex(ValueError, 'forbids'):
                proposal_reply(proposal(instruction), visual_intents=True, stable_selfie=True)
            self.assertEqual(proposal_reply(proposal(instruction), visual_intents=True), 'Hello')

    def test_selection_is_strict(self):
        for kwargs in (dict(stable_selfie=True), dict(visual_intents=True, stable_selfie=1)):
            with self.assertRaises(ValueError):
                proposal_reply(proposal(), **kwargs)

    def test_backend_defense_precedes_intent_mutation(self):
        backend=LiveNativeBackend.__new__(LiveNativeBackend)
        backend.visual_intents=True; backend.stable_selfie=True
        backend._intent_lock=threading.Lock()
        backend._intent=None; backend._intent_epoch=0; backend._intent_target=0
        backend.visual_events=[]
        with self.assertRaisesRegex(ValueError, 'forbids'):
            backend.accept_visual_intent(1, proposal('Pan right')['visual_intent'])
        self.assertEqual(backend._intent_target, 0)
        self.assertEqual(backend.visual_events, [])
        backend.accept_visual_intent(1, proposal()['visual_intent'])
        self.assertIsNone(backend._intent)
        self.assertEqual(backend._intent_epoch, 0)
        self.assertEqual(backend._intent_target, 1)

    def test_backend_requires_conversation_anchor(self):
        with tempfile.TemporaryDirectory() as root:
            for visual, policy in ((True,None),(False,'conversation-tail-every3-v1')):
                runtime=NS(visual_intents=visual)
                with self.assertRaisesRegex(ValueError, 'requires'):
                    LiveNativeBackend(runtime, Path(root)/'not-created',
                                      visual_ancestry_policy=policy, stable_selfie=True)
                self.assertFalse((Path(root)/'not-created').exists())


if __name__ == '__main__':
    unittest.main()
