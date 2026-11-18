"""Attended visual capability propagation across immutable session, READY and worker.

Device-free: no process, socket, provider, GPU, capture or browser operation.
"""
import copy
import unittest

from test_h3_direct_conversation_spec import spec
from test_h3_native_tts_backend import config as native_tts
from research.hf_h3_direct_conversation import validate_spec
from research.h3_personal import live_relay_worker


def attended(**changes):
    """A valid immutable session on the attended visual route."""
    item = spec()
    row = item['live_direct_session']
    row.update(tts_mode='native_voxcpm', native_tts=native_tts(),
               visual_intents=True, attended_visual=True, turn_interruptions=False)
    row.update(changes)
    return item


class ImmutableSessionTests(unittest.TestCase):
    def test_attended_visual_without_interruption_is_accepted(self):
        row = validate_spec(attended())
        self.assertIs(row['attended_visual'], True)
        self.assertIs(row['visual_intents'], True)
        self.assertIs(row['turn_interruptions'], False)

    def test_headless_interruption_route_is_unchanged(self):
        item = attended()
        row = item['live_direct_session']
        row.update(attended_visual=False, turn_interruptions=True, continuous_av=True)
        accepted = validate_spec(item)
        self.assertIs(accepted['turn_interruptions'], True)
        self.assertIs(accepted['attended_visual'], False)

    def test_attended_and_interruption_are_exclusive(self):
        with self.assertRaises(ValueError):
            validate_spec(attended(turn_interruptions=True))

    def test_attended_requires_visual_intents(self):
        with self.assertRaises(ValueError):
            validate_spec(attended(visual_intents=False))

    def test_visual_without_any_route_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_spec(attended(attended_visual=False))

    def test_attended_rejects_periodic_anchor(self):
        with self.assertRaises(ValueError):
            validate_spec(attended(anchor_policy='idle-listening-v1'))

    def test_attended_rejects_interruption_only_features(self):
        for changes in (dict(paired_cuts=True), dict(native_tts_cancel=True),
                        dict(native_cancel_trigger='native-preroll-v1')):
            with self.assertRaises(ValueError):
                validate_spec(attended(**changes))

    def test_attended_requires_native_https_mode(self):
        item = attended()
        item['live_direct_session'].update(media_transport='webrtc-rtp-v1',
                                          ice_profile='hf-cloudflare-turn-tls-v1')
        with self.assertRaises(ValueError):
            validate_spec(item)

    def test_attended_flag_must_be_exact_bool(self):
        with self.assertRaises(ValueError):
            validate_spec(attended(attended_visual=1))

    def test_attended_belongs_only_to_the_immutable_session(self):
        item = attended()
        item['attended_visual'] = True
        with self.assertRaises(ValueError):
            validate_spec(item)


class WorkerAdmissionTests(unittest.TestCase):
    """The worker gate rejects attended misuse before any runtime/adapter use."""

    def reject(self, **kwargs):
        args = dict(runtime=None, tts=None, adapter=None, binding=None, context=None,
                    root=None, deadline=None)
        args.update(kwargs)
        return live_relay_worker.run_worker(**args)

    def test_attended_with_interruption_rejected(self):
        with self.assertRaisesRegex(ValueError, 'attended visual requires visual turns'):
            self.reject(visual_intents=True, attended_visual=True, turn_interruptions=True)

    def test_attended_without_visual_rejected(self):
        with self.assertRaisesRegex(ValueError, 'attended visual requires visual turns'):
            self.reject(visual_intents=False, attended_visual=True, turn_interruptions=False)

    def test_attended_selection_must_be_bool(self):
        with self.assertRaisesRegex(ValueError, 'explicit attended visual selection'):
            self.reject(visual_intents=True, attended_visual=1)

    def test_attended_rejects_paired_cuts_and_native_cancel(self):
        with self.assertRaisesRegex(ValueError, 'paired cuts require the interruption'):
            self.reject(visual_intents=True, attended_visual=True, paired_cuts=True)
        with self.assertRaises(ValueError):
            self.reject(visual_intents=True, attended_visual=True, native_tts_cancel=True)

    def test_attended_runtime_and_latest_tail_are_required(self):
        class Runtime:
            visual_intents = False
        with self.assertRaisesRegex(ValueError, 'attended runtime with latest-tail'):
            self.reject(runtime=Runtime(), visual_intents=True, attended_visual=True,
                        tts_mode='native_voxcpm')

    def test_attended_ancestry_policy_override_rejected(self):
        class Runtime:
            visual_intents = True
        with self.assertRaises(ValueError):
            self.reject(runtime=Runtime(), visual_intents=True, attended_visual=True,
                        tts_mode='native_voxcpm',
                        visual_ancestry_policy={'kind': 'periodic'})

    def test_interrupt_gate_is_absent_on_the_attended_route(self):
        import ast, inspect
        tree = ast.parse(inspect.getsource(live_relay_worker.run_worker))
        gates = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'turn_gate' for t in n.targets)]
        self.assertTrue(gates)
        rendered = ast.unparse(gates[0])
        self.assertEqual(rendered, 'TurnInterruptionGate() if turn_interruptions else None')

    def test_initial_listening_still_rejects_interrupt_controls(self):
        import inspect
        source = inspect.getsource(live_relay_worker.run_worker)
        self.assertIn("raise ValueError('turn interruption not enabled')", source)


if __name__ == '__main__':
    unittest.main()
