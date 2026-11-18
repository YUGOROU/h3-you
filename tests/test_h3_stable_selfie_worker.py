"""Static attended or typed selfie is explicit, anchor-bound and movement closed."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from test_h3_attended_visual_propagation import attended
from research.hf_h3_direct_conversation import validate_spec
from research.h3_personal.live_relay_worker import run_worker, validate_turn, _visual_window_evidence, PI_MODEL

POLICY = 'conversation-tail-every3-v1'


def turn(instruction):
    return {'payload': dict(schema='h3-relay-typed-turn-v2', origin='fresh_typed_text',
        asr_used=False, director_model=PI_MODEL, reasoning_effort='none', reply='こんにちは。',
        previous_audio_reused=False, visual_intent=dict(
            schema='h3-visual-intent-v1', instruction=instruction))}


class StableSelfieTests(unittest.TestCase):
    def test_spec_accepts_exact_static_mode(self):
        row = validate_spec(attended(stable_selfie=True, anchor_policy=POLICY))
        self.assertIs(row['stable_selfie'], True)
        self.assertEqual(row['anchor_policy'], POLICY)

    def test_spec_accepts_devicefree_typed_static_mode(self):
        item = attended(attended_visual=False, stable_selfie=True, anchor_policy=POLICY)
        row = validate_spec(item)
        self.assertIs(row['attended_visual'], False)
        self.assertIs(row['visual_intents'], True)
        self.assertIs(row['turn_interruptions'], False)
        self.assertIs(row['stable_selfie'], True)
        self.assertNotIn('expected_asr', row)

    def test_spec_devicefree_static_rejects_asr_or_interruption_features(self):
        for changes in (dict(expected_asr={}), dict(native_tts_cancel=True),
                        dict(native_cancel_trigger='native-preroll-v1'),
                        dict(paired_cuts=True), dict(turn_interruptions=True),
                        dict(visual_intents=False), dict(long_session=True)):
            values = dict(attended_visual=False, stable_selfie=True, anchor_policy=POLICY)
            values.update(changes)
            with self.assertRaises(ValueError):
                validate_spec(attended(**values))

    def test_devicefree_worker_reaches_native_preroll_gate_without_io(self):
        with self.assertRaisesRegex(ValueError, 'immutable PCMPrerollPolicy'):
            run_worker(SimpleNamespace(visual_intents=True), None, None,
                binding=None, context=None, root=None, deadline=None,
                stable_selfie=True, attended_visual=False, visual_intents=True,
                visual_ancestry_policy=POLICY, tts_mode='native_voxcpm')

    def test_devicefree_worker_rejects_asr_before_io(self):
        with self.assertRaisesRegex(ValueError, 'stable selfie requires'):
            run_worker(None, None, None, binding=None, context=None,
                root=None, deadline=None, expected_asr={},
                stable_selfie=True, attended_visual=False, visual_intents=True,
                visual_ancestry_policy=POLICY, tts_mode='native_voxcpm')

    def test_spec_rejects_missing_or_wrong_anchor(self):
        for value in (None, 'epoch-idle-every3-v1'):
            item = attended(stable_selfie=True)
            if value is not None:
                item['live_direct_session']['anchor_policy'] = value
            with self.assertRaises(ValueError):
                validate_spec(item)

    def test_spec_rejects_incompatible_or_untyped_selection(self):
        for changes in (dict(stable_selfie=1), dict(attended_visual=1),
                        dict(visual_intents=False), dict(turn_interruptions=True),
                        dict(paired_cuts=True)):
            values = dict(stable_selfie=True, anchor_policy=POLICY)
            values.update(changes)
            with self.assertRaises(ValueError):
                validate_spec(attended(**values))
        item = attended()
        item['stable_selfie'] = True
        with self.assertRaises(ValueError):
            validate_spec(item)

    def test_default_visual_still_rejects_anchor(self):
        with self.assertRaises(ValueError):
            validate_spec(attended(anchor_policy=POLICY))
        validate_spec(attended())

    def test_worker_rejects_bad_static_binding_before_io(self):
        for changes in (dict(visual_ancestry_policy=None), dict(visual_intents=False),
                        dict(turn_interruptions=True), dict(paired_cuts=True),
                        dict(stable_selfie=1)):
            kwargs = dict(stable_selfie=True, attended_visual=True, visual_intents=True,
                visual_ancestry_policy=POLICY, tts_mode='native_voxcpm')
            kwargs.update(changes)
            with self.assertRaises(ValueError):
                run_worker(None, None, None, binding=None, context=None,
                    root=None, deadline=None, **kwargs)

    def test_null_only_static_turn(self):
        self.assertEqual(validate_turn(turn(None), visual_intents=True, stable_selfie=True), 'こんにちは。')
        for text in ('Turn right.', 'Hold the new camera position.'):
            with self.assertRaisesRegex(ValueError, 'forbids movement'):
                validate_turn(turn(text), visual_intents=True, stable_selfie=True)
            validate_turn(turn(text), visual_intents=True)

    def test_worker_passes_static_selection_to_native_backend(self):
        source = Path('research/h3_personal/live_relay_worker.py').read_text()
        self.assertIn("if stable_selfie:options['stable_selfie'] = True", source)
        self.assertIn("backend=LiveNativeBackend(runtime,root/'native',**options)", source)
        self.assertIn("if stable_selfie and getattr(backend, 'stable_selfie', False) is not True:", source)

    def test_ready_and_worker_echo_immutable_static_selection(self):
        source = Path('research/hf_h3_direct_conversation.py').read_text()
        self.assertIn('ready_options.update(stable_selfie=True, anchor_policy=CONVERSATION_ANCHOR_POLICY)', source)
        self.assertIn("if row.get('stable_selfie', False):worker_options['stable_selfie'] = True", source)

    def test_stable_evidence_exports_actual_scalar_selection(self):
        anchor = 'a' * 64
        for sequence in range(7):
            selection = dict(policy=POLICY, sequence=sequence,
                kind='unconditioned' if sequence == 0 else 'anchor' if sequence % 3 == 0 else 'latest',
                anchor_sha256=None if sequence == 0 else anchor,
                selected_sha256=None if sequence == 0 else anchor if sequence == 1 or sequence % 3 == 0 else 'b' * 64,
                anchor_sequence=None if sequence == 0 else 1)
            receipt = dict(sequence=sequence, visual_conditioning=dict(intent_epoch=0),
                visual_ancestry_selection=selection)
            result = _visual_window_evidence(receipt, stable_selfie=True, anchor_sha256=anchor)
            self.assertEqual(result['visual_ancestry_selection'], selection)
            self.assertIsNot(result['visual_ancestry_selection'], selection)
            self.assertNotIn('visual_ancestry_selection', _visual_window_evidence(receipt))

    def test_stable_evidence_rejects_missing_or_tampered_selection(self):
        receipt = dict(sequence=3, visual_conditioning=dict(intent_epoch=0))
        with self.assertRaises(ValueError):
            _visual_window_evidence(receipt, stable_selfie=True, anchor_sha256='a' * 64)
        selection = dict(policy=POLICY, sequence=3, kind='anchor', anchor_sha256='a' * 64,
            selected_sha256='b' * 64, anchor_sequence=1)
        receipt['visual_ancestry_selection'] = selection
        with self.assertRaises(ValueError):
            _visual_window_evidence(receipt, stable_selfie=True, anchor_sha256='a' * 64)
        selection['selected_sha256'] = 'a' * 64
        selection['private_payload'] = 'not allowed'
        with self.assertRaises(ValueError):
            _visual_window_evidence(receipt, stable_selfie=True, anchor_sha256='a' * 64)

    def test_event_and_return_both_use_same_selection_projection(self):
        source = Path('research/h3_personal/live_relay_worker.py').read_text()
        calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == '_visual_window_evidence']
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIn('stable_selfie', {keyword.arg for keyword in call.keywords})
            self.assertIn('anchor_sha256', {keyword.arg for keyword in call.keywords})

    def test_every_worker_turn_validation_gets_static_selection(self):
        path = Path('research/h3_personal/live_relay_worker.py')
        calls = [node for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'validate_turn']
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertIn('stable_selfie', {keyword.arg for keyword in call.keywords})


if __name__ == '__main__':
    unittest.main()
