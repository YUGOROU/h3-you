"""CPU data/prompt contract only; no Pi inference or native/model success claim."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from research.h3_personal.conversation_conditioning import (
    VISUAL_INTENT_SCHEMA, is_declared_prompt, role_prompt, resolve_visual_intent,
    validate_visual_intent, visual_prompt,
)
from research.h3_personal.fast_ref2va import FastRef2VAConfig


def intent(instruction):
    return dict(schema=VISUAL_INTENT_SCHEMA, instruction=instruction)


class VisualIntentContractTests(unittest.TestCase):
    def test_version_and_bounds_without_actions_or_completion_fields(self):
        self.assertEqual(validate_visual_intent(intent('  Show the surroundings.  ')),
                         intent('Show the surroundings.'))
        self.assertEqual(validate_visual_intent(intent(None)), intent(None))
        self.assertEqual(validate_visual_intent(intent('😀'*256)), intent('😀'*256))
        for bad in ('', ' ', 'x'*257, 'a\nb', '\x7f', '\ud800', 1, False, []):
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                validate_visual_intent(intent(bad))
        for extra in ('achieved', 'clip', 'device', 'reference'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_visual_intent(dict(intent(None), **{extra: True}))
        with self.assertRaises(ValueError):
            validate_visual_intent(dict(intent(None), schema='future'))

    def test_continue_is_not_reset_and_return_is_an_explicit_new_direction(self):
        showing = resolve_visual_intent(None, intent('Turn away from the face; show the surroundings.'))
        kept = resolve_visual_intent(showing, intent(None))
        self.assertEqual(kept, showing)
        self.assertIsNot(kept, showing)
        returned = resolve_visual_intent(kept, intent('Return to the self-facing view and continue the discussion.'))
        self.assertNotEqual(returned, showing)
        self.assertIn('Turn away', showing['instruction'])
        self.assertIsNone(resolve_visual_intent(None, intent(None)))
        with self.assertRaises(ValueError):
            resolve_visual_intent(intent(None), intent('Return.'))

    def test_no_visual_direction_retains_exact_legacy_prompts(self):
        for role in ('speech', 'listening'):
            self.assertEqual(visual_prompt(role, None), role_prompt(role))
            FastRef2VAConfig(prompt=role_prompt(role)).validate()
        with self.assertRaises(ValueError):
            visual_prompt('other', None)

    def test_candidate_composition_preserves_identity_not_fixed_composition(self):
        active = intent('Rotate the handheld view to show the surroundings while the face is out of frame.')
        for role in ('speech', 'listening'):
            prompt = visual_prompt(role, active)
            self.assertIn('reference image 1', prompt)
            self.assertIn('identity', prompt)
            self.assertIn('same physical setting', prompt)
            self.assertIn('face may leave the frame', prompt)
            self.assertIn(active['instruction'], prompt)
            self.assertNotIn('Stable lighting and background.', prompt)
            self.assertNotIn('speaking to the camera', prompt)
        self.assertIn('listening without speaking', visual_prompt('listening', active))
        with self.assertRaises(ValueError):
            visual_prompt('speech', intent(None))

    def test_finite_adjustment_holds_same_body_across_roles_and_null(self):
        active = resolve_visual_intent(None, intent(
            'Adjust the selfie view slightly to the left once, then hold that composition.'))
        continued = resolve_visual_intent(active, intent(None))
        for role in ('speech', 'listening'):
            prompt = visual_prompt(role, continued)
            self.assertIn('same single person', prompt)
            self.assertIn('same body and its current position and orientation', prompt)
            self.assertIn('rather than reinserting the reference pose', prompt)
            self.assertIn('move once and then hold', prompt)
            self.assertIn(active['instruction'], prompt)
            self.assertNotIn('wearing a plain white shirt', prompt)
        # A fresh owner has no prior movement; null is never converted into reset.
        self.assertIsNone(resolve_visual_intent(None, intent(None)))
        self.assertEqual(continued, active)

    def test_candidate_is_not_silently_admitted_by_existing_native_guard(self):
        # This fails closed UNTIL a versioned prompt/cache/native proof extension
        # is actually wired. A fixture must not imply native application.
        candidate = visual_prompt('speech', intent('Turn away and show the surroundings.'))
        self.assertFalse(is_declared_prompt(candidate))
        with self.assertRaises(ValueError):
            FastRef2VAConfig(prompt=candidate).validate()

    @unittest.skipUnless(shutil.which('node'), 'existing Node required')
    def test_python_and_existing_js_tool_contract_agree_on_wire_data(self):
        cases = [intent(s) for s in (None, 'Show the surroundings.', '戻して、話を続けよう。',
                 '😀'*256, '😀'*257, '', ' ', 'x'*257, 'a\nb', '\ud800', 1, False, [])]
        cases.extend([dict(intent(None), schema='future'), dict(intent(None), achieved=True)])
        code = """import {validateVisualIntent} from './research/director/contract.mjs';
let input=''; for await (const b of process.stdin) input+=b;
console.log(JSON.stringify(JSON.parse(input).map(v=>{try{return {ok:true,value:validateVisualIntent(v)}}catch{return {ok:false}}})));"""
        result = subprocess.run([shutil.which('node'), '--input-type=module', '-e', code],
            cwd=Path(__file__).resolve().parents[1], input=json.dumps(cases),
            text=True, capture_output=True, timeout=5, check=True)
        expected = []
        for case in cases:
            try:
                expected.append(dict(ok=True, value=validate_visual_intent(case)))
            except ValueError:
                expected.append(dict(ok=False))
        self.assertEqual(json.loads(result.stdout), expected)


if __name__ == '__main__':
    unittest.main()
