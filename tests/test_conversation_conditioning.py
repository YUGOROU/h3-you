import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from research.h3_personal import conversation_conditioning as c


class ConversationRolesTests(unittest.TestCase):
    def test_strict_manifest(self):
        self.assertEqual(c.validate_manifest(c.roles_manifest()), c.roles_manifest())
        for value in ({}, dict(c.roles_manifest(), extra=True),
                      dict(c.roles_manifest(), prompt_sha256={'speech': 'bad'})):
            with self.assertRaises(ValueError):
                c.validate_manifest(value)

    def test_current_role_and_context_binding(self):
        with tempfile.TemporaryDirectory() as root:
            plan, current = Path(root)/'plan', Path(root)/'current'
            plan.write_text(json.dumps(c.roles_manifest()))
            current.write_text(json.dumps(c.role_record('speech')))
            with patch.dict(os.environ, {c.ROLES_ENV: str(plan), c.CURRENT_ROLE_ENV: str(current)}):
                self.assertEqual(c.current_role(), 'speech')
                with c.bind_role('listening'):
                    self.assertEqual(c.current_role(), 'listening')
                self.assertEqual(c.current_role(), 'speech')
                current.write_text(json.dumps(dict(role='listening', prompt_sha256=c.prompt_sha('speech'))))
                with self.assertRaises(ValueError):
                    c.current_role()

    def test_effective_profile_role_hash_is_strict(self):
        from research.h3_personal.accepted_profiles import choose_profile
        profile = choose_profile(ordered_source_sha256=('a'*64,),
                                 prompt_sha256=c.prompt_sha('speech'))
        with tempfile.TemporaryDirectory() as root:
            plan, current = Path(root)/'plan', Path(root)/'current'
            plan.write_text(json.dumps(c.roles_manifest()))
            current.write_text(json.dumps(c.role_record('listening')))
            with patch.dict(os.environ, {c.ROLES_ENV: str(plan), c.CURRENT_ROLE_ENV: str(current)}):
                with self.assertRaisesRegex(ValueError, 'accepted/current role drift'):
                    c.effective_profile(profile)
                with c.bind_role('listening'):
                    effective = c.effective_profile(profile)
                    self.assertEqual(effective.prompt_sha256, c.prompt_sha('listening'))
                    self.assertNotEqual(effective.digest, profile.digest)
                self.assertEqual(profile.prompt_sha256, c.prompt_sha('speech'))

    def test_speech_unchanged(self):
        from research.h3_personal.ref2va_runtime import PROMPT
        self.assertEqual(c.SPEECH_PROMPT, PROMPT)
        self.assertNotEqual(c.prompt_sha('speech'), c.prompt_sha('listening'))
        with self.assertRaises(ValueError):
            c.role_prompt('unknown')
