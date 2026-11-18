"""CPU-only diagnostic policy and native preparation plumbing checks."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research.h3_personal.idle_anchor_policy import (
    DIAGNOSTIC_POLICY, ENVIRONMENT_VARIABLE, parse_idle_anchor_policy,
)


class IdleAnchorPolicyTests(unittest.TestCase):
    def test_exact_policy(self):
        self.assertIsNone(parse_idle_anchor_policy(None))
        self.assertEqual(parse_idle_anchor_policy(DIAGNOSTIC_POLICY), 3)

    def test_reject_every_other_value(self):
        for value in ('', '3', 'off', ' '+DIAGNOSTIC_POLICY,
                      DIAGNOSTIC_POLICY+'\n', DIAGNOSTIC_POLICY.upper(),
                      3, True, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_idle_anchor_policy(value)

    def prepare(self, value, state=None):
        # Execute the real nested preparation without importing GPU dependencies.
        path=Path(__file__).resolve().parents[1]/'research/h3_personal/streaming_native_stages.py'
        tree=ast.parse(path.read_text())
        append=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='append_conditions')
        prep=next(n for n in append.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
        prep.body=[ast.Global(names=n.names) if isinstance(n,ast.Nonlocal) else n for n in prep.body]
        module=ast.fix_missing_locations(ast.Module(body=[prep],type_ignores=[]))
        calls=[]
        def owner(*args, **kwargs):
            calls.append(kwargs)
            return NS(_sequence=0,committedF=0,active_native=None,emittedF=0)
        scope=dict(__name__='research.h3_personal.streaming_native_stages',
            __package__='research.h3_personal',
            state=state,m={},stage=NS(ref2va=True),batch=NS(extra={}),
            references=[NS(media_type='image',has_audio=False,num_audio_latents=0,
                           num_latent_frames=1,latent_height=2,latent_width=2)],
            audio_rows=[],video_rows=['video'],
            legacy=NS(_geometry=lambda b:(2,2),KEY='key',_source_hash=lambda m:None,
                      TorchOps=lambda:NS(digest=lambda v:'digest')),
            c=dict(id='id',profile_sha256='sha',max_windows=12,sequence=0,committed_before=0),
            StreamingWindowOwner=owner)
        env={} if value is None else {ENVIRONMENT_VARIABLE:value}
        with patch.dict(os.environ,env,clear=True):
            exec(compile(module,str(path),'exec'),scope)
            scope['prepare']()
        return scope['state'],calls

    def test_default_shape_and_constructor_unchanged(self):
        state,calls=self.prepare(None)
        self.assertEqual(len(state['binding']),6)
        self.assertNotIn('idle_anchor_interval',calls[0])

    def test_active_binding_and_constructor(self):
        state,calls=self.prepare(DIAGNOSTIC_POLICY)
        self.assertEqual(len(state['binding']),7)
        self.assertEqual(state['binding'][-1],3)
        self.assertEqual(calls[0]['idle_anchor_interval'],3)
        _,calls=self.prepare(DIAGNOSTIC_POLICY,state)
        self.assertEqual(calls,[])

    def test_policy_drift_rejected_both_directions(self):
        for before,after in ((None,DIAGNOSTIC_POLICY),(DIAGNOSTIC_POLICY,None)):
            state,_=self.prepare(before)
            with self.assertRaisesRegex(ValueError,'clock drift'):
                self.prepare(after,state)

    def test_invalid_environment_rejected(self):
        with self.assertRaisesRegex(ValueError,'H3_IDLE_ANCHOR_DIAGNOSTIC'):
            self.prepare('')


if __name__=='__main__':
    unittest.main()
