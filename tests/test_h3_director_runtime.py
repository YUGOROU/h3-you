import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from decimal import Decimal
from research.h3_personal.director_runtime import child_environment,reserve,reconcile,MODEL


class DirectorRuntimeTests(unittest.TestCase):
    def test_only_inject_token_in_child_env_no_debug_inheritance(self):
        with patch.dict(os.environ,{'DEBUG':'*','NODE_OPTIONS':'--inspect','OTHER_SECRET':'private'}):
            env=child_environment('fake-test-key',1)
        self.assertEqual(env['HF_TOKEN'],'fake-test-key')
        for key in ('DEBUG','NODE_OPTIONS','OTHER_SECRET'):self.assertNotIn(key,env)

    def test_bound_reconciles_without_prompt_or_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'budget.json';path.write_text(json.dumps({'spent_usd':169.35,'reservations':{}}))
            key,bound=reserve(path,1);self.assertEqual(bound,Decimal('.20'))
            result=reconcile(path,key,{'calls':2,'unknown_calls':0,'input':1700,'output':100})
            self.assertEqual(result['runtime_bound_usd'],'0.01')
            self.assertEqual(json.loads(path.read_text())['reservations'],{})
            self.assertNotIn('prompt',json.dumps(result))

    def test_unknown_calls_keep_conservative_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'budget.json';path.write_text(json.dumps({'spent_usd':169.35,'reservations':{}}))
            key,_=reserve(path,1);result=reconcile(path,key,None)
            self.assertEqual(result['runtime_bound_usd'],'0.20')

    def test_cap_and_finite_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            from research.h3_personal.ref2va_budget import HF_CAP
            path=Path(folder)/'budget.json';path.write_text(json.dumps({'spent_usd':str(HF_CAP-Decimal('.19')),'reservations':{}}))
            with self.assertRaises(ValueError):reserve(path,1)
            with self.assertRaises(ValueError):reserve(path,9)

    def test_exact_model_and_no_persisted_key(self):
        config=json.loads(Path('research/director/models.json').read_text())['providers']['h3-hf-cerebras']
        self.assertEqual(config['apiKey'],'$HF_TOKEN');self.assertEqual(config['models'][0]['id'],MODEL)
        self.assertEqual(config['models'][0]['samplingParams']['reasoning_effort'],'none')
        source=Path('research/director/agent.mjs').read_text()
        self.assertIn("noTools:'builtin'",source);self.assertIn('SessionManager.inMemory',source)
        self.assertIn('getAgentsFiles:()=>({agentsFiles:[]})',source)

if __name__=='__main__':unittest.main()
