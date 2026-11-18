import json
import os
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
from research.h3_personal.conversation_producer import SpokenLLM,HF_ROUTER_ENDPOINT,HF_ROUTER_MODEL


class ChatOfflineIsolationTests(unittest.TestCase):
    def test_offline_model_parent_not_mutated_online_only_child(self):
        with patch.dict(os.environ,{'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DEBUG':'1'}), \
             patch.dict('sys.modules',huggingface_hub=N(InferenceClient=None)), \
             patch('subprocess.run',return_value=N(returncode=0,stdout=json.dumps({'content':'こんにちは！','finish_reason':'stop'}))) as run:
            self.assertEqual(SpokenLLM(HF_ROUTER_ENDPOINT,HF_ROUTER_MODEL,allow_remote=True,key='synthetic').reply('hi',[]),'こんにちは！')
            self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
            self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'],'1')
            args=run.call_args.kwargs
            self.assertEqual(args['env']['HF_HUB_OFFLINE'],'0')
            self.assertEqual(args['env']['TRANSFORMERS_OFFLINE'],'0')
            self.assertEqual(args['env']['HF_DEBUG'],'0')
            self.assertNotIn('synthetic',args['input'])
            self.assertTrue(args['capture_output']);run.assert_called_once()

    def test_failed_child_does_not_echo_stderr_or_retry(self):
        with patch.dict(os.environ,{'HF_HUB_OFFLINE':'1'}), \
             patch.dict('sys.modules',huggingface_hub=N(InferenceClient=None)), \
             patch('subprocess.run',return_value=N(returncode=1,stdout='',stderr='secret')) as run:
            with self.assertRaises(RuntimeError) as e:
                SpokenLLM(HF_ROUTER_ENDPOINT,HF_ROUTER_MODEL,allow_remote=True,key='synthetic').reply('hi',[])
            self.assertNotIn('secret',str(e.exception));run.assert_called_once()
