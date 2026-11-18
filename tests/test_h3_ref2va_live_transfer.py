import json
import os
from threading import Event
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
from research.h3_personal.ref2va_live_transfer import publisher,NAME


class LiveTransferTests(unittest.TestCase):
    def test_only_four_generated_names(self):
        self.assertTrue(NAME.fullmatch('segment-003.mp4'))
        for name in ('../segment-000.mp4','segment-004.mp4','reference.png','profile.json'):
            self.assertIsNone(NAME.fullmatch(name))

    def test_child_online_parent_stays_offline(self):
        with patch.dict(os.environ,{'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}):
            with patch('subprocess.run',return_value=NS(returncode=0,stdout=json.dumps({'revision':'abc'}))) as run:
                result=publisher('owner/private')({'path':'/private/segment-000.mp4'},Event())
                self.assertEqual(result['remote_publication']['revision'],'abc')
                self.assertEqual(result['path'],'/private/segment-000.mp4')
                self.assertNotIn('HF_HUB_OFFLINE',run.call_args.kwargs['env'])
                self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
                self.assertEqual(run.call_args.kwargs['timeout'],90)

    def test_canceled_never_publishes(self):
        event=Event();event.set()
        with patch('subprocess.run') as run:
            with self.assertRaises(RuntimeError):publisher('private')({},event)
            run.assert_not_called()

    def test_failure_no_retry(self):
        with patch('subprocess.run',return_value=NS(returncode=1,stdout='')) as run:
            with self.assertRaises(RuntimeError):publisher('private')({'path':'segment-000.mp4'},Event())
            self.assertEqual(run.call_count,1)

if __name__=='__main__':unittest.main()
