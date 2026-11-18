import json
import struct
import unittest
from research.h3_personal.live_relay_commands import Commands


class InputLane:
    def __init__(self):self.body=bytearray()
    def send_input(self,body):self.body.extend(body);return len(body)
    def read_input(self):
        value=bytes(self.body[:7]);del self.body[:7];return value


class CommandTests(unittest.TestCase):
    def test_fragmented_coalesced_commands_and_eos(self):
        lane=InputLane();tx=Commands(lane,'session');rx=Commands(lane,'session')
        tx.send('turn',1,{'text':'synthetic'});tx.send('cancel_pending',1,{})
        tx.send('turn',2,{'text':'second'});tx.send('stop',2,{})
        self.assertEqual([rx.receive()['kind'] for _ in range(4)],['turn','cancel_pending','turn','stop'])
        with self.assertRaises(RuntimeError):rx.receive()
    def test_stale_epoch_and_huge_payload_before_submit(self):
        lane=InputLane();tx=Commands(lane,'session');tx.send('turn',1,{})
        size=len(lane.body)
        for kind,epoch,payload in [('turn',1,{}),('stop',0,{}),('turn',True,{}),('turn',2,{'text':'a'*20000})]:
            with self.assertRaises(ValueError):tx.send(kind,epoch,payload)
        self.assertEqual(len(lane.body),size)
    def test_bad_received_sequence_is_terminal(self):
        lane=InputLane();rx=Commands(lane,'session')
        value=dict(schema='h3-live-command-v1',session='session',sequence=2,kind='turn',epoch=1,payload={})
        body=json.dumps(value).encode();lane.body.extend(struct.pack('!I',len(body))+body)
        with self.assertRaises(ValueError):rx.receive()
        with self.assertRaises(RuntimeError):rx.receive()
