"""Actual relay invocation with fake I/O only; no capture/output/provider."""
from pathlib import Path
from test_h3_relay_attended_visual import Base, Director, Pump, Sink, relay
from unittest.mock import patch


class StableRelayTests(Base):
    def enable(self):
        self.ready.update(stable_selfie=True, anchor_policy='conversation-tail-every3-v1')
        self.director=Director(instruction=None)

    def test_two_fresh_turns_publish_null_with_bound_anchor(self):
        self.enable()
        result=self.attended(stable_selfie=True)
        self.assertTrue(result['complete'])
        turns=[row for row in self.transport.rows if row['kind']=='turn']
        self.assertEqual(len(turns),2)
        self.assertTrue(all(row['payload']['visual_intent']['instruction'] is None for row in turns))

    def test_movement_fails_before_publication(self):
        self.enable()
        self.director=Director(instruction='Pan right once and hold.')
        with self.assertRaisesRegex(ValueError,'forbids'):
            self.attended(stable_selfie=True)
        self.assertFalse(any(row['kind']=='turn' for row in self.transport.rows))

    def test_ready_tampering_is_rejected_before_director(self):
        self.enable()
        for key, value in (('anchor_policy','epoch-idle-every3-v1'),('stable_selfie',False)):
            original=self.ready[key]
            self.ready[key]=value
            with self.assertRaises(ValueError):
                self.attended(stable_selfie=True)
            self.ready[key]=original
        self.assertEqual(self.director.inputs,[])

    def test_typed_devicefree_without_interrupt_or_capture(self):
        self.enable()
        self.ready.pop('attended_visual')
        with patch.object(relay, 'HeadlessAVSink', Sink):
            result=relay.run_conversation(self.transport, self.binding, Path(self.temp.name),
                ['Hello', 'Continue'], self.director, ready=self.ready, visual_intents=True,
                stable_selfie=True, pump_factory=Pump, max_seconds=2)
        self.assertTrue(result['complete'])
        self.assertEqual(len(self.director.inputs), 2)

    def test_unselected_mode_cannot_accept_stable_ready(self):
        self.enable()
        with self.assertRaises(ValueError):
            self.attended()
