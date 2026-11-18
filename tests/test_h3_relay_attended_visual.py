"""Attended visual routing in the existing relay runner.

Fake-only: no Pi child, H3/Vox provider, browser, capture, device, network or GPU
operation is started. ASSR receipts are synthetic versioned records.
"""
import ast
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock as unittest_mock

from research.h3_personal.live_relay_worker import PI_MODEL, validate_turn
from research.h3_personal.paired_av import LiveAVBinding
from research import run_h3_relay_conversation as relay

BINARY = Path('/bin/echo')
PIN = dict(audio_binary='/bin/echo', binary_sha256='b' * 64, audio_delay_seconds=0.2)


def expected_asr(revision='a'):
    return dict(policy='endpoint-native-v1', endpoint_name='h3-asr-t4-7c926aae18',
                revision=revision * 40, endpoint_source='native_completed',
                completion_policy='input-finalized', capture_stopped=True,
                cleanup_complete=True)


def intent(instruction):
    return dict(schema='h3-visual-intent-v1', instruction=instruction)


class Transport:
    _session = 'fresh'
    _role = 'consumer'
    _opened = True
    _closed = False
    _attested = True

    def __init__(self):
        self.rows = []
        self.sent = threading.Event()
        self.done = threading.Event()

    def send_input(self, raw):
        n = struct.unpack('!I', raw[:4])[0]
        row = json.loads(raw[4:4 + n])
        self.rows.append(row)
        self.sent.set()
        if row['kind'] == 'stop':
            self.done.set()
        return len(raw)

    def abort(self):
        self.done.set()

    def close(self):
        self._closed = True


class Director:
    """Visual proposal child stand-in; never a real Pi/model call."""

    def __init__(self, instruction='Hold the neutral handheld selfie view.'):
        self.inputs = []
        self.instruction = instruction
        self.deadline = 0
        self.closed = False

    def turn(self, text):
        self.inputs.append(text)
        return dict(type='turn', turn_id='fixture-turn', epoch=len(self.inputs),
                    model=relay.MODEL, baseline='h3-visual-intent-v1-unverified',
                    schema='h3-director-visual-turn-v1', reply='少し右を見るね。',
                    visual_intent=intent(self.instruction), status='proposal_only',
                    effects_executed=False, renderer_attached=False, audio_attached=False)

    def close(self):
        self.closed = True


class LegacyDirector(Director):
    """Legacy reply-only proposal (no visual schema)."""

    def turn(self, text):
        self.inputs.append(text)
        return dict(type='turn', turn_id='fixture-turn', epoch=len(self.inputs),
                    model=relay.MODEL, reply='返事', expression='neutral', motion='still',
                    tone='calm', status='proposal_only', effects_executed=False,
                    renderer_attached=False, audio_attached=False)


class Sink:
    """Stands in for the caller-owned local browser sink."""

    instances = []

    def __init__(self, root, binding, **kwargs):
        self.kwargs = kwargs
        self.first_av_submission_monotonic = None
        self.cancelled = False
        self.instances.append(self)

    def cancel(self):
        self.cancelled = True


class Pump:
    def __init__(self, transport, binding, root, sink, *, deadline):
        self.transport = transport

    def run(self):
        if not self.transport.done.wait(5):
            raise TimeoutError()
        return {'complete': True, 'frames': 1}


class Presentation:
    """Stand-in for the caller's sink per-turn presentation evidence source."""

    session = 'fresh'

    def __init__(self):
        self.waited = []
        self.stopped = None

    def wait_for_presentation_complete(self, epoch, deadline):
        self.waited.append(epoch)
        return dict(schema='h3-sink-presentation-complete-v1', session='fresh',
                    epoch=epoch, completed_at_monotonic=time.monotonic(),
                    output_seconds=0.5)

    def stop(self, reason):
        self.stopped = reason
        return True


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        Sink.instances = []
        patcher = unittest.mock.patch.object(relay, 'direct_audio_pin', lambda *a, **k: dict(PIN))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.transport = Transport()
        self.director = Director()
        self.binding = LiveAVBinding('fresh', 'a' * 64, 1, max_total_frames=248)
        self.ready = dict(status='ready', session='fresh', profile_sha256='a' * 64,
                          epoch=1, expires_at=time.time() + 60,
                          visual_intents=True, attended_visual=True)

    def source(self, asr=None, text='private transcript'):
        def call(**kwargs):
            return dict(text=text, asr=asr or expected_asr())
        return call

    def attended(self, source=None, **kwargs):
        # The verified device route: Comet ASR input plus split OBS video and PCM audio.
        args = dict(ready=self.ready, visual_intents=True, attended_visual=True,
                    input_source=source or self.source(), max_seconds=2,
                    allow_obs=True, obs_port=48124,
                    direct_audio_binary=BINARY, audio_delay_seconds=0.2,
                    sink_factory=Sink, pump_factory=Pump)
        args.update(kwargs)
        return relay.run_conversation(self.transport, self.binding,
                                      Path(self.temp.name), None, self.director, **args)


class AttendedVisualAcceptedTests(Base):
    def test_attended_visual_runs_two_asr_turns_without_headless(self):
        result = self.attended()
        self.assertTrue(result['complete'])
        self.assertEqual(len(self.director.inputs), 2)
        kinds = [row['kind'] for row in self.transport.rows]
        self.assertEqual(kinds, ['turn', 'turn', 'stop'])
        # No synthetic onset/interruption control is fabricated for attended use.
        self.assertNotIn('interrupt_turn', kinds)
        self.assertNotIn('cancel_pending', kinds)
        for row in self.transport.rows:
            if row['kind'] != 'turn':
                continue
            payload = row['payload']
            self.assertEqual(payload['schema'], 'h3-relay-asr-turn-v2')
            self.assertEqual(payload['origin'], 'comet_process_audio_endpoint_asr')
            self.assertIs(payload['asr_used'], True)
            self.assertEqual(payload['asr'], expected_asr())
            self.assertEqual(payload['visual_intent'], intent(self.director.instruction))
            # The existing worker validates exactly this record with the bound receipt.
            self.assertEqual(validate_turn(dict(payload=payload), expected_asr(),
                                           visual_intents=True), '少し右を見るね。')

    def test_caller_browser_sink_is_used_without_headless_output_observation(self):
        self.attended()
        self.assertEqual(len(Sink.instances), 1)
        kwargs = Sink.instances[0].kwargs
        self.assertNotIn('observe_output', kwargs)
        self.assertNotIn('paced', kwargs)
        self.assertEqual(kwargs['max_frames'], 124)

    def test_attended_does_not_start_the_turn_interruption_controller(self):
        self.attended()
        source = Path(relay.__file__).read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == 'run_conversation')
        guarded = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.If) or 'controller' not in ast.unparse(node.test):
                continue
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr in ('begin_input', 'publish_reply',
                                                'wait_for_speech_output')):
                    guarded.append(inner.func.attr)
        # Every interruption-only call stays inside a controller-guarded branch, so
        # attended turns (controller is None) send plain turn rows only.
        self.assertIn('begin_input', guarded)
        self.assertIn('publish_reply', guarded)
        self.assertIn('wait_for_speech_output', guarded)

    def test_attended_null_intent_is_retained_not_reset(self):
        self.director.instruction = None
        self.attended()
        for row in self.transport.rows:
            if row['kind'] == 'turn':
                self.assertIsNone(row['payload']['visual_intent']['instruction'])


class AttendedVisualRejectedTests(Base):
    def test_attended_and_headless_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            self.attended(headless_interrupt=True)
        self.assertEqual(self.director.inputs, [])

    def test_visual_without_any_route_is_rejected(self):
        with self.assertRaises(ValueError):
            self.attended(attended_visual=False)

    def test_attended_requires_bound_attended_ready(self):
        for ready_change in (dict(attended_visual=False), dict(attended_visual=True),
                             dict(attended_visual=None)):
            ready = dict(self.ready)
            if ready_change.get('attended_visual') is True:
                ready.pop('attended_visual')
            else:
                ready.update(ready_change)
            with self.subTest(ready=ready):
                with self.assertRaises(ValueError):
                    self.attended(ready=ready)
        self.assertEqual(self.director.inputs, [])

    def test_attended_requires_visual_ready_binding(self):
        for ready in ({}, dict(self.ready, visual_intents=False)):
            with self.subTest(ready=ready):
                with self.assertRaises(ValueError):
                    self.attended(ready=ready)

    def test_attended_rejects_periodic_anchor_policy(self):
        with self.assertRaises(ValueError):
            self.attended(ready=dict(self.ready, anchor_policy='fixed-idle-anchor-v1'))

    def test_attended_rejects_native_preroll_trigger(self):
        with self.assertRaises(ValueError):
            self.attended(native_cancel_trigger='native-preroll-v1')
        with self.assertRaises(ValueError):
            self.attended(native_progress=lambda *a, **k: None, native_preroll_samples=4800)
        with self.assertRaises(ValueError):
            self.attended(native_preroll_samples=4800)

    def test_attended_rejects_review_archive_cuts(self):
        with self.assertRaises(ValueError):
            self.attended(review_av=True,
                          ready=dict(self.ready, paired_cuts=True))

    def test_attended_requires_the_verified_split_obs_output(self):
        for change in (dict(allow_obs=False), dict(obs_port=None)):
            with self.subTest(change=change):
                with self.assertRaises((ValueError, PermissionError)):
                    self.attended(**change)

    def test_attended_requires_a_real_input_source(self):
        with self.assertRaises(ValueError):
            self.attended(input_source=None)

    def test_attended_flag_must_be_explicit_bool(self):
        for value in (1, 'yes', None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.attended(attended_visual=value)


class UnchangedRouteTests(Base):
    def test_typed_visual_turn_stays_typed_v2(self):
        provenance = dict(schema='h3-relay-typed-turn-v1', origin='fresh_typed_text',
                          asr_used=False, director_model=relay.MODEL,
                          reasoning_effort='none', reply='返事', previous_audio_reused=False)
        published = relay.visual_turn_payload(provenance, visual_intent=intent('Hold.'))
        self.assertEqual(published['schema'], 'h3-relay-typed-turn-v2')
        self.assertIs(published['asr_used'], False)
        self.assertNotIn('asr', published)
        self.assertEqual(validate_turn(dict(payload=published), visual_intents=True), '返事')

    def test_headless_route_still_rejects_attended_shapes(self):
        # Headless keeps its device-free typed contract; attended inputs are refused.
        with self.assertRaises(ValueError):
            self.attended(headless_interrupt=True, ready=dict(self.ready, attended_visual=False))
        with self.assertRaises(ValueError):
            relay.run_conversation(
                self.transport, self.binding, Path(self.temp.name), None, self.director,
                ready=dict(self.ready, attended_visual=False), max_seconds=2,
                visual_intents=True, headless_interrupt=True,
                input_source=self.source(), pump_factory=Pump)

    def test_legacy_non_visual_route_is_untouched(self):
        ready = dict(status='ready', session='fresh', profile_sha256='a' * 64,
                     epoch=1, expires_at=time.time() + 60)
        relay.run_conversation(
            self.transport, self.binding, Path(self.temp.name), ['一', '二'], LegacyDirector(),
            ready=ready, max_seconds=2, sink_factory=Sink, pump_factory=Pump)
        for row in self.transport.rows:
            if row['kind'] == 'turn':
                self.assertEqual(row['payload']['schema'], 'h3-relay-typed-turn-v1')
                self.assertNotIn('visual_intent', row['payload'])

    def test_legacy_non_visual_attended_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            self.attended(visual_intents=False)

    def test_attended_keeps_private_transcript_out_of_publication(self):
        self.attended(source=self.source(text='very private words'))
        self.assertNotIn('very private words', json.dumps(self.transport.rows))


if __name__ == '__main__':
    unittest.main()
