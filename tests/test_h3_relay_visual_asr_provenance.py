"""Visual publication must preserve input provenance; fake-only, no Pi/H3/devices."""
import ast
import copy
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest

from research.h3_personal.asr_source_contract import asr_origin
from research.h3_personal.live_relay_worker import PI_MODEL, validate_turn
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.zoom_input import source_contract
from research import run_h3_relay_conversation as relay
from research.run_h3_relay_conversation import visual_turn_payload


def intent(text):
    return dict(schema='h3-visual-intent-v1', instruction=text)


def expected_asr(revision='a'):
    return dict(policy='endpoint-native-v1', endpoint_name='h3-asr-t4-7c926aae18',
                revision=revision * 40, endpoint_source='native_completed',
                completion_policy='input-finalized', capture_stopped=True,
                cleanup_complete=True)


def zoom_asr():
    expected = dict(expected_asr(), input_source=source_contract(
        reference_sha256='b' * 64, capture_source_sha256='c' * 64))
    actual = dict(expected, zoom_capture=dict(
        schema='h3-zoom-asr-capture-v1',
        profile_sha256=expected['input_source']['profile_sha256'], input_epoch=1,
        captured_samples=16000, extracted_samples=16000, submitted_samples=16000,
        capture_frames=16384, capture_stopped=True, capture_cleanup=True, capture_app='zoom',
        mute_behavior='unmuted', eof=True, pending_samples=0, discarded_pending_samples=0,
        raw_audio_fallback=False, operator_confirmed_remote_idle=True))
    return expected, actual


def base_payload(provenance, reply='少し周りを見せるね。'):
    return dict(**provenance, director_model=PI_MODEL, reasoning_effort='none',
                reply=reply, previous_audio_reused=False)


def asr_provenance(asr):
    return dict(schema='h3-relay-asr-turn-v1', origin=asr_origin(asr),
                asr_used=True, asr=asr)


def typed_provenance():
    return dict(schema='h3-relay-typed-turn-v1', origin='fresh_typed_text', asr_used=False)


class PublicationProvenanceTests(unittest.TestCase):
    def test_asr_visual_turn_keeps_receipt_and_becomes_asr_v2(self):
        expected, actual = zoom_asr()
        original = base_payload(asr_provenance(actual))
        snapshot = copy.deepcopy(original)
        published = visual_turn_payload(original, visual_intent=intent('Hold the surroundings.'))
        # The pre-visual record is never mutated in place.
        self.assertEqual(original, snapshot)
        self.assertIsNot(published, original)
        self.assertEqual(published['schema'], 'h3-relay-asr-turn-v2')
        self.assertEqual(published['origin'], 'zoom_process_voxlock_endpoint_asr')
        self.assertIs(published['asr_used'], True)
        self.assertEqual(published['asr'], actual)
        self.assertEqual(published['visual_intent'], intent('Hold the surroundings.'))
        self.assertEqual(set(published) - {'visual_intent'}, set(snapshot))
        # The worker accepts exactly this record only with the bound expectation.
        self.assertEqual(validate_turn(dict(payload=published), expected, visual_intents=True),
                         '少し周りを見せるね。')
        # A typed-visual expectation for an ASR receipt is refused.
        with self.assertRaises(ValueError):
            validate_turn(dict(payload=published), expected, visual_intents=False)

    def test_comet_half_duplex_receipt_stays_comet_origin(self):
        actual = expected_asr()
        published = visual_turn_payload(base_payload(asr_provenance(actual)),
                                       visual_intent=intent(None))
        self.assertEqual(published['schema'], 'h3-relay-asr-turn-v2')
        self.assertEqual(published['origin'], 'comet_process_audio_endpoint_asr')
        self.assertEqual(validate_turn(dict(payload=published), actual, visual_intents=True),
                         '少し周りを見せるね。')
        # Null intent still means "continue the active direction", not neutral reset.
        self.assertIsNone(published['visual_intent']['instruction'])

    def test_typed_visual_turn_is_typed_v2_exactly(self):
        original = base_payload(typed_provenance())
        published = visual_turn_payload(original, visual_intent=intent('Pan slightly right once.'))
        self.assertEqual(published['schema'], 'h3-relay-typed-turn-v2')
        self.assertEqual(published['origin'], 'fresh_typed_text')
        self.assertIs(published['asr_used'], False)
        self.assertNotIn('asr', published)
        self.assertEqual(set(published) - {'visual_intent'}, set(original) | {'schema'})
        self.assertEqual(validate_turn(dict(payload=published), visual_intents=True),
                         '少し周りを見せるね。')

    def test_asr_turn_cannot_be_relabelled_as_typed_and_vice_versa(self):
        _, actual = zoom_asr()
        # A forged typed schema carrying an ASR receipt is rejected on both sides.
        forged = visual_turn_payload(base_payload(asr_provenance(actual)),
                                     visual_intent=intent('Hold.'))
        forged['schema'] = 'h3-relay-typed-turn-v2'
        with self.assertRaises(ValueError):
            validate_turn(dict(payload=forged), actual, visual_intents=True)
        with self.assertRaises(ValueError):
            visual_turn_payload(dict(base_payload(typed_provenance()), asr=actual),
                                visual_intent=intent('Hold.'))
        with self.assertRaises(ValueError):
            visual_turn_payload(dict(base_payload(asr_provenance(actual)), origin='fresh_typed_text'),
                                visual_intent=intent('Hold.'))

    def test_missing_or_mutated_receipt_fails_closed_before_publish(self):
        expected, actual = zoom_asr()
        with self.assertRaises(ValueError):
            visual_turn_payload(base_payload(dict(schema='h3-relay-asr-turn-v1',
                                                  origin='comet_process_audio_endpoint_asr',
                                                  asr_used=True)),
                                visual_intent=intent('Hold.'))
        mutated = copy.deepcopy(actual)
        mutated['cleanup_complete'] = 1
        with self.assertRaises(ValueError):
            visual_turn_payload(base_payload(asr_provenance(mutated)),
                                visual_intent=intent('Hold.'))
        drifted = copy.deepcopy(actual)
        drifted['revision'] = 'd' * 40
        published = visual_turn_payload(base_payload(asr_provenance(drifted)),
                                        visual_intent=intent('Hold.'))
        # Publish-side validation is structural; the worker still enforces the
        # immutable session expectation and rejects the drift.
        with self.assertRaises(ValueError):
            validate_turn(dict(payload=published), expected, visual_intents=True)

    def test_unversioned_or_ambiguous_input_is_rejected(self):
        _, actual = zoom_asr()
        for provenance in (dict(schema='h3-relay-asr-turn-v1', origin='x', asr_used=1, asr=actual),
                           dict(schema='h3-relay-typed-turn-v2', origin='fresh_typed_text', asr_used=False),
                           dict(schema='h3-relay-typed-turn-v1', origin='other', asr_used=False)):
            with self.subTest(provenance=provenance):
                with self.assertRaises(ValueError):
                    visual_turn_payload(base_payload(provenance), visual_intent=intent('Hold.'))
        with self.assertRaises(ValueError):
            visual_turn_payload(base_payload(typed_provenance()), visual_intent=dict(schema='x', instruction=None))
        with self.assertRaises(ValueError):
            visual_turn_payload(base_payload(typed_provenance()), visual_intent=intent(''))

    def test_publication_site_uses_the_provenance_preserving_helper(self):
        source = Path(relay.__file__).read_text()
        tree = ast.parse(source)
        run_conversation = next(n for n in tree.body
                                if isinstance(n, ast.FunctionDef) and n.name == 'run_conversation')
        calls = [n for n in ast.walk(run_conversation) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'visual_turn_payload']
        self.assertEqual(len(calls), 1)
        assigned = [n for n in ast.walk(run_conversation) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets)
                    and isinstance(n.value, ast.Call)
                    and getattr(n.value.func, 'id', None) == 'visual_turn_payload']
        self.assertEqual(len(assigned), 1)
        # Schema versioning for visual turns lives only in the helper: no
        # unconditional literal rewrite of a visual payload may return here.
        body = ast.get_source_segment(source, run_conversation)
        self.assertNotIn('payload.update(', body)
        self.assertNotIn('h3-relay-typed-turn-v2', body)
        self.assertNotIn('h3-relay-asr-turn-v2', body)


class LegacyPublicationTests(unittest.TestCase):
    """Non-visual ASR publication must stay byte-identical."""

    class Transport:
        _session = 'fresh'
        _role = 'consumer'
        _opened = True
        _closed = False
        _attested = True

        def __init__(self):
            self.rows = []
            self.done = threading.Event()

        def send_input(self, raw):
            n = struct.unpack('!I', raw[:4])[0]
            self.rows.append(json.loads(raw[4:4 + n]))
            if self.rows[-1]['kind'] == 'stop':
                self.done.set()
            return len(raw)

        def abort(self):
            self.done.set()

        def close(self):
            self._closed = True

    class Director:
        def __init__(self):
            self.inputs = []

        def turn(self, text):
            self.inputs.append(text)
            return dict(type='turn', turn_id='fixture-turn', epoch=1, model=relay.MODEL, reply='返事',
                        expression='neutral', motion='still', tone='calm',
                        status='proposal_only', effects_executed=False,
                        renderer_attached=False, audio_attached=False)

        def close(self):
            pass

    class Sink:
        instances = []

        def __init__(self, root, binding, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def cancel(self):
            pass

    class Pump:
        def __init__(self, transport, binding, root, sink, *, deadline):
            self.transport = transport

        def run(self):
            if not self.transport.done.wait(2):
                raise TimeoutError()
            return {'complete': True, 'frames': 1}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.transport = self.Transport()
        self.director = self.Director()
        self.binding = LiveAVBinding('fresh', 'a' * 64, 1, max_total_frames=248)
        self.ready = dict(status='ready', session='fresh', profile_sha256='a' * 64,
                          epoch=1, expires_at=time.time() + 60)

    def run_source(self, source, **kwargs):
        return relay.run_conversation(
            self.transport, self.binding, Path(self.temp.name), None, self.director,
            ready=self.ready, max_seconds=2, sink_factory=self.Sink,
            pump_factory=self.Pump, input_source=source, **kwargs)

    def test_legacy_asr_publication_is_unchanged_v1(self):
        actual = expected_asr()
        result = self.run_source(lambda **kw: dict(text='private transcript', asr=actual))
        self.assertTrue(result['complete'])
        self.assertEqual(self.director.inputs, ['private transcript'] * 2)
        payload = self.transport.rows[0]['payload']
        self.assertEqual(payload, dict(
            schema='h3-relay-asr-turn-v1', origin='comet_process_audio_endpoint_asr',
            asr_used=True, asr=actual, director_model=relay.MODEL, reasoning_effort='none',
            reply='返事', previous_audio_reused=False))
        self.assertEqual(validate_turn(dict(payload=payload), actual), '返事')
        self.assertNotIn('visual_intent', payload)
        self.assertNotIn('private transcript', json.dumps(self.transport.rows))

    def test_legacy_typed_publication_is_unchanged_v1(self):
        transport = self.Transport()
        director = self.Director()
        relay.run_conversation(
            transport, self.binding, Path(self.temp.name), ['一', '二'], director,
            ready=self.ready, max_seconds=2, sink_factory=self.Sink, pump_factory=self.Pump)
        payload = transport.rows[0]['payload']
        self.assertEqual(payload, dict(
            schema='h3-relay-typed-turn-v1', origin='fresh_typed_text', asr_used=False,
            director_model=relay.MODEL, reasoning_effort='none', reply='返事',
            previous_audio_reused=False))
        self.assertEqual(validate_turn(dict(payload=payload)), '返事')

    def test_headless_route_still_rejects_a_capture_input(self):
        with self.assertRaises(ValueError):
            self.run_source(lambda **kw: dict(text='x', asr=expected_asr()), headless_interrupt=True)
        self.assertEqual(self.director.inputs, [])


if __name__ == '__main__':
    unittest.main()
