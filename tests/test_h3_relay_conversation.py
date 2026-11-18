import json
from pathlib import Path
import struct
import subprocess
from unittest.mock import patch
import tempfile
import threading
import time
import unittest

from research.run_h3_relay_conversation import run_conversation, MODEL, PiDirector, proposal_reply


def rpc_proposal(reply='新しい返事'):
    # agent.mjs emit({type, turn_id, epoch, model, baseline, ...state.draft,
    # status, effects_executed, renderer_attached, audio_attached}).
    return dict(type='turn', turn_id='fixture-turn', epoch=1, model=MODEL,
                baseline=dict(candidate='v12-portrait', reference='original-only',
                              canvas=[256, 448], export=[252, 448], forwards=4,
                              visualPrompt='accepted-v12-unchanged'),
                reply=reply, expression='neutral', motion='still', tone='calm',
                status='proposal_only', effects_executed=False,
                renderer_attached=False, audio_attached=False)
from research.h3_personal.paired_av import LiveAVBinding


class Transport:
    _session = 'fresh'
    _role = 'consumer'
    _opened = True
    _closed = False
    _attested = True

    def __init__(self):
        self.rows = []
        self.done = threading.Event()
        self.aborted = False

    def send_input(self, raw):
        n = struct.unpack('!I', raw[:4])[0]
        self.rows.append(json.loads(raw[4:4+n]))
        if self.rows[-1]['kind'] == 'stop':
            self.done.set()
        return len(raw)

    def abort(self):
        self.aborted = True
        self.done.set()

    def close(self):
        self._closed = True


class Director:
    def __init__(self, fail=False):
        self.inputs = []
        self.closed = False
        self.fail = fail

    def turn(self, text):
        self.inputs.append(text)
        if self.fail:
            raise RuntimeError('private details')
        return rpc_proposal()

    def close(self):
        self.closed = True


class Sink:
    instances = []

    def __init__(self, root, binding, **kwargs):
        self.kwargs = kwargs
        self.canceled = False
        self.instances.append(self)

    def cancel(self):
        self.canceled = True


class Pump:
    def __init__(self, transport, binding, root, sink, *, deadline):
        self.transport = transport

    def run(self):
        if not self.transport.done.wait(1):
            raise TimeoutError()
        return {'complete': True, 'frames': 1}


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.transport = Transport()
        self.director = Director()
        self.binding = LiveAVBinding('fresh', 'a'*64, 1, max_total_frames=248)
        self.ready = dict(status='ready', session='fresh', profile_sha256='a'*64,
                          epoch=1, expires_at=time.time()+60)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def run_controller(self, **kwargs):
        return run_conversation(self.transport, self.binding, Path(self.temp.name),
                                ['今の入力', '次の入力'], self.director,
                                ready=self.ready, max_seconds=2,
                                sink_factory=Sink, pump_factory=Pump, **kwargs)

    def test_two_fresh_typed_turns_and_bound_lifecycle(self):
        result = self.run_controller()
        self.assertTrue(result['complete'])
        self.assertEqual(self.director.inputs, ['今の入力', '次の入力'])
        self.assertEqual([r['kind'] for r in self.transport.rows], ['turn', 'turn', 'stop'])
        self.assertEqual([r['epoch'] for r in self.transport.rows], [1, 2, 2])
        payload = self.transport.rows[0]['payload']
        self.assertEqual(payload['schema'], 'h3-relay-typed-turn-v1')
        self.assertEqual(payload['reply'], rpc_proposal()['reply'])
        self.assertFalse(payload['previous_audio_reused'])
        self.assertEqual(payload['origin'], 'fresh_typed_text')
        self.assertFalse(payload['asr_used'])
        self.assertEqual(payload['director_model'], MODEL)
        self.assertEqual(payload['reasoning_effort'], 'none')
        self.assertNotIn('text', payload)
        self.assertTrue(self.director.closed)
        self.assertTrue(self.transport._closed)
        self.assertTrue(Sink.instances[-1].canceled)
        self.assertFalse(any(t.name == 'h3-mac-receiver' for t in threading.enumerate()))
        self.assertEqual({p.name for p in Path(self.temp.name).iterdir()}, {'stage-timing.json'})
        evidence = json.loads((Path(self.temp.name)/'stage-timing.json').read_text())
        self.assertEqual(evidence['status'], 'completed')
        self.assertEqual(evidence['component'], 'mac_controller')
        self.assertNotIn('今の入力', json.dumps(evidence, ensure_ascii=False))

    def asr_item(self):
        return dict(text='private transcript', asr=dict(
            policy='endpoint-native-v1', endpoint_name='h3-asr-t4-7c926aae18',
            revision='a'*40, endpoint_source='native_completed',
            completion_policy='input-finalized', capture_stopped=True,
            cleanup_complete=True))

    def run_source(self, source, **kwargs):
        return run_conversation(
            self.transport, self.binding, Path(self.temp.name), None,
            self.director, ready=self.ready, max_seconds=2,
            sink_factory=Sink, pump_factory=Pump, input_source=source, **kwargs)

    def test_dynamic_source_bounded_context_and_private_transcript(self):
        calls = []
        def source(*, context, deadline):
            self.assertTrue(any(t.name == 'h3-mac-receiver'
                                for t in threading.enumerate()))
            self.assertGreater(deadline, time.monotonic())
            calls.append(context)
            return self.asr_item()
        self.run_source(source)
        self.assertEqual(calls, [dict(session='fresh', epoch=1),
                                 dict(session='fresh', epoch=2)])
        self.assertEqual(self.director.inputs, ['private transcript']*2)
        payload = self.transport.rows[0]['payload']
        self.assertEqual(payload['asr'], self.asr_item()['asr'])
        self.assertEqual(payload['schema'], 'h3-relay-asr-turn-v1')
        self.assertEqual(payload['origin'], 'comet_process_audio_endpoint_asr')
        self.assertIs(payload['asr_used'], True)
        self.assertNotIn('private transcript', json.dumps(self.transport.rows))

    def test_source_none_finishes_without_director(self):
        self.run_source(lambda **kw: None)
        self.assertEqual(self.director.inputs, [])
        self.assertEqual([r['kind'] for r in self.transport.rows], ['stop'])

    def test_source_invalid_metadata_cancels(self):
        item = self.asr_item()
        item['asr']['cleanup_complete'] = 1
        with self.assertRaises(ValueError):
            self.run_source(lambda **kw: item)
        self.assertEqual(self.director.inputs, [])
        self.assertEqual([r['kind'] for r in self.transport.rows],
                         ['cancel_pending', 'stop'])

    def test_source_cancellation_rejects_even_none(self):
        cancel = threading.Event()
        def source(**kw):
            cancel.set()
            return None
        with self.assertRaises(RuntimeError):
            self.run_source(source, cancel=cancel)
        self.assertEqual(self.director.inputs, [])
        self.assertTrue(self.director.closed)

    def test_source_late_completion_rejected(self):
        def source(**kw):
            self.clock = patch('research.run_h3_relay_conversation.time.monotonic',
                               return_value=kw['deadline']+1)
            self.clock.start()
            self.addCleanup(self.clock.stop)
            return self.asr_item()
        with self.assertRaises(RuntimeError):
            self.run_source(source)
        self.assertEqual(self.director.inputs, [])

    def test_actual_rpc_fixture_matches_offline_js_contract(self):
        script = """
import {validateDraft, MODEL, BASELINE} from './research/director/contract.mjs';
const draft = validateDraft({reply:'新しい返事', expression:'neutral', motion:'still', tone:'calm'});
console.log(JSON.stringify({type:'turn',turn_id:'fixture-turn',epoch:1,
 model:MODEL,baseline:BASELINE,...draft,status:'proposal_only',
 effects_executed:false,renderer_attached:false,audio_attached:false}));
"""
        output = subprocess.run(['node', '--input-type=module', '-e', script],
                                cwd=Path(__file__).resolve().parents[1],
                                check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(json.loads(output.stdout), rpc_proposal())

    def test_rpc_turn_extracts_actual_correlated_proposal(self):
        from io import BytesIO
        from types import SimpleNamespace
        director = PiDirector.__new__(PiDirector)
        director.child = SimpleNamespace(stdin=BytesIO())
        with patch('research.run_h3_relay_conversation.uuid.uuid4',
                   return_value=SimpleNamespace(hex='fixture-turn')):
            director._read = lambda: rpc_proposal()
            self.assertEqual(proposal_reply(director.turn('今の入力')), '新しい返事')
        packet = json.loads(director.child.stdin.getvalue())
        self.assertEqual(packet, dict(op='turn', turn_id='fixture-turn', text='今の入力'))

    def test_reply_character_limit_and_no_flat_fallback(self):
        self.assertEqual(proposal_reply(rpc_proposal('あ'*160)), 'あ'*160)
        for row in (rpc_proposal('あ'*161), rpc_proposal(' '),
                    dict(model=MODEL, reply='invented'),
                    dict(rpc_proposal(), status='output'),
                    dict(rpc_proposal(), effects_executed=True)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                proposal_reply(row)

    def test_invalid_proposal_cancels_before_worker_turn(self):
        self.director.turn = lambda text: rpc_proposal('あ'*161)
        with self.assertRaises(ValueError):
            self.run_controller()
        self.assertEqual([r['kind'] for r in self.transport.rows], ['cancel_pending', 'stop'])

    def test_failure_cancels_without_fabricated_eos(self):
        self.director.fail = True
        with self.assertRaises(RuntimeError):
            self.run_controller()
        self.assertEqual([r['kind'] for r in self.transport.rows], ['cancel_pending', 'stop'])
        self.assertTrue(self.director.closed)
        self.assertTrue(self.transport._closed)

    def test_receiver_failure_has_only_structural_diagnostic(self):
        class FailedPump(Pump):
            def run(self):
                self.transport.done.wait(1)
                payload = 'SECRET-PAYLOAD'
                raise ValueError('provider message ' + payload)

        self.transport.last_failure = dict(phase='response-body', http_status=200,
            error_type='IncompleteRead', message='SECRET', url='SECRET')
        with patch('research.run_h3_relay_conversation.ReceiverPump', FailedPump):
            with self.assertRaises(RuntimeError) as caught:
                run_conversation(self.transport, self.binding, Path(self.temp.name),
                                 ['fresh'], self.director, ready=self.ready,
                                 max_seconds=2, sink_factory=Sink, pump_factory=FailedPump)
        message = str(caught.exception)
        self.assertIn('receiver failed; no replay: ValueError', message)
        self.assertRegex(message, r'test_h3_relay_conversation.py:run:\d+')
        self.assertNotIn('SECRET', message)
        self.assertNotIn('provider message', message)
        self.assertNotIn(str(Path(__file__).parent), message)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(self.transport._closed)
        raw=(Path(self.temp.name)/'receiver-failure.json').read_text()
        evidence=json.loads(raw)
        self.assertEqual(evidence['transport_failure'], dict(phase='response-body',
            http_status=200,error_type='IncompleteRead'))
        self.assertNotIn('SECRET', raw)
        self.assertNotIn('provider message', raw)

    def test_receiver_diagnostic_is_bounded_and_sanitized(self):
        from research.run_h3_relay_conversation import _receiver_diagnostic

        class PrivateError(Exception):
            def __str__(self):
                raise AssertionError('must not format provider exception')

        PrivateError.__name__ = 'Bad\\n' + 'x' * 200

        def recurse(depth):
            if depth:
                return recurse(depth - 1)
            raise PrivateError('SECRET')

        try:
            recurse(30)
        except PrivateError as exc:
            diagnostic = _receiver_diagnostic(exc)
        self.assertTrue(diagnostic.startswith('Bad_'))
        self.assertNotIn('\n', diagnostic)
        self.assertNotIn('SECRET', diagnostic)
        self.assertEqual(diagnostic.count(':recurse:'), 8)
        self.assertLess(len(diagnostic), 1500)

    def test_obs_requires_explicit_gate(self):
        with self.assertRaises(PermissionError):
            self.run_controller(obs_port=12345)
        self.assertEqual(self.transport.rows, [])

    def test_explicit_obs_gate_passes_only_output_parameters(self):
        self.run_controller(allow_obs=True, obs_port=12345)
        self.assertTrue(Sink.instances[-1].kwargs['playback_approved'])
        self.assertEqual(Sink.instances[-1].kwargs['port'], 12345)

    def test_stale_ready_and_unopened_rejected(self):
        self.ready['expires_at'] = time.time()-1
        with self.assertRaises(ValueError):
            self.run_controller()
        self.ready['expires_at'] = time.time()+60
        self.transport._opened = False
        with self.assertRaises(ValueError):
            self.run_controller()

    def test_cancel_before_pi(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(RuntimeError):
            self.run_controller(cancel=cancel)
        self.assertEqual(self.director.inputs, [])
        self.assertTrue(self.director.closed)


if __name__ == '__main__':
    unittest.main()
