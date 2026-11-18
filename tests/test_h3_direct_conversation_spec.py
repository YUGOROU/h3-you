"""Offline direct entry contract tests; no GPU, downloads or network."""
import ast
import inspect
from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch

from research.hf_h3_direct_conversation import validate_spec, run_session
from research.h3_personal.fast_ref2va import config_for_layout
from research.launch_h3_fast_ref2va import IDENTITY_PREFIX, REFERENCE_SHA


def spec():
    session = 'a'*32
    return dict(live_direct_session=dict(schema='h3-live-direct-session-v1', session=session,
        port=7860, max_windows=16, max_frames=1654, lease_seconds=1200),
        live_voice_session=dict(id=session, max_turns=2, fresh_input_required=True),
        fast_config=asdict(config_for_layout('portrait')), layout='portrait',
        attention_mode='fa4-packed', runtime_policy='resident-visual-cached',
        identity_prefix=IDENTITY_PREFIX, reference_sha256=REFERENCE_SHA)


def asr_binding():
    return dict(policy='endpoint-native-v1', endpoint_name='h3-asr-t4-7c926aae18',
        revision='b'*40, endpoint_source='native_completed',
        completion_policy='input-finalized', capture_stopped=True, cleanup_complete=True)


class DirectSpecTests(unittest.TestCase):
    def test_optional_asr_binding_is_pure_and_matches_turn_contract(self):
        from research.h3_personal.live_relay_worker import validate_turn, PI_MODEL
        item = spec(); item['live_direct_session']['expected_asr'] = asr_binding()
        before = deepcopy(item)
        self.assertEqual(validate_spec(item), item['live_direct_session'])
        self.assertEqual(item, before)
        payload = dict(schema='h3-relay-asr-turn-v1', origin='comet_process_audio_endpoint_asr',
            asr_used=True, director_model=PI_MODEL, reasoning_effort='none',
            reply='hello', previous_audio_reused=False, asr=asr_binding())
        self.assertEqual(validate_turn({'payload': payload},
            item['live_direct_session']['expected_asr']), 'hello')

    def test_asr_binding_exact_keys_and_values(self):
        invalid = [None, [], 'asr', {}]
        for key in asr_binding():
            missing = asr_binding(); del missing[key]; invalid.append(missing)
            wrong = asr_binding(); wrong[key] = None; invalid.append(wrong)
        for key, values in dict(policy=['other'], endpoint_name=['other'],
                revision=['B'*40, 'b'*39, 'b'*41, 'g'*40, 1, True],
                endpoint_source=['other'], completion_policy=['other'],
                capture_stopped=[False, 1, 'true'], cleanup_complete=[False, 1, 'true'],
                extra=[True]).items():
            for value in values:
                wrong = asr_binding(); wrong[key] = value; invalid.append(wrong)
        for binding in invalid:
            with self.subTest(binding=binding):
                item = spec(); item['live_direct_session']['expected_asr'] = binding
                with self.assertRaises(ValueError): validate_spec(item)
        item = spec(); item['live_direct_session'].update(expected_asr=asr_binding(), extra=True)
        with self.assertRaises(ValueError): validate_spec(item)

    def test_worker_asr_keyword_only_when_present(self):
        # Execute the real wiring statements in isolation from GPU/device startup.
        from pathlib import Path
        from unittest.mock import Mock
        tree = ast.parse(inspect.getsource(run_session))
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'worker_options' for t in node.targets))
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'run_worker')
        code = compile(ast.fix_missing_locations(ast.Module(
            body=[assignment, ast.Expr(value=call)], type_ignores=[])), '<worker wiring>', 'exec')
        for present in (False, True):
            row = spec()['live_direct_session']
            if present:
                row['expected_asr'] = asr_binding()
            worker = Mock()
            namespace = dict(row=row, run_worker=worker, runtime=object(), tts=object(),
                server=Mock(), binding=object(), context=object(), output=Path('/unused'),
                deadline=1200, on_ready=lambda: None)
            exec(code, namespace)
            self.assertEqual(worker.call_count, 1)
            kwargs = worker.call_args.kwargs
            self.assertEqual('expected_asr' in kwargs, present)
            if present:
                self.assertIs(kwargs['expected_asr'], row['expected_asr'])

    def test_direct_ready_print_is_deferred_to_worker_callback(self):
        # Guard the entrypoint wiring without importing/starting GPU dependencies.
        tree=ast.parse(inspect.getsource(run_session))
        callbacks=[node for node in ast.walk(tree)
                   if isinstance(node,ast.FunctionDef) and node.name=='on_ready']
        self.assertEqual(len(callbacks),1)
        callback=callbacks[0]
        prints=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Name) and node.func.id=='print'
                and any(isinstance(child,ast.Constant) and child.value=='h3_direct_ready'
                        for child in ast.walk(node))]
        self.assertEqual(len(prints),1)
        self.assertIn(prints[0],list(ast.walk(callback)))
        self.assertIn('h3_direct_ready',ast.unparse(callback))
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Name) and node.func.id=='run_worker']
        self.assertEqual(len(calls),1)
        ready=[kw.value for kw in calls[0].keywords if kw.arg=='on_ready']
        self.assertEqual(len(ready),1)
        self.assertIsInstance(ready[0],ast.Name)
        self.assertEqual(ready[0].id,callback.name)

    def test_valid_and_pure_without_wall_clock(self):
        value = spec(); before = deepcopy(value)
        with patch('time.time', side_effect=AssertionError('no submission expiry')):
            self.assertEqual(validate_spec(value), value['live_direct_session'])
        self.assertEqual(value, before)

    def test_exact_session_schema(self):
        for field, value in [('not_after', 9999999999), ('producerKey', 'secret'),
                             ('space_id', 'not-used'), ('expires_at', 1)]:
            with self.subTest(field=field):
                item = spec(); item['live_direct_session'][field] = value
                with self.assertRaises(ValueError): validate_spec(item)
        for field in spec()['live_direct_session']:
            item = spec(); del item['live_direct_session'][field]
            with self.assertRaises(ValueError): validate_spec(item)

    def test_session_limits(self):
        for field, values in dict(port=[7861, 7860.0, True], max_windows=[15, 17, 16.0],
                max_frames=[1653, 1655, 1654.0], session=[None, 'A'*32, 'a'*31],
                lease_seconds=[0, -1, 1201, float('inf'), float('nan'), True, '1200']).items():
            for value in values:
                with self.subTest(field=field, value=value):
                    item = spec(); item['live_direct_session'][field] = value
                    with self.assertRaises(ValueError): validate_spec(item)

    def test_every_quality_constant_is_locked(self):
        for key, value in spec()['fast_config'].items():
            item = spec()
            item['fast_config'][key] = None
            with self.subTest(key=key), self.assertRaises(ValueError): validate_spec(item)
            if type(value) is int:
                item['fast_config'][key] = float(value)
                with self.assertRaises(ValueError): validate_spec(item)
        item = spec(); item['fast_config']['unsupported'] = True
        with self.assertRaises(ValueError): validate_spec(item)

    def test_runtime_flags_and_selectors(self):
        for key, value in dict(layout='landscape', attention_mode='fa2', runtime_policy='baseline',
                identity_prefix='other', reference_sha256='0'*64, regional_compile=True,
                warm_stage_diagnostic=1, incremental_transfer=True, face_crop_box=[0, 0, 1, 1],
                locked_speech={}, streaming_trial={}, conversation_scenario={}, live_relay_session={}).items():
            with self.subTest(key=key):
                item = spec(); item[key] = value
                with self.assertRaises(ValueError): validate_spec(item)

    def test_fresh_input_binding(self):
        for key, value in [('id', 'b'*32), ('max_turns', True), ('max_turns', 16),
                           ('fresh_input_required', False), ('extra', True)]:
            item = spec(); item['live_voice_session'][key] = value
            with self.assertRaises(ValueError): validate_spec(item)


if __name__ == '__main__':
    unittest.main()
