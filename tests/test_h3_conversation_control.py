"""Offline host-RPC tests with real owner/WindowInput and no native model."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from research.h3_personal import conversation_control as control
from research.h3_personal.fast_continuation import WindowInput
import test_h3_conversation_continuation as fixtures


class ControlTests(unittest.TestCase):
    def command(self, cid='conversation'):
        return {'conversation_control': dict(schema=control.SCHEMA, action='abandon',
                    conversation_id=cid, reason='local-sink-failure')}

    def test_pinned_native_empty_frame_format(self):
        import ast
        from pathlib import Path
        root = Path('/tmp/h3-control-audit/FastVideo-ba371e33ed080e51f1884ae166baf7c4447a66fe/fastvideo')
        if not root.exists():
            self.skipTest('pinned native archive audit fixture unavailable')
        source = (root / 'entrypoints/video_generator.py').read_text()
        tree = ast.parse(source)
        expressions = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'frames' for t in n.targets)
                       and isinstance(n.value, ast.ListComp)]
        self.assertEqual(len(expressions), 1)
        # Execute the actual pinned expression: no grid/media call can occur.
        class FakeNativeEngine:
            def format(self):
                return eval(compile(ast.Expression(expressions[0]), '<native>', 'eval'),
                            {'vid_u8': [], 'torchvision': None})
        result_tree = ast.parse((root / 'api/results.py').read_text())
        cls = next(n for n in result_tree.body if isinstance(n, ast.ClassDef) and n.name == 'GenerationResult')
        from dataclasses import dataclass, field
        namespace = dict(dataclass=dataclass, field=field, Any=object, ContinuationState=object)
        module = ast.Module(body=[cls], type_ignores=[])
        exec(compile(module, '<native-result>', 'exec'), namespace)
        result = namespace['GenerationResult'](frames=FakeNativeEngine().format())
        self.assertEqual(result.frames, [])
        self.assertIsNone(result.video_path)

    def test_off_does_not_parse_or_vote(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(control, 'read_manifest', side_effect=AssertionError('parse')), \
             patch('research.h3_personal.conversation_stages.common', side_effect=AssertionError('vote')):
            self.assertFalse(control.is_control())
            self.assertFalse(control.boundary(NS(), NS(), 'cpu'))

    def test_independent_startup_no_source_digest_or_vote(self):
        with patch.dict(os.environ, H3_REF2VA_CONVERSATION_ROLES_FILE='configured',
                        H3_SMD_CONTINUATION_WINDOW_FILE='unused'), \
             patch.object(control, 'read_manifest', return_value={'mode': 'independent'}), \
             patch('research.h3_personal.fast_continuation_stages._manifest', return_value={'mode': 'independent'}) as validate, \
             patch('research.h3_personal.fast_continuation_stages._source_hash', side_effect=AssertionError('hash')), \
             patch('research.h3_personal.conversation_stages.common', side_effect=AssertionError('vote')):
            pipeline = NS(_stage_name_mapping={'latent_preparation_stage': NS(),
                          'denoising_stage': NS(), 'video_decoding_stage': NS()})
            self.assertFalse(control.boundary(pipeline, NS(), 'cpu'))
            validate.assert_called_once()

    def test_native_inflight_cleared_on_unknown_failure_preserves_faults(self):
        from research.h3_personal import fast_ref2va as fast
        pipeline = NS(_smd_continuation_warm_faulted=True)
        def fail(*args):
            self.assertTrue(pipeline._h3_conversation_forward_inflight)
            raise RuntimeError('unknown')
        with patch.dict(os.environ, H3_REF2VA_CONVERSATION_ROLES_FILE='configured'), \
             patch.object(fast, '_warm_forward', side_effect=fail):
            with self.assertRaises(RuntimeError): fast._guarded_forward(pipeline, NS(), None, None)
        self.assertFalse(pipeline._h3_conversation_forward_inflight)
        self.assertTrue(pipeline._smd_continuation_warm_faulted)

    def test_rpc_owns_boundary_flag_without_parent_setup(self):
        from research.h3_personal import fast_ref2va as fast
        import sys
        pipeline, batch = NS(), NS()
        def boundary(p, b, device):
            self.assertTrue(p._h3_conversation_control_boundary)
            self.assertFalse(getattr(p, '_h3_conversation_forward_inflight', False))
            return True
        module = NS(get_local_torch_device=lambda: 'cpu')
        with patch.dict(os.environ, H3_REF2VA_CONVERSATION_ROLES_FILE='configured'), \
             patch.dict(sys.modules, {'fastvideo.pipelines.composed_pipeline_base': module}), \
             patch.object(control, 'boundary', side_effect=boundary):
            self.assertIs(fast.warm_forward(pipeline, batch, None, None), batch)
        self.assertFalse(pipeline._h3_conversation_control_boundary)

    def test_exact_manifest(self):
        self.assertEqual(control.validate_control(self.command())['action'], 'abandon')
        for m in ({'mode': 'latent-tail-v1', **self.command()},
                  {'conversation_control': {}}, self.command(123)):
            with self.assertRaises(ValueError): control.validate_control(m)

    def test_real_owner_abandon_and_late_command(self):
        fixture = fixtures.ConversationTests()
        owner = fixture.owner()
        fixture.source(owner, 'a')
        token = owner.begin_window()
        self.assertIsInstance(token.local, WindowInput)
        owner._active = None  # Simulate the completed serialized native call.
        stage = NS(_h3_conversation_model_healthy=True,
                   _h3_conversation_audio_history=object(),
                   _h3_conversation_state={'owner': owner})
        decoder = NS()
        pipeline = NS(_h3_conversation_control_boundary=True, _stage_name_mapping={'latent_preparation_stage': stage, 'denoising_stage': NS(),
                                         'video_decoding_stage': decoder})
        batch = NS()
        empty = NS(shape=(0, 3, 0, 0, 0), device='cpu')
        torch = NS(empty=lambda *a, **k: empty, float32='float32')
        with tempfile.TemporaryDirectory() as d:
            path = d + '/command.json'
            with open(path, 'w') as f: json.dump(self.command(), f)
            with patch.dict(os.environ, H3_REF2VA_CONVERSATION_ROLES_FILE='configured', H3_SMD_CONTINUATION_WINDOW_FILE=path), \
                 patch('research.h3_personal.conversation_stages.common', side_effect=lambda d, op: op()), \
                 patch.dict('sys.modules', torch=torch):
                self.assertTrue(control.boundary(pipeline, batch, 'cpu'))
                self.assertIs(batch.output, empty)
                self.assertIsNone(stage._h3_conversation_state)
                self.assertFalse(hasattr(stage, '_h3_conversation_audio_history'))
                with self.assertRaises(ValueError): control.boundary(pipeline, batch, 'cpu')
                new = fixture.owner()
                fixture.source(new, 'new-source')
                self.assertIsNotNone(new.begin_window())

    def test_retirement_requires_final_decoder_ack(self):
        for final in (False,True):
            fixture=fixtures.ConversationTests();owner=fixture.owner();fixture.source(owner,'a')
            fixture.commit(owner,owner.begin_window(),final=final)
            owner.ack_decoded_count(124 if final else 119,final_flush=final)
            stage=NS(_h3_conversation_model_healthy=True,_h3_conversation_state={'owner':owner},
                     _h3_conversation_audio_history=object())
            decoder=NS()
            pipeline=NS(_h3_conversation_control_boundary=True,_stage_name_mapping={
                'latent_preparation_stage':stage,'denoising_stage':NS(),'video_decoding_stage':decoder})
            command=self.command();command['conversation_control'].update(action='retire',reason='completed')
            torch=NS(empty=lambda *a,**k:NS(),float32='float32')
            with patch.dict(os.environ,H3_REF2VA_CONVERSATION_ROLES_FILE='configured',H3_SMD_CONTINUATION_WINDOW_FILE='unused'), \
                 patch.object(control,'read_manifest',return_value=command), \
                 patch('research.h3_personal.conversation_stages.common',side_effect=lambda d,op:op()), \
                 patch.dict('sys.modules',torch=torch):
                if not final:
                    with self.assertRaises(ValueError):control.boundary(pipeline,NS(),'cpu')
                    self.assertIs(stage._h3_conversation_state['owner'],owner)
                else:
                    self.assertTrue(control.boundary(pipeline,NS(),'cpu'))
                    self.assertIsNone(stage._h3_conversation_state)
                    self.assertFalse(hasattr(stage,'_h3_conversation_audio_history'))

    def test_rejected_common_cleanup_preserves_slots(self):
        fixture = fixtures.ConversationTests()
        owner = fixture.owner()
        stage = NS(_h3_conversation_model_healthy=True,
                   _h3_conversation_state={'owner': owner})
        decoder = NS(_h3_conversation_decoder='retained')
        pipeline = NS(_h3_conversation_control_boundary=True,
                      _stage_name_mapping={'latent_preparation_stage': stage, 'denoising_stage': NS(), 'video_decoding_stage': decoder})
        calls = []
        def common(device, op):
            calls.append(op)
            if len(calls) == 2:
                raise RuntimeError('peer cleanup rejected')
            return op()
        with patch.dict(os.environ, H3_REF2VA_CONVERSATION_ROLES_FILE='configured', H3_SMD_CONTINUATION_WINDOW_FILE='unused'), \
             patch.object(control, 'read_manifest', return_value=self.command()), \
             patch('research.h3_personal.conversation_stages.common', side_effect=common):
            with self.assertRaises(RuntimeError): control.boundary(pipeline, NS(), 'cpu')
        self.assertIs(stage._h3_conversation_state['owner'], owner)
        self.assertEqual(decoder._h3_conversation_decoder, 'retained')
        self.assertFalse(hasattr(stage, '_h3_retired_conversations'))

    def test_missing_or_false_health_never_promoted_by_control(self):
        for stage in (NS(), NS(_h3_conversation_model_healthy=False)):
            with self.assertRaises(RuntimeError): control.healthy(NS(), stage)

    def test_parse_failure_stays_continuation(self):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write('{'); f.flush()
            with patch.dict(os.environ, H3_REF2VA_CONVERSATION_ROLES_FILE='configured', H3_SMD_CONTINUATION_WINDOW_FILE=f.name):
                self.assertTrue(control.is_control())
                with self.assertRaises(ValueError): control.read_manifest()


if __name__ == '__main__': unittest.main()
