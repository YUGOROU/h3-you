import unittest
from types import SimpleNamespace
from unittest.mock import patch
from test_h3_conversation_continuation import ConversationTests
from research.h3_personal.conversation_stages import (
    validate_manifest, SourceFacade, common, source_identity)
from research.h3_personal.fast_continuation import WindowInput, WindowCancelled


def manifest(epoch=0, index=0):
    final = index == (2 if epoch == 1 else 0)
    return dict(mode='latent-tail-v1', session_id='source', index=index,
                source_index=index, total_frames=328 if epoch == 1 else 124,
                final=final, candidate={'audio_context': 'window-context-canonical-v1'},
                conversation=dict(schema='h3-model-conversation-v1', id='c',
                    profile_sha256='a'*64, source_epoch=epoch, source_id='source',
                    role='speech' if epoch == 1 else 'listening',
                    source_onset_frame=(0, 124, 430)[epoch],
                    conversation_final=epoch == 2 and final, total_frames=532))


class StageTests(unittest.TestCase):
    def test_contract(self):
        for epoch in range(3):
            validate_manifest(manifest(epoch))
        for key, value in [('source_epoch', True), ('total_frames', 328),
                           ('profile_sha256', 'X'*64), ('extra', 1),
                           ('conversation_final', True)]:
            m = manifest()
            m['conversation'][key] = value
            with self.assertRaises(ValueError): validate_manifest(m)

    def test_source_identity_ignores_only_window_fields(self):
        a, b = manifest(1, 0), manifest(1, 2)
        self.assertEqual(source_identity(a), source_identity(b))
        b['conversation']['profile_sha256'] = 'b'*64
        self.assertNotEqual(source_identity(a), source_identity(b))

    def test_actual_model_token_and_new_source_guide(self):
        fixture = ConversationTests()
        o = fixture.owner()
        fixture.source(o, 'a')
        first = o.begin_window()
        fixture.commit(o, first)
        o.ack_decoded_count(119)
        fixture.source(o, 'b', value=7)
        token = o.begin_window()
        facade = SourceFacade(o, token)
        self.assertIsInstance(token.model, WindowInput)
        self.assertIs(facade.pending_window, token.model)
        self.assertEqual(token.model.window.start, 102)
        self.assertEqual(token.model.window.guide_video_t, 7)
        self.assertEqual(token.model.guide_audio.values[0], 7)
        facade._source_ok()
        facade.fault_window(token.model)
        self.assertTrue(facade.faulted)

    def test_common_peer_and_local_failure(self):
        class Status:
            def __init__(self, value): self.value = value
            def item(self): return self.value
        dist = SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 2,
                               ReduceOp=SimpleNamespace(MIN='min'))
        fake = SimpleNamespace(tensor=lambda v, **kw: Status(v), int32='int', distributed=dist)
        with patch.dict('sys.modules', torch=fake):
            dist.all_reduce = lambda s, **kw: None
            self.assertEqual(common('cpu', lambda: 4), 4)
            with self.assertRaises(WindowCancelled):
                common('cpu', lambda: 1/0)
            dist.all_reduce = lambda s, **kw: setattr(s, 'value', 0)
            with self.assertRaises(WindowCancelled): common('cpu', lambda: 4)
