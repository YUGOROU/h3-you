"""Real owner/window CPU receipts; no GPU, network or playback acceptance."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research.h3_personal import conditioning_experiment as e
from research.h3_personal import conversation_conditioning as roles
from research.h3_personal.accepted_profiles import choose_profile
from research.h3_personal.conversation_continuation import ConversationOwner
from research.h3_personal.conversation_audio import audio_origin
from research.h3_personal.conversation_stages import SourceFacade, SCHEMA
from research.h3_personal.continuation_plan import audio_boundary, video_t
from research.h3_personal.fast_continuation import SamplerContract, WindowInput
from research.h3_personal.fast_ref2va import condition_key, store_condition, get_condition_cache
from test_h3_fast_continuation import Tensor, Ops


class Tags:
    def __init__(self, n): self.shape = (n,)
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return Tags(self.shape[0])


class FakeNativeEncoder:
    def __init__(self, processor):
        self.processor, self.calls = processor, []
    def forward(self, batch, args):
        edge, role = e.reference_short_edge(), roles.current_role()
        self.calls.append((edge, role, hashlib.sha256(batch.prompt.encode()).hexdigest()))
        batch.prompt_embeds = [Tags(3)]
        # Listening is intentionally not the historical speech T.
        batch.extra['minimax_h3_text_token_tags'] = Tags(
            e.PORTRAIT_EXPECTED_RT[edge]['R'] + (43 if role == 'speech' else 31))
        store_condition(self, batch, args)
        return batch


class ConversationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        env = patch.dict(os.environ, {}, clear=True); env.start(); self.addCleanup(env.stop)
        self.source = self.root/'image'; self.source.write_bytes(b'original')
        self.processor = NS(image_processor=NS(to_dict=lambda: {'native': 'exact'}))
        processor_pin = patch('research.h3_personal.accepted_profiles.PROCESSOR_SHA256',
                              e.processor_digest(NS(processor=self.processor)))
        processor_pin.start(); self.addCleanup(processor_pin.stop)
        self.profile = choose_profile(ordered_source_sha256=(hashlib.sha256(b'original').hexdigest(),),
            prompt_sha256=roles.prompt_sha('listening'),
            processor_sha256=e.processor_digest(NS(processor=self.processor)))
        self.files = {}
        for name, value, envkey in (
            ('policy', self.profile.legacy_conditioning_manifest(), e.EXPERIMENT_ENV),
            ('accepted', self.profile.wire_record(), e.ACCEPTED_PROFILE_ENV),
            ('roles', roles.roles_manifest(), roles.ROLES_ENV),
            ('role', roles.role_record('listening'), roles.CURRENT_ROLE_ENV)):
            p = self.root/name; p.write_text(json.dumps(value)); os.environ[envkey] = str(p)
            self.files[name] = p
        os.environ['H3_REF2VA_COVERAGE_DIR'] = str(self.root/'receipts')
        self.window_file = self.root/'window'
        os.environ['H3_SMD_CONTINUATION_WINDOW_FILE'] = str(self.window_file)
        torch = patch.dict(sys.modules, {'torch': NS(distributed=NS(get_rank=lambda: 0))})
        torch.start(); self.addCleanup(torch.stop)
        self.stage = FakeNativeEncoder(self.processor)
        self.batch = self.fresh_batch()
        e.snapshot_sources(self.batch)
        with patch.object(e, 'prepare_reference', self.prepare):
            store_condition(self.stage, self.batch)
        self.activate('listening')
        self.owner = ConversationOwner('CONVERSATION', profile=self.profile.digest,
            height=2, width=2, enabled=True, model_healthy=True, ops=Ops())
        self.start_source('SOURCE0', 124)
        self.attach(0)
        parameter = object()
        self.pipeline = NS(_stage_name_mapping={'conditioning_stage': self.stage},
            _denoise_stages_ready=True, get_module=lambda name: NS(parameters=lambda: iter([parameter])))

    def fresh_batch(self):
        return NS(prompt=roles.LISTENING_PROMPT, width=256, height=448, num_frames=124, fps=24,
            references=[NS(source=self.source, media_type='image')], extra={}, prompt_embeds=[])

    def prepare(self, ref, frames, rate):
        edge = e.reference_short_edge()
        return NS(media_type='image', image=NS(size=(edge, {1024:1824, 512:896}[edge])))

    def activate(self, role):
        self.files['role'].write_text(json.dumps(roles.role_record(role)))
        p = dataclasses.replace(self.profile, prompt_sha256=roles.prompt_sha(role))
        self.files['accepted'].write_text(json.dumps(p.wire_record()))
        self.batch.prompt = roles.role_prompt(role)
        self.batch.references = [self.prepare(None, 124, 32000)]
        e.capture_prepared_geometry(self.batch)
        self.batch.references = None
        key = condition_key(self.batch)
        entry = get_condition_cache(self.stage, key)
        self.batch.extra['minimax_h3_text_token_tags'] = entry[2]
        self.batch.extra['_h3_conditioning_cache_hit'] = True

    def start_source(self, sid, total):
        self.owner.start_source(sid, canonical_audio=Tensor((2, audio_boundary(total), 32)),
            total_frames=total, profile=self.profile.digest, height=2, width=2,
            audio_origin=audio_origin(self.owner.committedF, total))

    def attach(self, epoch):
        token = self.owner.begin_window()
        self.assertIsInstance(token.model, WindowInput)
        m = dict(mode='latent-tail-v1', session_id=token.source_id, index=token.local.index,
            source_index=token.local.index, final=token.local.window.end == (328 if epoch == 1 else 124),
            total_frames=328 if epoch == 1 else 124,
            candidate={'audio_context':'window-context-canonical-v1'},
            conversation=dict(schema=SCHEMA, id='CONVERSATION', profile_sha256=self.profile.digest,
                source_epoch=epoch, source_id=token.source_id, role='speech' if epoch == 1 else 'listening',
                source_onset_frame=(0,124,430)[epoch], conversation_final=False, total_frames=532))
        self.manifest = m
        self.window_file.write_text(json.dumps(m))
        self.batch.extra['_h3_continuation'] = dict(manifest=m, conversation_owner=self.owner,
            global_token=token, token=token.model, owner=SourceFacade(self.owner, token))
        n = self.batch.extra['minimax_h3_text_token_tags'].shape[0]
        self.batch.extra['minimax_h3_layout'] = NS(
            num_condition_video_rows=1824 + token.window.guide_video_t*112,
            text_indices=NS(numel=lambda: n))
        return token

    def test_four_native_keys_and_actual_listening_receipt(self):
        self.assertEqual(len(self.stage._h3_conditioning_cache), 4)
        self.assertEqual({(edge, role) for edge, role, sha in self.stage.calls},
                         {(edge, role) for edge in (1024,512) for role in ('speech','listening')})
        for edge, role, sha in self.stage.calls:
            self.assertEqual(sha, roles.prompt_sha(role))
        r = e.write_evidence(self.pipeline, self.batch)
        self.assertEqual((r['R'], r['T']), (1824,1855))
        self.assertEqual((r['session_id'],r['conversation_id'],r['source_epoch']), ('SOURCE0','CONVERSATION',0))
        self.assertEqual(r['fixed_base_profile_digest'], self.profile.digest)

    def test_streaming_window_uses_global_context_not_legacy_session_id(self):
        from research.h3_personal.streaming_native_stages import SCHEMA as STREAM
        m=dict(self.manifest)
        m.pop('conversation')
        m.update(expected_windows=1,source_samples=48000,source_sample_rate=48000,
            streaming_audio=dict(schema=STREAM,id='CONVERSATION',profile_sha256=self.profile.digest,
                sequence=0,logical_source_id='idle',logical_index=0,role='listening',source_final=False,
                conversation_final=False,committed_before=0,max_windows=8,audio_offer=None))
        self.batch.extra['_h3_continuation']['manifest']=m
        self.window_file.write_text(json.dumps(m))
        r=e.write_evidence(self.pipeline,self.batch)
        self.assertEqual(r['streaming_sequence'],0)
        self.assertEqual(r['logical_source_id'],'idle')
        self.assertEqual(r['current_role'],'listening')

    def test_guided_actual_window_subtracts_seven(self):
        t = self.batch.extra['_h3_continuation']['global_token']
        self.owner.commit_window(t, clean_video=Tensor((video_t(124),96)),
            clean_audio=t.model.locked_audio, observed_sampler=SamplerContract())
        self.owner.ack_decoded_count(119)
        self.activate('speech')
        self.start_source('SOURCE1',328)
        t = self.attach(1)
        self.assertEqual(t.window.guide_video_t,7)
        r = e.write_evidence(self.pipeline,self.batch)
        self.assertEqual((r['R'],r['T'],r['current_role']), (1824,1867,'speech'))
        self.assertEqual(r['source_epoch'],1)

    def test_wrong_native_t_role_prompt_profile_processor_rejected(self):
        key = condition_key(self.batch)
        proof = self.stage._h3_role_condition_proofs[key]
        for field, bad in [('T',1856), ('role','speech'), ('prompt_sha256',roles.prompt_sha('speech')),
                           ('profile_digest','0'*64)]:
            with self.subTest(field=field):
                old = proof[field]; proof[field] = bad
                with self.assertRaises(RuntimeError): e.write_evidence(self.pipeline,self.batch)
                proof[field] = old
        entry = self.stage._h3_conditioning_cache[key]
        self.stage._h3_conditioning_cache[key] = (key,entry[1],Tags(1856))
        with self.assertRaises(RuntimeError): e.write_evidence(self.pipeline,self.batch)
        self.stage._h3_conditioning_cache[key] = entry
        self.stage.processor = NS(image_processor=NS(to_dict=lambda:{'wrong':True}))
        with self.assertRaises(ValueError): e.write_evidence(self.pipeline,self.batch)

    def test_foreign_token_epoch_and_owner_rejected(self):
        ctx = self.batch.extra['_h3_continuation']
        for key, bad in [('token',dataclasses.replace(ctx['token'])),
                         ('global_token',dataclasses.replace(ctx['global_token']))]:
            old = ctx[key]; ctx[key] = bad
            with self.assertRaises(RuntimeError): e.write_evidence(self.pipeline,self.batch)
            ctx[key] = old
        self.owner._epoch += 1
        with self.assertRaises(RuntimeError): e.write_evidence(self.pipeline,self.batch)

    def test_warm_missing_key_refuses_and_off_defaults(self):
        self.stage._h3_conditioning_cache.pop(condition_key(self.batch))
        with self.assertRaises(RuntimeError): e.write_evidence(self.pipeline,self.batch)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(e.read_policy())
            self.assertEqual(e.reference_short_edge(),2048)
            self.assertIsNone(e.write_evidence(self.pipeline,self.batch))
