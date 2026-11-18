"""Existing production boundaries with explicit fixtures, never real Pi/H3 evidence."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock

from research.h3_personal import conversation_conditioning as cc, conditioning_experiment as ce
from research.h3_personal.fast_ref2va import FastRef2VAConfig
from research.h3_personal.live_native_backend import LiveNativeBackend
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
from research.h3_personal.live_relay_commands import Commands
from research.h3_personal.live_relay_worker import validate_turn, PI_MODEL
from research.run_h3_relay_conversation import proposal_reply
from test_h3_live_native_backend import FakeRuntime, driver, export_stub
from test_h3_live_worker_interrupt import MemoryInput


def intent(text):return dict(schema=cc.VISUAL_INTENT_SCHEMA,instruction=text)


def proposal(text):
    return dict(type='turn', model=PI_MODEL, status='proposal_only', effects_executed=False,
        renderer_attached=False, audio_attached=False, turn_id='fixture', epoch=1,
        baseline=dict(visualPrompt='h3-visual-intent-v1-unverified'),
        schema='h3-director-visual-turn-v1', reply='少し周りを見せるね。', visual_intent=intent(text))


def payload(text):
    p=proposal(text)
    return dict(schema='h3-relay-typed-turn-v2', origin='fresh_typed_text', asr_used=False,
        director_model=PI_MODEL, reasoning_effort='none', reply=proposal_reply(p,visual_intents=True),
        previous_audio_reused=False, visual_intent=p['visual_intent'])


class VisualRuntime(FakeRuntime):
    """Mock native math; real role transaction and real both-rank proof reader."""
    _switch_role=LiveVoiceRuntime._switch_role
    def __init__(self, root):
        super().__init__(root)
        self.visual_intents=True; self.config=FastRef2VAConfig(height=448,width=256)
        self.roles_file=self.coverage/'roles.json'; self.current_role_file=self.coverage/'role.json'
        self.quality_profile_file=self.coverage/'profile.json'
        self.roles_file.write_text(json.dumps(cc.roles_manifest(visual_intents=True)))
        self.current_role_file.write_text(json.dumps(cc.role_record('speech',visual_intents=True)))
        self.quality_profile_file.write_text(json.dumps(self.quality_profile.wire_record()))
    def _validate_quality_binding(self, profile):
        self.config.validate()
        profile.validate_binding(ordered_source_sha256=self.quality_profile.ordered_source_sha256,
            prompt_sha256=hashlib.sha256(self.config.prompt.encode()).hexdigest(),
            processor_sha256=profile.processor_sha256, geometry=(self.config.width,self.config.height))
    def _generate(self, **kwargs):
        # Reuse existing fixture generation without asserting its legacy prompt.
        with patch('test_h3_live_native_backend.role_prompt',cc.conditioning_prompt):
            result=super()._generate(**kwargs)
        m=json.loads(self.continuation_file.read_text())
        if 'streaming_audio' in m:
            c=m['streaming_audio']; visual=c['visual_conditioning']
            for rank in (0,1):
                p=self.coverage/f'conditioning-experiment-{m["session_id"]}-window-0-rank-{rank}.json'
                row=json.loads(p.read_text())
                row.update(role_proof=dict(prompt_sha256=visual['prompt_sha256']),
                    visual_condition_refresh=dict(encoded=False,duration_seconds=0.0),visual_encoder_retained=True,
                    visual_conditioning={k:visual[k] for k in ('target_epoch','intent_epoch','prompt_sha256')})
                p.write_text(json.dumps(row))
        return result


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve(); self.r=VisualRuntime(self.root)
        env=patch.dict(os.environ,{cc.ROLES_ENV:str(self.r.roles_file),cc.CURRENT_ROLE_ENV:str(self.r.current_role_file)})
        env.start();self.addCleanup(env.stop)
    def test_proposal_wire_worker_and_native_boundary_show_return_no_reset(self):
        d=driver(self.root);b=LiveNativeBackend(self.r,self.root/'native',exporter=export_stub)
        stream=MemoryInput();s=Commands(stream,'live-test',turn_interruptions=True);rx=Commands(stream,'live-test',turn_interruptions=True)
        d.step(b)
        for epoch,text in ((1,'Turn away from the face and hold the surroundings in view.'),
                           (2,'Return to the self-facing view and continue the earlier discussion.')):
            s.send('turn',epoch,payload(text));row=rx.receive()
            self.assertEqual(validate_turn(row,visual_intents=True),proposal(text)['reply'])
            b.accept_visual_intent(row['epoch'],row['payload']['visual_intent'])
            d.step(b)
            actual=self.r.manifests[-1]['streaming_audio']['visual_conditioning']
            self.assertEqual(actual['target_epoch'],epoch)
            self.assertEqual(actual['visual_intent']['instruction'],text)
            self.assertIn(text,self.r.config.prompt)
            self.assertEqual(b.receipts[-1]['visual_conditioning']['visual_achievement'],'unverified')
        self.assertEqual(len({m['streaming_audio']['id'] for m in self.r.manifests}),1)
        self.assertEqual(len({m['streaming_audio']['profile_sha256'] for m in self.r.manifests}),1)
        self.assertEqual(self.r.start_calls,0)
        with self.assertRaises(ValueError):b.accept_visual_intent(1,intent('Old return.'))
        with self.assertRaises(ValueError):s.send('turn',1,payload('Old return.'))
        d.request_stop();d.step(b);b.retire();d.close()
    def test_null_continues_across_new_speech_epoch_without_front_reset(self):
        d=driver(self.root);b=LiveNativeBackend(self.r,self.root/'native',exporter=export_stub)
        b.accept_visual_intent(1,intent('Hold the surroundings in view.'));d.step(b)
        before=self.r.config.prompt
        b.accept_visual_intent(2,intent(None));d.step(b)
        self.assertEqual(self.r.config.prompt,before)
        record=cc.current_record()
        self.assertEqual((record['intent_epoch'],record['target_epoch']),(1,2))
        self.assertEqual(b.visual_events[-1]['visual_achievement'],'unverified')
        d.request_stop();d.step(b);b.retire();d.close()
    def test_new_session_starts_neutral_and_stale_intent_cannot_change_hold(self):
        old = LiveNativeBackend(self.r, self.root/'old-native', exporter=export_stub)
        movement = intent('Adjust the selfie view slightly left once, then hold the resulting view.')
        old.accept_visual_intent(1, movement)
        old.accept_visual_intent(2, intent(None))
        with self.assertRaises(ValueError):
            old.accept_visual_intent(1, intent('Continue panning.'))
        self.assertEqual(old._intent, movement)
        self.assertEqual((old._intent_epoch, old._intent_target), (1, 2))
        for role in ('speech', 'listening'):
            record = cc.role_record(role, visual_intents=True, active=old._intent,
                                    intent_epoch=1, target_epoch=2)
            self.assertEqual(record['visual_intent'], movement)
            self.assertIn('move once and then hold', cc.visual_prompt(role, old._intent))
        fresh_root = self.root/'fresh'
        fresh_root.mkdir()
        fresh_runtime = VisualRuntime(fresh_root)
        fresh = LiveNativeBackend(fresh_runtime, fresh_root/'native', exporter=export_stub)
        self.assertIsNone(fresh._intent)
        self.assertEqual((fresh._intent_epoch, fresh._intent_target), (0, 0))
        fresh.accept_visual_intent(1, intent(None))
        self.assertIsNone(fresh._intent)
        for role in ('speech', 'listening'):
            # New-session null uses the original calm self-facing role binding,
            # not another session's direction or a rewritten accepted baseline.
            self.assertEqual(cc.visual_prompt(role, fresh._intent), cc.role_prompt(role))
            self.assertIn('camera', cc.visual_prompt(role, fresh._intent))

    def test_legacy_route_refuses_visual_proposal_or_payload(self):
        with self.assertRaises(ValueError):proposal_reply(proposal('Show surroundings.'))
        with self.assertRaises(ValueError):proposal_reply(dict(proposal('Show surroundings.'),achieved=True),visual_intents=True)
        with self.assertRaises(ValueError):validate_turn(dict(payload=payload('Show surroundings.')))
        bad=payload('Show surroundings.');bad['achieved']=True
        with self.assertRaises(ValueError):validate_turn(dict(payload=bad),visual_intents=True)
    def test_role_transaction_rolls_back_all_files_on_rejection(self):
        self.r.continuation_file.write_text(json.dumps(dict(mode='independent')))
        old=(self.r.config,self.r.quality_profile,self.r.current_role_file.read_bytes(),self.r.quality_profile_file.read_bytes())
        record=cc.role_record('listening',visual_intents=True,active=intent('Show surroundings.'),intent_epoch=1,target_epoch=1)
        with patch.object(self.r,'_validate_quality_binding',side_effect=ValueError('fixture reject')):
            with self.assertRaises(ValueError):self.r._switch_role('listening',visual_binding=record)
        self.assertEqual((self.r.config,self.r.quality_profile,self.r.current_role_file.read_bytes(),self.r.quality_profile_file.read_bytes()),old)
    def test_native_proof_cannot_claim_prompt_applied_when_one_rank_differs(self):
        d=driver(self.root);b=LiveNativeBackend(self.r,self.root/'native',exporter=export_stub)
        b.accept_visual_intent(1,intent('Show surroundings.'))
        native=self.r._generate
        def corrupt(**kwargs):
            result=native(**kwargs);m=self.r.manifests[-1]
            p=self.r.coverage/f'conditioning-experiment-{m["session_id"]}-window-0-rank-1.json'
            row=json.loads(p.read_text());row['role_proof']['prompt_sha256']='0'*64;p.write_text(json.dumps(row))
            return result
        with patch.object(self.r,'_generate',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'visual prompt'):d.step(b)
        self.assertFalse(b.receipts)
        d.close()


class Tensor:
    def __init__(self,n=1900):self.shape=(n,)
    def detach(self):return self
    def cpu(self):return self
    def clone(self):return Tensor(self.shape[0])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name).resolve();self.r=VisualRuntime(root)
        env=patch.dict(os.environ,{cc.ROLES_ENV:str(self.r.roles_file),cc.CURRENT_ROLE_ENV:str(self.r.current_role_file)})
        env.start();self.addCleanup(env.stop)
        self.encoder=NS(parameters=lambda:iter(self.params));self.params=[object(),object()]
        self.stage=NS(conditioner=self.encoder,_h3_conditioning_cache={},_h3_role_condition_proofs={},
            _h3_conditioning_processor_digest='processor',_h3_visual_conditioner_identity=tuple(map(id,self.params)))
        self.calls=0
        def forward(batch,args):
            self.calls+=1;batch.prompt_embeds=[Tensor()];batch.extra['minimax_h3_text_token_tags']=Tensor(self.tokens);return batch
        self.tokens=1900;self.stage.forward=forward
        self.pipeline=NS(_stage_name_mapping={'conditioning_stage':self.stage},get_module=lambda name:self.encoder)
        for name,new in (('read_policy',lambda:dict(active_short_edge=1024)),('processor_digest',lambda s:'processor'),
            ('validate_accepted_batch',lambda *a:None),('read_accepted_profile',lambda:NS(digest='profile')),
            ('prepare_reference',lambda r,*a:r)):
            p=patch.object(ce,name,new);p.start();self.addCleanup(p.stop)
        p=patch('research.h3_personal.fast_ref2va.condition_key',lambda b:(b.prompt,));p.start();self.addCleanup(p.stop)
    def batch(self, text, epoch=1, role='speech'):
        self.r.current_role_file.write_text(json.dumps(cc.role_record(role,visual_intents=True,
            active=intent(text),intent_epoch=epoch,target_epoch=epoch)))
        return NS(prompt=cc.conditioning_prompt(role),num_frames=124,extra={'_h3_conditioning_sources':[NS(source='fixture')]},prompt_embeds=[])
    def test_same_prompt_hits_but_new_intent_same_role_encodes_and_budget_is_bounded(self):
        for n in range(4):
            batch=self.batch('Natural direction '+str(n),epoch=n+1)
            ce.refresh_visual_condition(self.pipeline,batch,None)
            self.assertTrue(batch.extra['_h3_visual_condition_refresh']['encoded'])
            ce.refresh_visual_condition(self.pipeline,batch,None)
            self.assertFalse(batch.extra['_h3_visual_condition_refresh']['encoded'])
        self.assertEqual(self.calls,4);self.assertEqual(len(self.stage._h3_conditioning_cache),4)
        with self.assertRaisesRegex(RuntimeError,'bank exhausted'):
            ce.refresh_visual_condition(self.pipeline,self.batch('Fifth direction',5),None)
        self.assertEqual(self.calls,4)
    def test_opt_in_cold_retention_preserves_exact_encoder_without_moving_or_loading(self):
        self.assertTrue(ce.retain_visual_conditioner(self.pipeline))
        self.assertIs(self.stage.conditioner,self.encoder)
        self.assertTrue(self.stage._h3_visual_conditioner_retained)
        self.assertEqual(self.calls,0)
        self.r.roles_file.write_text(json.dumps(cc.roles_manifest()))
        self.assertFalse(ce.retain_visual_conditioner(self.pipeline))
        self.assertEqual(self.calls,0)

    def test_missing_owned_encoder_fails_without_reload(self):
        self.stage.conditioner=None
        with self.assertRaisesRegex(RuntimeError,'implicit reload forbidden'):
            ce.refresh_visual_condition(self.pipeline,self.batch('Show surroundings.'),None)
        self.assertEqual(self.calls,0)
    def test_token_limit_does_not_truncate_or_publish_partial_cache(self):
        self.tokens=2400
        with self.assertRaisesRegex(RuntimeError,'no truncation'):
            ce.refresh_visual_condition(self.pipeline,self.batch('Show surroundings.'),None)
        self.assertFalse(self.stage._h3_conditioning_cache)
        self.assertFalse(self.stage._h3_conditioning_precomputing)


class CollectionTests(unittest.TestCase):
    def events(self):
        rows=[]
        for n in (1,2):
            rows.append(dict(type='h3_visual_intent_received',session='a'*32,target_epoch=n,
                intent_epoch=n,stage='instruction_received',instruction_sha256='d'*64,visual_achievement='unverified'))
            rows.append(dict(type='h3_visual_conditioning',session='a'*32,sequence=n-1,target_epoch=n,
                intent_epoch=n,stage='native_conditioning_applied',prompt_sha256='b'*64,
                presentation_epoch=n-1,presentation_outcome='unverified',visual_achievement='unverified'))
        rows.append(dict(type='h3_native_tts_receipts',receipts=[dict(epoch=1),dict(epoch=2)]))
        return rows
    def test_records_native_application_without_claiming_visual_achievement(self):
        from research.launch_h3_direct_conversation import collect_native_worker_receipts
        api=Mock();api.fetch_job_logs.return_value=[json.dumps(r) for r in self.events()]
        with tempfile.TemporaryDirectory() as tmp:
            collect_native_worker_receipts(api,NS(id='fixture'),tmp,time.monotonic()+2,visual_intents=True)
            value=json.loads((Path(tmp)/'visual-worker-evidence.json').read_text())
            self.assertTrue(value['native_application_verified'])
            self.assertEqual(value['visual_achievement'],'unverified')
            self.assertEqual(len(value['windows']),2)
    def test_missing_second_application_or_forged_achievement_is_rejected(self):
        from research.launch_h3_direct_conversation import collect_native_worker_receipts
        for kind in ('missing','achieved'):
            rows=self.events()
            if kind=='missing':rows.pop(3)
            else:rows[1]['visual_achievement']='achieved'
            api=Mock();api.fetch_job_logs.return_value=[json.dumps(r) for r in rows]
            with tempfile.TemporaryDirectory() as tmp, self.subTest(kind=kind),self.assertRaises(ValueError):
                collect_native_worker_receipts(api,NS(id='fixture'),tmp,time.monotonic()+2,visual_intents=True)


if __name__=='__main__':unittest.main()
