"""Synthetic rank JSON proof validation, NOT native execution evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research.h3_personal.accepted_profiles import choose_profile
from research.h3_personal.ordered_references import content_identity,content_sha256
from research.h3_personal.candidate_references import build_candidate_binding
from research.hf_h3_streaming_trial import validate_ordered_rank_evidence,read_evidence


def fixture():
    from research.h3_personal.conditioning_experiment import ordered_admission_bounds
    config=dict(schema='h3-ordered-references-v1',references=[
        dict(role='person',path='/private/synthetic-person.png',sha256='a'*64,width=9,height=16),
        dict(role='surroundings',path='/private/synthetic-scene.png',sha256='b'*64,width=14,height=11)])
    profile=choose_profile(ordered_source_sha256=('c'*64,),prompt_sha256='d'*64)
    context=dict(ordered_references=config,identity=content_identity(config),identity_sha256=content_sha256(config))
    images=[]
    for source,prepared,h,w in zip(config['references'],([1024,1824],[1312,1024]),(114,64),(64,82)):
        images.append(dict(role=source['role'],source_sha256=source['sha256'],
            original_size=[source['width'],source['height']],original_aspect=[source['width'],source['height']],
            stored_size=[source['width'],source['height']],oriented_size=[source['width'],source['height']],
            orientation_policy='png-exif-identity-only-v1',exif_present=False,exif_orientation=None,
            prepared_size=prepared,resize_policy=profile.resize_policy,crop='none',
            latent_frames=1,latent_height=h,latent_width=w,latent_rows=(h//2)*(w//2)))
    geometry=dict(schema='h3-ordered-native-geometry-v1',identity_sha256=context['identity_sha256'],
        candidate_visual_acceptance='unreviewed',images=images,R=3136,T=3000,
        admission_kind='candidate-engineering-not-native-capability',**ordered_admission_bounds(1024),
        conditioning_and_vae_rows='independent-native-quantities',reference_rows_source='native-layout-image-segments',
        guide_segments_excluded=0,native_layout_reference_segments=[dict(media_type='image',rows=i['latent_rows'],
            latent_shape=[i['latent_frames'],i['latent_height'],i['latent_width']]) for i in images])
    from test_h3_ordered_guide_layout import batch as guide_fixture
    from research.h3_personal.conditioning_experiment import ordered_layout_evidence,ordered_packed_evidence
    native=guide_fixture();layout=native.extra['minimax_h3_layout']
    from types import SimpleNamespace as NS
    layout.text_indices=NS(numel=lambda:3000);layout.sequence_length+=2000
    with patch('research.h3_personal.conditioning_experiment.reference_short_edge',return_value=1024):
        geometry.update(guide_layout=ordered_layout_evidence(native),packed_rows=ordered_packed_evidence(native),
            guide_segments_excluded=2,native_layout_reference_segments=[dict(media_type=s[0],rows=s[1],latent_shape=list(s[2]))
                for s in layout.reference_segments])
    from research.h3_personal.ordered_qwen_tokens import capture_qwen_presentation
    from research.h3_personal.conversation_conditioning import role_prompt
    with patch('research.h3_personal.conversation_conditioning.read_ordered_context',return_value=context),patch(
            'research.h3_personal.conditioning_experiment.reference_short_edge',return_value=1024):
        prompt=role_prompt('speech')
        qb=NS(prompt=prompt,references=[NS(media_type='image',has_audio=False)]*2,extra={})
        qwen=capture_qwen_presentation(NS(),qb,[[1,40,100],[1,40,180]],2,[1000,1800],list(range(3000)),
            [1]*2+[0]*1002+[1]*2+[0]*1802+[1]*192)
    binding=build_candidate_binding(config,profile)
    report=dict(role_proof=dict(prompt_sha256=qwen['prompt_sha256']),qwen_tokens=qwen,ordered_reference_sha256=context['identity_sha256'],candidate_visual_acceptance='unreviewed',
        candidate_binding_sha256=binding['binding_sha256'],candidate_reference_binding=binding,ordered_geometry=geometry,R=3136,T=3000)
    return profile,context,report


class Tests(unittest.TestCase):
    def test_actual_rows_and_qwen_tokens_are_independent(self):
        profile,context,report=fixture()
        with patch('research.h3_personal.conditioning_experiment.read_accepted_profile',return_value=profile):
            value=validate_ordered_rank_evidence(report,context)
        self.assertEqual(value['geometry']['R'],3136)
        self.assertEqual(value['geometry']['T'],3000)
        self.assertEqual(value['visual_acceptance'],'unreviewed')

    def test_missing_binding_wrong_order_crop_latent_and_token_bounds_rejected(self):
        profile,context,report=fixture()
        for mutate in (lambda x:x.pop('candidate_binding_sha256'),
            lambda x:x.update(candidate_visual_acceptance='accepted'),
            lambda x:x['ordered_geometry']['images'].reverse(),
            lambda x:x['ordered_geometry']['images'][0].update(crop='center'),
            lambda x:x['ordered_geometry']['images'][0].update(latent_height=115),
            lambda x:x['ordered_geometry']['images'][0].update(latent_rows=True),
            lambda x:x['ordered_geometry'].update(R=3137),
            lambda x:x['ordered_geometry'].update(T=10**9),
            lambda x:x['ordered_geometry'].update(max_tokens=10**9),
            lambda x:x['ordered_geometry']['native_layout_reference_segments'][1].update(rows=1),
            lambda x:x['ordered_geometry'].update(guide_segments_excluded=1)):
            wrong=copy.deepcopy(report);mutate(wrong)
            with patch('research.h3_personal.conditioning_experiment.read_accepted_profile',return_value=profile):
                with self.assertRaises(ValueError):validate_ordered_rank_evidence(wrong,context)

    def test_native_adapter_requires_candidate_both_rank_proof(self):
        from test_h3_live_native_backend import FakeRuntime,driver,export_stub
        from research.h3_personal.live_native_backend import LiveNativeBackend
        with tempfile.TemporaryDirectory() as root:
            runtime=FakeRuntime(root)
            runtime.ordered_reference_sha256='e'*64;runtime.candidate_binding_sha256='f'*64
            with driver(root) as d:
                backend=LiveNativeBackend(runtime,Path(root)/'native',exporter=export_stub)
                with self.assertRaisesRegex(ValueError,'candidate conditioning proof'):
                    d.step(backend)
            self.assertEqual(runtime.start_calls,0)

    def test_both_rank_candidate_match_and_missing_rank_evidence_fail_closed(self):
        import test_h3_streaming_windows as fixtures
        from research.h3_personal.streaming_admission import StreamingAdmission,NativeCommitEvidence
        from research.h3_personal.streaming_windows import ResamplingContext
        from research.h3_personal.streaming_trial import manifest_for,write_wav
        profile,context,report=fixture()
        controller=StreamingAdmission('c',context=ResamplingContext('same-rate-32k'))
        idle=controller.reserve(controller.inspect());controller.commit(idle,NativeCommitEvidence(124,None,'a'*64));controller.ack_decoded(idle,119)
        source=fixtures.source();fixtures.append_samples(source,136000);controller.attach(source)
        ticket=controller.reserve(controller.inspect())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wav=root/'prefix.wav';write_wav(wav,b'\x01\x00'*136000,32000)
            manifest=manifest_for(ticket,conversation_id='c',profile_sha256='a'*64,wav=wav)
            report['ordered_geometry']['guide_layout']['provenance'].update(owner_session_id=manifest['session_id'],window_index=1)
            for rank in (0,1):
                row=dict(schema=manifest['streaming_audio']['schema'],streaming=manifest['streaming_audio'],
                    native_window_id=manifest['session_id'],committed_frame=226,forwards=4,steps_held=4,
                    waveform_samples=165600,window_context_encodes=1,received_source_samples=136000,
                    canonical_sha256='b'*64,guide_digests=['c'*64]*3,prefix_pcm_sha256=ticket.decision.audio.prefix_sha256,
                    source_eos=False,padded_samples=0)
                (root/f'streaming-rank-{rank}-window-1.json').write_text(json.dumps(row))
                (root/f'h3-continuation-decode-rank-{rank}-window-0.json').write_text(json.dumps(dict(
                    session_id=manifest['session_id'],worker_rank=rank,committed_frames=226,emitted_frames=221,final=False)))
                conditioning=dict(report,current_role='speech',streaming_sequence=1,parameter_identity_preserved=True,cache_hit=True,
                    candidate_memory=dict(measurement='worker-current-all-models-not-reference-delta',allocated_bytes=100+rank,reserved_bytes=200+rank))
                (root/f'conditioning-experiment-{manifest["session_id"]}-window-0-rank-{rank}.json').write_text(json.dumps(conditioning))
            with patch('research.h3_personal.conversation_conditioning.read_ordered_context',return_value=context),patch(
                    'research.h3_personal.conditioning_experiment.read_accepted_profile',return_value=profile):
                result=read_evidence(root,manifest)
                self.assertTrue(result['evidence']['ordered_both_ranks'])
                path=root/f'conditioning-experiment-{manifest["session_id"]}-window-0-rank-1.json'
                bad=json.loads(path.read_text());bad['ordered_geometry']['T']=2999;bad['T']=2999;path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):read_evidence(root,manifest)
                bad.update(T=3000);bad['ordered_geometry']['T']=3000
                bad['candidate_memory']['reserved_bytes']+=1
                path.write_text(json.dumps(bad))
                rebound=read_evidence(root,manifest)
                self.assertNotEqual(result['receipt_sha256'],rebound['receipt_sha256'])
                bad.pop('ordered_reference_sha256');path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):read_evidence(root,manifest)


if __name__=='__main__':unittest.main()
