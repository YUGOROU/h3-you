"""Synthetic native layouts/owner tokens; no tensor/model execution."""
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from research.h3_personal import conditioning_experiment as c


def batch(*,guide=True):
    images=(('image',1824,(1,114,64)),('image',1312,(1,64,82)))
    from research.h3_personal.continuation_plan import Window
    window=Window(102,226,22) if guide else Window(0,124,0)
    token=NS(window=window,index=1,session_id='session',guide_video=NS(shape=(784,96)) if guide else None,
             guide_audio=NS(shape=(2,37,32)) if guide else None)
    owner=NS(session_id='session',_per_frame=112,pending_window=token)
    ctx=dict(token=token,owner=owner,manifest=dict(session_id='session',index=1))
    segments=images+(('audio',74,(0,0,0)),('video',784,(7,28,16))) if guide else images
    b=NS(extra=dict(minimax_h3_layout=NS(reference_segments=segments,
        num_condition_video_rows=3920 if guide else 3136,num_condition_audio_rows=74 if guide else 0,
        latent_height=28,latent_width=16,num_video_latent_frames=37),
        _h3_continuation=ctx),raw_latent_shape=(1,24,37,28,16),num_frames=124)
    layout=b.extra['minimax_h3_layout'];layout.num_audio_latents=207
    layout.text_indices=NS(numel=lambda:1000)
    layout.video_indices=NS(numel=lambda:layout.num_condition_video_rows+4144)
    layout.audio_indices=NS(numel=lambda:layout.num_condition_audio_rows+414)
    layout.sequence_length=1000+layout.num_condition_video_rows+layout.num_condition_audio_rows+4144+414
    return b


class Tests(unittest.TestCase):
    def test_actual_images_guides_and_canvas(self):
        b=batch();proof=c.ordered_layout_evidence(b)
        self.assertEqual((proof['image_rows'],proof['guide_video_rows'],proof['guide_audio_rows']),(3136,784,74))
        self.assertEqual(proof['provenance']['target_latent_shape'],[1,24,37,28,16])
        # Different valid canvas derives rows, not the former hardcoded112.
        b.raw_latent_shape=(1,24,37,32,16);ctx=b.extra['_h3_continuation'];ctx['owner']._per_frame=128
        ctx['token'].guide_video.shape=(896,96)
        layout=b.extra['minimax_h3_layout'];layout.reference_segments=layout.reference_segments[:3]+(('video',896,(7,32,16)),)
        layout.num_condition_video_rows=4032;layout.latent_height=32
        self.assertEqual(c.ordered_layout_evidence(b)['guide_video_rows'],896)

    def test_missing_extra_reorder_and_rows_reject(self):
        for mutate in (lambda s:s[:-1],lambda s:s+(s[-1],),lambda s:s[:2]+tuple(reversed(s[2:])),
                       lambda s:s[:2]+(('audio',57,(0,0,0)),s[3]),
                       lambda s:s[:3]+(('video',671,(6,28,16)),),
                       lambda s:s[:3]+(('video',672,(6,16,28)),)):
            b=batch();layout=b.extra['minimax_h3_layout'];layout.reference_segments=mutate(layout.reference_segments)
            with self.assertRaises(RuntimeError):c.ordered_layout_evidence(b)

    def test_owner_window_canvas_and_coverage_reject(self):
        mutations=(lambda b:setattr(b.extra['_h3_continuation']['owner'],'session_id','other'),
            lambda b:b.extra['_h3_continuation']['manifest'].update(index=2),
            lambda b:setattr(b.extra['_h3_continuation']['owner'],'pending_window',None),
            lambda b:setattr(b,'raw_latent_shape',(1,24,31,16,28)),
            lambda b:setattr(b.extra['_h3_continuation']['token'].guide_audio,'shape',(2,28,32)),
            lambda b:setattr(b.extra['minimax_h3_layout'],'num_condition_audio_rows',59))
        for mutate in mutations:
            b=batch();mutate(b)
            with self.assertRaises(RuntimeError):c.ordered_layout_evidence(b)

    def test_cold_and_first_window_require_no_guides(self):
        b=batch(guide=False);self.assertEqual(c.ordered_layout_evidence(b)['guide_video_rows'],0)
        b.extra.pop('_h3_continuation');self.assertEqual(c.ordered_layout_evidence(b)['image_rows'],3136)
        b=batch();b.extra.pop('_h3_continuation')
        with self.assertRaises(RuntimeError):c.ordered_layout_evidence(b)

    def test_candidate_export_uses_actual_rows_legacy_math_unchanged(self):
        b=batch();b.extra['minimax_h3_layout'].text_indices=NS(numel=lambda:1000)
        # A candidate context must carry its portable identity; the two-reference
        # fixture layout below therefore declares exactly two references.
        candidate={'identity':{'references':[{'role':'person'},{'role':'surroundings'}]}}
        with patch.object(c,'read_policy',return_value={'active_short_edge':1024}),patch.object(c,'_receipt_sizes',return_value=[]),patch(
                'research.h3_personal.conversation_conditioning.read_ordered_context',return_value=candidate):
            self.assertEqual(c.export_receipt(batch=b)['R'],3136)
        with patch.object(c,'read_policy',return_value={'active_short_edge':1024}),patch.object(c,'_receipt_sizes',return_value=[]),patch(
                'research.h3_personal.conversation_conditioning.read_ordered_context',return_value=None):
            self.assertEqual(c.export_receipt(batch=b)['R'],3920-7*112)

    @patch.object(c,'reference_short_edge',return_value=1024)
    def test_packed_actual_decomposition_and_bound(self,_edge):
        b=batch();proof=c.ordered_packed_evidence(b)
        self.assertEqual(proof['sequence_length'],9552)
        self.assertEqual(proof['reference_rows'],3994)
        self.assertEqual(proof['target_video_rows'],4144)
        self.assertEqual(proof['target_audio_rows'],414)
        for mutate in (lambda l:setattr(l,'sequence_length',9553),
                       lambda l:setattr(l,'video_indices',NS(numel=lambda:9999)),
                       lambda l:setattr(l,'num_audio_latents',208)):
            b=batch();mutate(b.extra['minimax_h3_layout'])
            with self.assertRaises(RuntimeError):c.ordered_packed_evidence(b)
        with patch('research.h3_personal.ordered_qwen_tokens.ordered_token_policy',return_value={'max_packed_rows':9552}):
            self.assertEqual(c.ordered_packed_evidence(batch())['sequence_length'],9552)
        with patch('research.h3_personal.ordered_qwen_tokens.ordered_token_policy',return_value={'max_packed_rows':9551}):
            with self.assertRaises(RuntimeError):c.ordered_packed_evidence(batch())

    def test_global_source_facade_identity(self):
        b=batch();ctx=b.extra['_h3_continuation'];token=ctx['token'];token.session_id='conversation';token.index=0
        global_token=NS(model=token,source_id=ctx['manifest']['session_id']);ctx.update(global_token=global_token,
            conversation_owner=NS(conversation_id='conversation',active_native=global_token,_sequence=3))
        ctx['owner'].token=global_token;ctx['manifest']['streaming_audio']=dict(id='conversation',sequence=3)
        self.assertEqual(c.ordered_layout_evidence(b)['provenance']['window_index'],3)
        ctx['manifest']['streaming_audio']['sequence']=4
        with self.assertRaises(RuntimeError):c.ordered_layout_evidence(b)
        ctx['manifest']['streaming_audio']['sequence']=3
        global_token.source_id='foreign-source'
        with self.assertRaises(RuntimeError):c.ordered_layout_evidence(b)


if __name__=='__main__':unittest.main()
