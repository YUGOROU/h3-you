"""Single-reference native proof validator regression; synthetic, no cloud."""
import copy
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from test_h3_ordered_rank_evidence import fixture
from research.hf_h3_streaming_trial import validate_ordered_rank_evidence
from research.h3_personal.ordered_references import content_identity, content_sha256
from research.h3_personal.candidate_references import build_candidate_binding
from research.h3_personal.conditioning_experiment import ordered_admission_bounds
from research.h3_personal.ordered_qwen_tokens import capture_qwen_presentation
from research.h3_personal.conversation_conditioning import role_prompt


def single_fixture(*, guide=True):
    profile, context, report = fixture()
    config = context['ordered_references']
    config['schema'] = 'h3-single-reference-v1'
    config['references'] = config['references'][:1]
    context.update(identity=content_identity(config), identity_sha256=content_sha256(config))
    binding = build_candidate_binding(config, profile)
    with patch('research.h3_personal.conversation_conditioning.read_ordered_context', return_value=context), patch(
            'research.h3_personal.conditioning_experiment.reference_short_edge', return_value=1024):
        prompt = role_prompt('speech')
        qwen = capture_qwen_presentation(NS(), NS(prompt=prompt,
            references=[NS(media_type='image', has_audio=False)], extra={}),
            [[1, 40, 100]], 2, [1000], list(range(1200)), [1]*2+[0]*1002+[1]*196)
    report.update(role_proof=dict(prompt_sha256=qwen['prompt_sha256']), qwen_tokens=qwen,
        ordered_reference_sha256=context['identity_sha256'], candidate_binding_sha256=binding['binding_sha256'],
        candidate_reference_binding=binding, R=1824, T=1200)
    g = report['ordered_geometry']
    g.update(identity_sha256=context['identity_sha256'], images=g['images'][:1], R=1824, T=1200,
             **ordered_admission_bounds(1024, references=1))
    segments = g['native_layout_reference_segments']
    g['native_layout_reference_segments'] = segments[:1] + (segments[2:] if guide else [])
    g['guide_segments_excluded'] = 2 if guide else 0
    layout = g['guide_layout']
    layout.update(image_rows=1824, guide_video_rows=784 if guide else 0, guide_audio_rows=74 if guide else 0)
    if not guide:
        layout['provenance'].update(guide_video_frames=0, guide_audio_latents=0)
    p = g['packed_rows']
    vr, ar = 1824 + (784 if guide else 0), 74 if guide else 0
    p.update(text_rows=1200, reference_rows=vr+ar, video_index_rows=vr+p['target_video_rows'],
             audio_index_rows=ar+p['target_audio_rows'],
             sequence_length=1200+vr+ar+p['target_video_rows']+p['target_audio_rows'])
    return profile, context, report


class Tests(unittest.TestCase):
    def validate(self, profile, context, report):
        with patch('research.h3_personal.conditioning_experiment.read_accepted_profile', return_value=profile):
            return validate_ordered_rank_evidence(report, context)

    def test_single_cold_and_guide_suffix_pass(self):
        for guide in (False, True):
            with self.subTest(guide=guide):
                value = self.validate(*single_fixture(guide=guide))
                self.assertEqual(value['geometry']['R'], 1824)
                self.assertEqual(value['geometry']['max_reference_rows'], 4096)

    def test_single_metadata_projection_preserves_independent_bounds(self):
        from research.h3_personal.ordered_metadata import project_metadata
        for guide in (False, True):
            profile, context, report = single_fixture(guide=guide)
            ordered = self.validate(profile, context, report)
            visual = dict(target_epoch=0, intent_epoch=0, prompt_sha256=report['qwen_tokens']['prompt_sha256'])
            source = dict(sequence=1, receipt_sha256='c'*64, visual_conditioning=visual,
                native_evidence=dict(ordered_references=ordered, ordered_both_ranks=True,
                    ordered_worker_memory=[dict(measurement='worker-current-all-models-not-reference-delta',
                        allocated_bytes=100, reserved_bytes=200) for _ in (0, 1)]))
            metadata = project_metadata(source, expected_identity=context['identity_sha256'])
            geometry = metadata['ordered_references']['geometry']
            self.assertEqual((geometry['max_reference_rows'], geometry['max_tokens']), (4096, 4607))
            self.assertEqual(metadata['ordered_references']['qwen_tokens']['policy']['max_reference_rows'], 8192)

    def test_extra_image_wrong_bounds_and_guide_order_rejected(self):
        profile, context, report = single_fixture()
        for mutate in (
            lambda g: g.update(max_reference_rows=8192),
            lambda g: g['images'].append(copy.deepcopy(g['images'][0])),
            lambda g: g['native_layout_reference_segments'].insert(1, copy.deepcopy(g['native_layout_reference_segments'][0])),
            lambda g: g['native_layout_reference_segments'].reverse(),
            lambda g: g.update(guide_segments_excluded=1),
        ):
            wrong = copy.deepcopy(report)
            mutate(wrong['ordered_geometry'])
            with self.assertRaises(ValueError):
                self.validate(profile, context, wrong)


if __name__ == '__main__':
    unittest.main()
