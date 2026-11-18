import copy
import json
import unittest

from research.h3_personal.accepted_profiles import choose_profile
from research.h3_personal.candidate_references import (
    build_candidate_binding, candidate_binding_sha256, validate_candidate_binding,
    validate_geometry_evidence, GEOMETRY_SCHEMA,
)


def config(root='/private/synthetic'):
    return dict(schema='h3-ordered-references-v1', references=[
        dict(role='person', path=root+'/person.png', sha256='a'*64, width=9, height=16),
        dict(role='surroundings', path=root+'/scene.png', sha256='b'*64, width=14, height=11)])


def profile(name='reference1024'):
    return choose_profile(name, ordered_source_sha256=('c'*64,), prompt_sha256='d'*64)


def binding():
    return build_candidate_binding(config(), profile())


def evidence(b):
    return dict(schema=GEOMETRY_SCHEMA, candidate_sha256=b['binding_sha256'],
        images=[dict(role='person', sha256='a'*64, original_size=[9,16],
                     prepared_size=[1024,1824], latent_rows=17),
                dict(role='surroundings', sha256='b'*64, original_size=[14,11],
                     prepared_size=[1312,1024], latent_rows=23)], R=40, T=53)


class CandidateReferencesTests(unittest.TestCase):
    def test_roundtrip_detached_base_unchanged(self):
        p=profile(); before=p.wire_record(); c=config(); b=build_candidate_binding(c,p)
        self.assertEqual(validate_candidate_binding(json.loads(json.dumps(b))),b)
        self.assertEqual(candidate_binding_sha256(b),b['binding_sha256'])
        self.assertEqual(p.wire_record(),before)
        self.assertEqual(b['content_visual_acceptance'],'unreviewed')
        text=json.dumps(b)
        for forbidden in ('quality_review_sha256','acceptance_source','prompt_sha256',
                          'ordered_source_sha256','expected_rt','expected_prepared_geometry','/private/'):
            self.assertNotIn(forbidden,text)
        c['references'][0]['width']=99
        self.assertEqual(b['ordered_content']['references'][0]['width'],9)
        parsed=validate_candidate_binding(b); parsed['base_settings']['geometry'][0]=99
        self.assertEqual(b['base_settings']['geometry'],[256,448])

    def test_portability_and_old_review_independence(self):
        a=binding(); b=build_candidate_binding(config('/other/private'),profile())
        self.assertEqual(a,b)
        p=choose_profile(ordered_source_sha256=('e'*64,),prompt_sha256='f'*64)
        self.assertEqual(a,build_candidate_binding(config(),p))
        self.assertNotEqual(a['binding_sha256'],build_candidate_binding(config(),profile('reference512'))['binding_sha256'])

    def test_content_and_settings_tamper_rejected(self):
        for mutate in (
            lambda b:b.update(content_visual_acceptance='accepted'),
            lambda b:b.update(schema='other'),
            lambda b:b.update(extra=True),
            lambda b:b['base_settings'].update(quality_review_sha256='a'*64),
            lambda b:b['base_settings'].update(window=True),
            lambda b:b['base_settings'].update(geometry=[256.0,448]),
            lambda b:b['base_settings'].update(geometry=(256,448)), 
            lambda b:b['ordered_content']['references'].reverse(),
            lambda b:b['ordered_content']['references'][0].update(sha256='e'*64),
            lambda b:b['ordered_content']['references'][0].update(path='/private/extra'),
            lambda b:b['ordered_content']['preprocessing'].update(crop='center'),
            lambda b:b.update(binding_sha256='0'*64),
            lambda b:b.update(ordered_content_sha256='0'*64)):
            b=binding();mutate(b)
            with self.assertRaises(ValueError):validate_candidate_binding(b)

    def test_bad_selection_no_count_order_or_type_coercion(self):
        for mutate in (lambda c:c['references'].reverse(),lambda c:c['references'].pop(),
                       lambda c:c['references'].append(copy.deepcopy(c['references'][0])),
                       lambda c:c['references'][0].update(width=True)):
            c=config();mutate(c)
            with self.assertRaises(ValueError):build_candidate_binding(c,profile())
        with self.assertRaises(ValueError):build_candidate_binding(config(),profile().wire_record())

    def test_actual_rows_not_inferred_from_png(self):
        b=binding();e=evidence(b)
        result=validate_geometry_evidence(e,b,max_reference_rows=40,max_tokens=53)
        self.assertEqual(result,e)
        result['images'][0]['latent_rows']=99
        self.assertEqual(e['images'][0]['latent_rows'],17)

    def test_bad_actual_geometry_and_rows_fail_closed(self):
        b=binding()
        for mutate in (
            lambda e:e['images'].reverse(),lambda e:e['images'].pop(),
            lambda e:e['images'][0].update(sha256='f'*64),
            lambda e:e['images'][0].update(original_size=[9,15]),
            lambda e:e['images'][1].update(prepared_size=[1024,1824]),
            lambda e:e['images'][0].update(prepared_size=[1024.0,1824]),
            lambda e:e['images'][0].update(latent_rows=True),
            lambda e:e['images'][0].update(latent_rows=0),
            lambda e:e.update(R=41),lambda e:e.update(T=0),
            lambda e:e.update(T=float('inf')),lambda e:e.update(R=True),
            lambda e:e.update(candidate_sha256='0'*64),lambda e:e.update(extra=1)):
            e=evidence(b);mutate(e)
            with self.assertRaises(ValueError):validate_geometry_evidence(e,b,max_reference_rows=100,max_tokens=100)

    def test_qwen_tokens_need_not_equal_or_exceed_vae_rows(self):
        b=binding();e=evidence(b);e['T']=23
        self.assertEqual(validate_geometry_evidence(e,b,max_reference_rows=40,max_tokens=100),e)

    def test_explicit_bounds_required_and_strict(self):
        b=binding();e=evidence(b)
        for r,t in ((39,53),(40,52),(True,100),(100,True),(0,100),(100,float('inf'))):
            with self.assertRaises(ValueError):validate_geometry_evidence(e,b,max_reference_rows=r,max_tokens=t)
        with self.assertRaises(TypeError):validate_geometry_evidence(e,b)


if __name__=='__main__':unittest.main()
