"""CPU-only visual ancestry contracts; no device/model operations."""
import unittest

from test_h3_idle_visual_anchor import IdleAnchorTests
from research.h3_personal.visual_ancestry_policy import (
    ENV, VALUE, MANIFEST_KEY, CANDIDATE, COMPARATOR, parse_visual_ancestry_policy,
)


class VisualAncestryTests(IdleAnchorTests):
    def test_conversation_anchor_all_roles_and_previous_speech(self):
        owner=self.owner(visual_ancestry_policy=CANDIDATE)
        baseline=self.owner()
        program=[('a',0,'listening'), ('a',1,'listening'),
                 ('s',0,'speech'), ('s2',0,'speech'),
                 ('b',0,'listening'), ('s3',0,'speech'),
                 ('s4',0,'speech'), ('c',0,'listening'),
                 ('s5',0,'speech'), ('d',0,'listening')]
        for seq,(source,index,role) in enumerate(program):
            token=self.start(owner,seq,source,index,role)
            default=self.start(baseline,seq,source,index,role)
            expected=None if seq==0 else (1 if seq%3==0 else seq)
            guide=token.model.guide_video
            self.assertEqual(None if guide is None else set(guide.values),
                             None if expected is None else {expected})
            self.assertEqual(token.model.window,default.model.window)
            for field in ('temporal_offset','video_row_offset','audio_time_offset'):
                self.assertEqual(getattr(token.model,field),getattr(default.model,field))
            self.assertEqual(token.model.locked_audio.values,default.model.locked_audio.values)
            self.assertEqual(None if token.model.guide_audio is None else token.model.guide_audio.values,
                             None if default.model.guide_audio is None else default.model.guide_audio.values)
            evidence=owner.visual_ancestry_evidence
            self.assertEqual(evidence['sequence'],seq)
            self.assertEqual(evidence['policy'],CANDIDATE)
            self.assertEqual(evidence['kind'],'unconditioned' if seq==0 else
                             ('anchor' if seq%3==0 else 'latest'))
            self.assertEqual(evidence['anchor_sequence'],1 if seq else None)
            self.assertEqual(evidence['selected_sha256'],None if guide is None else owner.ops.digest(guide))
            if seq:
                self.assertIsNot(guide,owner._ancestry_anchor)
                self.assertEqual(set(owner._ancestry_anchor.values),{1})
                self.assertEqual(evidence['anchor_sha256'],owner.ops.digest(owner._ancestry_anchor))
            evidence['kind']='tampered'
            self.assertNotEqual(owner.visual_ancestry_evidence['kind'],'tampered')
            self.commit(owner,token,seq)
            self.commit(baseline,default,seq)
            self.assertEqual((owner.committedF,owner.emittedF),
                             (baseline.committedF,baseline.emittedF))

    def test_conversation_integrity_and_copies(self):
        for mutation in ('shape','hash','finite','tail','model'):
            with self.subTest(mutation=mutation):
                o=self.owner(visual_ancestry_policy=CANDIDATE)
                for i in range(3):
                    self.commit(o,self.start(o,i),i)
                if mutation=='shape': o._ancestry_anchor.shape=(1,672)
                if mutation=='hash': o._ancestry_anchor.values[0]=-1
                if mutation=='finite': o._ancestry_anchor.values[0]=float('nan')
                if mutation=='tail': o._tail.values[0]=-1
                if mutation=='model':
                    t=self.start(o,3)
                    t.model.guide_video.values[0]=-1
                    self.assertEqual(set(o._ancestry_anchor.values),{1})
                    with self.assertRaises(ValueError): self.commit(o,t,3)
                else:
                    with self.assertRaises(ValueError): self.start(o,3)

    def test_strict_constructor(self):
        for value in ('epoch-idle-every3-v1','',True,3,{}):
            with self.assertRaises(ValueError): self.owner(visual_ancestry_policy=value)
        with self.assertRaises(ValueError):
            self.owner(visual_ancestry_policy=CANDIDATE,idle_anchor_interval=3)


class PolicyTests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(ENV,'H3_VISUAL_ANCESTRY_DIAGNOSTIC')
        self.assertIsNone(parse_visual_ancestry_policy({},None))
        for arm in (COMPARATOR,CANDIDATE):
            manifest={MANIFEST_KEY:dict(schema=VALUE,policy=arm)}
            self.assertEqual(parse_visual_ancestry_policy(manifest,VALUE),arm)
        invalid=[({},VALUE), ({},''), ({MANIFEST_KEY:None},None),
                 ({MANIFEST_KEY:dict(schema=VALUE,policy=CANDIDATE)},None),
                 ({MANIFEST_KEY:dict(schema=VALUE,policy='unknown')},VALUE),
                 ({MANIFEST_KEY:dict(schema=VALUE,policy=CANDIDATE,extra=1)},VALUE),
                 ({MANIFEST_KEY:dict(policy=CANDIDATE)},VALUE),
                 ({MANIFEST_KEY:dict(schema='unknown',policy=CANDIDATE)},VALUE),
                 ({'nested':{MANIFEST_KEY:dict(schema=VALUE,policy=CANDIDATE)}},VALUE)]
        for manifest,env in invalid:
            with self.subTest(manifest=manifest,env=env), self.assertRaises(ValueError):
                parse_visual_ancestry_policy(manifest,env)


if __name__=='__main__':
    unittest.main()
