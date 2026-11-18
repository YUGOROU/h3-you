"""CPU diagnostic selection tests; no GPU or visual quality claim."""
import unittest

from test_h3_fast_continuation import Tensor
from test_h3_idle_visual_lineage import TimeOps
from research.h3_personal.streaming_native_owner import StreamingWindowOwner
from research.h3_personal.fast_continuation import SamplerContract


class IdleAnchorTests(unittest.TestCase):
    def owner(self, **kwargs):
        return StreamingWindowOwner('c', profile='p', height=2, width=2,
            max_windows=32, enabled=True, model_healthy=True, ops=TimeOps(), **kwargs)

    def start(self, owner, sequence, logical='idle', index=None, role='listening'):
        owner.start_window(str(sequence), canonical_audio=Tensor((2,207,32),
            [sequence+10]*(2*207*32)), logical_source_id=logical,
            logical_index=sequence if index is None else index, role=role,
            source_final=role=='speech', profile='p', height=2, width=2)
        return owner.begin_window()

    def commit(self, owner, token, sequence):
        result=owner.commit_window(token, clean_video=Tensor((37,96),
            [sequence+1]*(37*96)), clean_audio=owner.ops.copy(token.model.locked_audio),
            observed_sampler=SamplerContract())
        owner.ack_decoded_count(owner.committedF-5)
        return result

    def test_default_and_every_three_audio_offsets_counters_identical(self):
        owners=[self.owner(), self.owner(idle_anchor_interval=None),
                self.owner(idle_anchor_interval=3)]
        for i in range(8):
            tokens=[self.start(o,i) for o in owners]
            for j,(o,t) in enumerate(zip(owners,tokens)):
                expected=None if i==0 else (1 if j==2 and i in (3,6) else i)
                self.assertEqual(None if t.model.guide_video is None else
                    set(t.model.guide_video.values), None if expected is None else {expected})
                if i:
                    self.assertEqual(set(o._tail.values),{i})
                    self.assertIsNot(t.model.guide_video,o._tail)
                self.assertEqual(t.model.window,tokens[0].model.window)
                for field in ('temporal_offset','video_row_offset','audio_time_offset'):
                    self.assertEqual(getattr(t.model,field),getattr(tokens[0].model,field))
                self.assertEqual(t.model.locked_audio.values,tokens[0].model.locked_audio.values)
                self.assertEqual(None if t.model.guide_audio is None else t.model.guide_audio.values,
                    None if tokens[0].model.guide_audio is None else tokens[0].model.guide_audio.values)
            results=[self.commit(o,t,i) for o,t in zip(owners,tokens)]
            self.assertTrue(all(o.committedF==124+102*i and o.emittedF==119+102*i for o in owners))
            self.assertTrue(all(r.suffix.audio.values==results[0].suffix.audio.values for r in results))

    def test_source_role_invalidation_and_no_speech_anchor(self):
        o=self.owner(idle_anchor_interval=3)
        program=[('a',i,'listening') for i in range(4)]
        program += [('b',0,'listening'),('b',1,'listening')]
        program += [('s',0,'speech'),('c',0,'listening'),('c',1,'listening')]
        for seq,(logical,index,role) in enumerate(program):
            t=self.start(o,seq,logical,index,role)
            if index==0:
                self.assertIsNone(o._idle_anchor)
            if role=='speech':
                self.assertEqual(set(t.model.guide_video.values),{seq})
            self.commit(o,t,seq)
        self.assertEqual(set(o._idle_anchor.values),{8})

    def test_anchor_and_model_mutation_rejected(self):
        for mutation in ('digest','finite','shape','model','tail'):
            with self.subTest(mutation=mutation):
                o=self.owner(idle_anchor_interval=3)
                for i in range(3):
                    self.commit(o,self.start(o,i),i)
                if mutation=='model':
                    t=self.start(o,3)
                    t.model.guide_video.values[0]=-1
                    with self.assertRaisesRegex(ValueError,'conditioning mutated'):
                        self.commit(o,t,3)
                    continue
                if mutation=='digest': o._idle_anchor.values[0]=-1
                if mutation=='finite': o._idle_anchor.values[0]=float('nan')
                if mutation=='shape': o._idle_anchor.shape=(1,672)
                if mutation=='tail': o._tail.values[0]=-1
                with self.assertRaises(ValueError): self.start(o,3)

    def test_interval_strict(self):
        for value in (True,False,3.0,0,1,2,4,'3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.owner(idle_anchor_interval=value)


if __name__=='__main__':
    unittest.main()
