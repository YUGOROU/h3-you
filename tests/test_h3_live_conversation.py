import hashlib
import tempfile
import unittest
from pathlib import Path

from research.h3_personal.live_conversation import LiveConversation
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.streaming_pcm import PCMIngress, PCMStreamBinding, PCMChunk
from research.h3_personal.streaming_windows import ResamplingContext


SHA = 'a'*64


class ConversationTests(unittest.TestCase):
    def make(self, root):
        return LiveConversation(Path(root)/'live',
            binding=LiveAVBinding('conversation', SHA, 0, max_segments=24),
            context=ResamplingContext(), max_idle_windows=12)

    def source(self, epoch, samples):
        b = PCMStreamBinding('conversation', f'speech-{epoch}', epoch, SHA)
        s = PCMIngress(b)
        pcm = (epoch.to_bytes(2, 'little'))*samples
        for offset in range(0, samples, 32000):
            s.append(PCMChunk(b, s.snapshot()['chunks'], offset,
                              pcm[offset*2:(offset+32000)*2]))
        return s, pcm

    def backend(self, manifest):
        c = manifest['streaming_audio']
        end = c['committed_before'] + (124 if c['sequence'] == 0 else 102)
        emitted = end-(0 if c['conversation_final'] else 5)
        before = 0 if c['sequence'] == 0 else c['committed_before']-5
        return dict(video=b'video', count=emitted-before,
                    committed_frame=end, emitted_frame=emitted,
                    conditioning_prefix_sha256=None if c['audio_offer'] is None else
                    c['audio_offer']['prefix_sha256'], receipt_sha256='b'*64)

    def test_two_turns(self):
        with tempfile.TemporaryDirectory() as root:
            d = self.make(root)
            packets, manifests = [], []
            def finish(p):
                manifests.append(p.manifest)
                packets.append(d.complete_native(p, self.backend(p.manifest)))
            finish(d.prepare())
            # Arrival after reservation must not replace the idle window.
            reserved = d.prepare()
            s, first = self.source(1, 70000)
            s.seal(total_samples=70000, sha256=hashlib.sha256(first).hexdigest())
            d.attach(s)
            finish(reserved)
            finish(d.prepare())
            self.assertEqual(d.inspect().status, 'end')
            with self.assertRaises(RuntimeError):
                d.request_stop()
            d.retire_speech()
            finish(d.prepare())
            finish(d.prepare())
            s, second = self.source(2, 90000)
            d.attach(s)
            # eos-only policy: pending speech stays listening until sealed.
            finish(d.prepare())
            s.seal(total_samples=90000, sha256=hashlib.sha256(second).hexdigest())
            finish(d.prepare())
            d.retire_speech()
            d.request_stop()
            finish(d.prepare())
            self.assertEqual(d.control.snapshot()['phase'], 'closed')
            idle = [m['streaming_audio'] for m in manifests if m['streaming_audio']['role']=='listening']
            self.assertEqual([(x['logical_source_id'], x['logical_index']) for x in idle],
                [('idle-epoch-0',0),('idle-epoch-0',1),('idle-epoch-1',0),
                 ('idle-epoch-1',1),('idle-epoch-1',2),('idle-epoch-2',0)])
            for epoch, original in ((1, first), (2, second)):
                actual = b''
                for p in packets:
                    for r in p.envelope.regions:
                        if r.source_epoch == epoch:
                            actual += p.pcm[(r.sample_start-p.envelope.sample_start)*2:
                                            (r.sample_stop-p.envelope.sample_start)*2]
                self.assertEqual(actual, original)
            self.assertTrue(packets[-1].envelope.conversation_final)
            self.assertFalse(any(p.envelope.conversation_final for p in packets[:-1]))

    def test_streaming_starvation_waits(self):
        from research.h3_personal.streaming_windows import (
            NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256)
        with tempfile.TemporaryDirectory() as root:
            d = LiveConversation(Path(root)/'live',
                binding=LiveAVBinding('conversation', SHA, 0, max_segments=24),
                context=ResamplingContext('native-48k32k-prefix128',
                    NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256))
            d.step(self.backend)
            s, pcm = self.source(1, 204128)
            d.attach(s)
            d.step(self.backend)
            self.assertEqual(d.inspect().status, 'wait')
            self.assertIsNone(d.prepare())
            with self.assertRaises(RuntimeError):
                d.request_stop()
            s.seal(total_samples=204128, sha256=hashlib.sha256(pcm).hexdigest())
            while d.inspect().status == 'ready':
                d.step(self.backend)
            self.assertEqual(d.inspect().status, 'end')
            d.retire_speech()
            d.request_stop()
            d.step(self.backend)

    def test_exact_eos_tail_survives_nonfinal_listening_return(self):
        with tempfile.TemporaryDirectory() as root:
            with self.make(root) as d:
                d.step(self.backend)
                s,pcm=self.source(1,204000)
                s.seal(total_samples=204000,sha256=hashlib.sha256(pcm).hexdigest());d.attach(s)
                voice=d.step(self.backend)
                self.assertEqual(d.control.snapshot()['emitted_frame'],221)
                d.retire_speech()
                p=d.prepare();self.assertFalse(p.manifest['streaming_audio']['conversation_final'])
                idle=d.complete_native(p,self.backend(p.manifest))
                actual=b''.join(packet.pcm[(r.sample_start-packet.envelope.sample_start)*2:
                    (r.sample_stop-packet.envelope.sample_start)*2] for packet in (voice,idle)
                    for r in packet.envelope.regions if r.source_epoch==1)
                self.assertEqual(actual,pcm)
                self.assertEqual(sum(r.sample_stop-r.sample_start for r in idle.envelope.regions if r.kind=='speech'),10000)
                d.request_stop();d.step(self.backend)
            self.assertEqual(d.control.snapshot()['phase'],'closed')
            self.assertFalse(list(d.root.glob('*.wav')))

    def test_complete_synthesis_eos_is_not_faked_as_open_frontier(self):
        with tempfile.TemporaryDirectory() as root:
            with self.make(root) as d:
                d.step(self.backend);s,pcm=self.source(1,70000)
                s.seal(total_samples=70000,sha256=hashlib.sha256(pcm).hexdigest());d.attach(s)
                p=d.prepare();self.assertTrue(p.manifest['streaming_audio']['audio_offer']['eos'])
                self.assertEqual(p.manifest['source_samples'],70000)

    def test_cancel_unstarted_source_frees_capacity_without_reusing_epoch(self):
        with tempfile.TemporaryDirectory() as root:
            with self.make(root) as d:
                d.step(self.backend);s,_=self.source(1,0);d.attach(s);d.cancel_pending_speech()
                self.assertEqual(d._capacity,0)
                again,_=self.source(1,0)
                with self.assertRaises(ValueError):d.attach(again)
                again.cancel()
                next_source,pcm=self.source(2,2000)
                next_source.seal(total_samples=2000,sha256=hashlib.sha256(pcm).hexdigest());d.attach(next_source)
                d.step(self.backend);d.retire_speech();d.request_stop();d.step(self.backend)

    def test_foreign_native_completion_poisoned_and_opaque_id_not_in_paths(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as root:
            with LiveConversation(Path(root)/'opaque',binding=LiveAVBinding('../../opaque',SHA,0,max_segments=24),
                                  context=ResamplingContext()) as d:
                p=d.prepare();self.assertNotIn('/',p.manifest['session_id'])
                self.assertEqual(p.manifest['streaming_audio']['id'],'../../opaque')
                with self.assertRaises(ValueError):d.complete_native(replace(p),self.backend(p.manifest))
                with self.assertRaises(RuntimeError):d.prepare()

    def test_backend_uncertainty_is_terminal(self):
        with tempfile.TemporaryDirectory() as root:
            d = self.make(root)
            def broken(m):
                raise OSError('uncertain native')
            with self.assertRaises(OSError):
                d.step(broken)
            with self.assertRaises(RuntimeError):
                d.prepare()

    def test_impossible_frame_budget_rejected_before_any_native_or_media(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'live'
            with self.assertRaises(ValueError):
                LiveConversation(target,binding=LiveAVBinding('conversation',SHA,0,max_segments=24,max_total_frames=634),
                                 context=ResamplingContext())
            self.assertFalse(target.exists())

    def test_bad_evidence_latches(self):
        with tempfile.TemporaryDirectory() as root:
            d = self.make(root)
            p = d.prepare()
            result = self.backend(p.manifest)
            result['count'] += 1
            with self.assertRaises(ValueError):
                d.complete_native(p, result)
            with self.assertRaises(RuntimeError):
                d.complete_native(p, self.backend(p.manifest))


if __name__ == '__main__':
    unittest.main()
