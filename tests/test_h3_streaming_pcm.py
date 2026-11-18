import dataclasses
import hashlib
import random
import threading
import unittest

from research.h3_personal.streaming_pcm import (
    PCMStreamBinding, PCMChunk, PCMIngress, PCMBackpressure,
)


def binding(**kw):
    return PCMStreamBinding('conversation', 'speech', 1, 'a'*64, **kw)


def ingress(b=None, **kw):
    return PCMIngress(b or binding(), max_bytes=128, max_chunk_bytes=32,
                      max_read_bytes=64, max_samples=64, **kw)


class PCMIngressTests(unittest.TestCase):
    def test_arbitrary_partitions_same_pcm_and_digest(self):
        pcm=bytes(range(128)); digest=hashlib.sha256(pcm).hexdigest()
        for seed in range(20):
            s=ingress(); rng=random.Random(seed); pos=index=0
            while pos < 64:
                n=min(rng.randint(1,16),64-pos)
                self.assertEqual(s.append(PCMChunk(s.binding,index,pos,pcm[pos*2:(pos+n)*2])),pos+n)
                pos+=n; index+=1
            source=s.seal(total_samples=64,sha256=digest)
            self.assertEqual(source.pcm,pcm)
            self.assertEqual(source.samples,64)
            self.assertIs(s.complete_source(),source)
            self.assertEqual(s.snapshot()['highwater_bytes'],128)
            self.assertIsNone(s.snapshot()['conditioning_ready_samples'])

    def test_starvation_is_not_silence_or_eos(self):
        s=ingress();s.append(PCMChunk(s.binding,0,0,b'\x01\x00'*4))
        result=s.read(0,8)
        self.assertEqual((result.status,result.pcm),('wait',b''))
        with self.assertRaises(RuntimeError):s.complete_source()
        self.assertEqual(s.read(0,4).pcm,b'\x01\x00'*4)
        s.seal(total_samples=4,sha256=hashlib.sha256(b'\x01\x00'*4).hexdigest())
        result=s.read(0,8)
        self.assertEqual((result.status,result.available_stop,result.pcm),('end',4,b'\x01\x00'*4))
        self.assertEqual(s.read(4,8).pcm,b'')

    def test_reject_gap_overlap_duplicate_and_stale_binding(self):
        s=ingress();good=PCMChunk(s.binding,0,0,b'\x01\x00'*4);s.append(good)
        bad=(good,PCMChunk(s.binding,1,5,b'00'),PCMChunk(s.binding,1,3,b'00'),
             PCMChunk(dataclasses.replace(s.binding,epoch=2),1,4,b'00'),
             PCMChunk(dataclasses.replace(s.binding,sample_rate=24000),1,4,b'00'),
             PCMChunk(dataclasses.replace(s.binding,voice_binding_sha256='b'*64),1,4,b'00'))
        for chunk in bad:
            with self.subTest(chunk=repr(chunk)):
                with self.assertRaises(ValueError):s.append(chunk)
                self.assertEqual(s.snapshot()['received_samples'],4)
        s.append(PCMChunk(s.binding,1,4,b'\x02\x00'))
        self.assertEqual(s.read(0,5).pcm,b'\x01\x00'*4+b'\x02\x00')

    def test_chunk_format_and_bounds_before_cursor_advance(self):
        s=ingress()
        for pcm in (b'',b'x',bytearray(b'00'),b'x'*34):
            with self.assertRaises(ValueError):s.append(PCMChunk(s.binding,0,0,pcm))
        for field in ('index','sample_start'):
            changes={field:True}
            with self.assertRaises(ValueError):s.append(dataclasses.replace(PCMChunk(s.binding,0,0,b'00'),**changes))
        self.assertEqual(s.snapshot()['received_samples'],0)
        with self.assertRaises(ValueError):s.read(0,33)

    def test_capacity_refuses_without_losing_received_pcm(self):
        s=PCMIngress(binding(),max_bytes=8,max_chunk_bytes=8,max_read_bytes=8,max_samples=4)
        s.append(PCMChunk(s.binding,0,0,b'12345678'))
        with self.assertRaises(PCMBackpressure):s.append(PCMChunk(s.binding,1,4,b'90'))
        self.assertEqual(s.snapshot()['received_samples'],4)
        self.assertEqual(s.read(0,4).pcm,b'12345678')

    def test_tiny_chunks_have_a_metadata_bound(self):
        s=ingress(max_chunks=2)
        s.append(PCMChunk(s.binding,0,0,b'12'))
        s.append(PCMChunk(s.binding,1,1,b'34'))
        with self.assertRaises(PCMBackpressure):s.append(PCMChunk(s.binding,2,2,b'56'))
        self.assertEqual(s.snapshot()['chunks'],2)
        self.assertEqual(s.read(0,2).pcm,b'1234')

    def test_bad_eos_is_terminal(self):
        for count,digest in ((3,hashlib.sha256(b'1234').hexdigest()),(2,'f'*64)):
            s=ingress();s.append(PCMChunk(s.binding,0,0,b'1234'))
            with self.assertRaises(ValueError):s.seal(total_samples=count,sha256=digest)
            with self.assertRaises(RuntimeError):s.complete_source()
            with self.assertRaises(RuntimeError):s.append(PCMChunk(s.binding,1,2,b'56'))

    def test_seal_once_and_append_after_seal(self):
        s=ingress();s.append(PCMChunk(s.binding,0,0,b'12'))
        digest=hashlib.sha256(b'12').hexdigest();s.seal(total_samples=1,sha256=digest)
        with self.assertRaises(RuntimeError):s.seal(total_samples=1,sha256=digest)
        with self.assertRaises(RuntimeError):s.append(PCMChunk(s.binding,1,1,b'34'))

    def test_deadline_and_cancel_are_terminal(self):
        now=[0.0];s=ingress(clock=lambda:now[0],timeout=1)
        s.append(PCMChunk(s.binding,0,0,b'12'));now[0]=1
        with self.assertRaises(TimeoutError):s.read(0,1)
        self.assertFalse(s._chunks)
        with self.assertRaises(RuntimeError):s.seal(total_samples=1,sha256=hashlib.sha256(b'12').hexdigest())
        s=ingress();s.cancel();s.cancel()
        with self.assertRaises(RuntimeError):s.append(PCMChunk(s.binding,0,0,b'12'))

    def test_seal_cancel_race_never_reopens(self):
        for _ in range(20):
            s=ingress();s.append(PCMChunk(s.binding,0,0,b'12'));barrier=threading.Barrier(3);errors=[]
            def seal():
                barrier.wait()
                try:s.seal(total_samples=1,sha256=hashlib.sha256(b'12').hexdigest())
                except RuntimeError:errors.append('cancelled')
            def cancel():barrier.wait();s.cancel()
            threads=[threading.Thread(target=seal),threading.Thread(target=cancel)]
            for t in threads:t.start()
            barrier.wait()
            for t in threads:t.join(2);self.assertFalse(t.is_alive())
            with self.assertRaises(RuntimeError):s.complete_source()
            self.assertEqual(s._state,'cancelled')

    def test_stereo_samples_are_frames_not_scalar_values(self):
        b=binding(channels=2);s=PCMIngress(b,max_bytes=16,max_chunk_bytes=8,max_read_bytes=16)
        s.append(PCMChunk(b,0,0,b'12345678'));s.append(PCMChunk(b,1,2,b'abcdefgh'))
        self.assertEqual(s.read(1,3).pcm,b'5678abcd')
        self.assertEqual(s.seal(total_samples=4,sha256=hashlib.sha256(b'12345678abcdefgh').hexdigest()).channels,2)

    def test_binding_and_deadline_validation(self):
        for changes in ({'epoch':True},{'sample_width':True},{'format':'float32'},
                        {'sample_rate':0},{'channels':3},{'source_id':''}):
            with self.assertRaises(ValueError):dataclasses.replace(binding(),**changes)
        for timeout in (True,0,301,float('nan'),float('inf')):
            with self.assertRaises(ValueError):ingress(timeout=timeout)

    def test_clock_rejects_nonfinite_and_backwards(self):
        for bad in (float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):ingress(clock=lambda:bad)
        for bad in (float('nan'),float('inf'),-1):
            now=[0];s=ingress(clock=lambda:now[0]);now[0]=bad
            with self.assertRaises(ValueError):s.append(PCMChunk(s.binding,0,0,b'12'))
            self.assertEqual(s._state,'failed')

    def test_binding_cannot_be_spoofed_by_equality(self):
        class AlwaysEqual:
            def __eq__(self, other):return True
        s=ingress()
        with self.assertRaises(ValueError):s.append(PCMChunk(AlwaysEqual(),0,0,b'12'))
        self.assertEqual(s.snapshot()['received_samples'],0)

    def test_context_cleanup_and_lazy_expiry_are_explicit(self):
        now=[0]
        with ingress(clock=lambda:now[0]) as s:
            s.append(PCMChunk(s.binding,0,0,b'12'));now[0]=121
            self.assertTrue(s._chunks)  # no implicit background timer
        self.assertFalse(s._chunks)
        self.assertEqual(s._state,'cancelled')

    def test_append_cancel_and_seal_races_are_serializable(self):
        for mode in ('cancel','seal'):
            for _ in range(10):
                s=ingress();s.append(PCMChunk(s.binding,0,0,b'12'))
                barrier=threading.Barrier(3);outcomes=[]
                def append():
                    barrier.wait()
                    try:s.append(PCMChunk(s.binding,1,1,b'34'));outcomes.append('append')
                    except RuntimeError:outcomes.append('rejected')
                def finish():
                    barrier.wait()
                    if mode=='cancel':s.cancel()
                    else:
                        try:s.seal(total_samples=1,sha256=hashlib.sha256(b'12').hexdigest())
                        except ValueError:outcomes.append('count-mismatch')
                ts=[threading.Thread(target=append),threading.Thread(target=finish)]
                for t in ts:t.start()
                barrier.wait()
                for t in ts:t.join(2);self.assertFalse(t.is_alive())
                if mode=='cancel':self.assertEqual(s._state,'cancelled')
                elif 'append' in outcomes:
                    self.assertEqual(s._state,'failed')
                    self.assertIn('count-mismatch',outcomes)
                else:self.assertEqual(s.complete_source().pcm,b'12')

    def test_prefix_digest_is_partition_and_seal_invariant(self):
        s=ingress();s.append(PCMChunk(s.binding,0,0,b'12345678'))
        for n in range(5):
            self.assertEqual(s.prefix_digest(n),hashlib.sha256(b'12345678'[:n*2]).hexdigest())
        before=s.prefix_digest(3)
        s.append(PCMChunk(s.binding,1,4,b'abcdefgh'))
        self.assertEqual(s.prefix_digest(3),before)
        s.seal(total_samples=8,sha256=hashlib.sha256(b'12345678abcdefgh').hexdigest())
        self.assertEqual(s.prefix_digest(3),before)
        with self.assertRaises(ValueError):s.prefix_digest(9)

    def test_pcm_hidden_in_repr(self):
        chunk=PCMChunk(binding(),0,0,b'PRIVATE_PCM')
        self.assertNotIn('PRIVATE_PCM',repr(chunk))
