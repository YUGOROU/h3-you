import io
import struct
import unittest

from _numpy_shim import install as _install_numpy

np = _install_numpy()

from research.h3_personal.zoom_input import EchoGuard,ZoomPCMReader,InputBlocked,source_contract,transcribe_extracted


def _pcm16(values):
    return struct.pack('<%dh' % len(values), *(int(v) for v in values))


class Extractor:
    def __init__(self,zero=False,fail=False):self.zero=zero;self.fail=fail;self.resets=0
    def reset(self):self.resets+=1
    def push(self,x):
        if self.fail:raise RuntimeError('model failure')
        return np.zeros_like(x) if self.zero else x*.5
    def flush(self):return np.zeros(0,np.float32)

class ZoomInputTests(unittest.TestCase):
    def setUp(self):self.time=[0.];self.g=EchoGuard(tail_seconds=1,clock=lambda:self.time[0])
    def arm(self):
        self.g.initial_idle()
        return self.g.arm(deadline=self.time[0]+10,manual_human_turn=True,all_participants_consented=True,offline=True)
    def reader(self,values=None,ex=None):
        data=_pcm16([1000]*640 if values is None else values)
        return ZoomPCMReader(io.BytesIO(data),ex or Extractor(),self.g,self.arm())
    def test_default_closed_and_live_requires_validated_tail(self):
        with self.assertRaises(InputBlocked):self.g.check(0)
        self.g.initial_idle()
        with self.assertRaises(InputBlocked):self.g.arm(deadline=10,manual_human_turn=True,all_participants_consented=True)
        for consent,human in ((False,True),(True,False)):
            with self.assertRaises(InputBlocked):self.g.arm(deadline=10,manual_human_turn=human,all_participants_consented=consent,offline=True)
    def test_halfduplex_tail_and_no_pending_replay(self):
        r=self.reader();self.assertEqual(r.read(2),_pcm16([500]))
        self.g.output_started()
        with self.assertRaises(InputBlocked):self.g.initial_idle()
        with self.assertRaises(InputBlocked):r.read(640)
        self.assertEqual(len(r.pending),0);self.assertTrue(r.failed)
        self.g.output_stopped();self.time[0]=.9
        with self.assertRaises(InputBlocked):self.g.arm(deadline=5,manual_human_turn=True,all_participants_consented=True,offline=True)
        self.time[0]=1.1
        epoch=self.g.arm(deadline=5,manual_human_turn=True,all_participants_consented=True,offline=True)
        fresh=ZoomPCMReader(io.BytesIO(_pcm16([2000])),Extractor(),self.g,epoch)
        self.assertEqual(fresh.read(640),_pcm16([1000]))
    def test_only_extracted_pcm_exact_clock(self):
        r=self.reader();out=b''
        while data:=r.read(640):out+=data
        self.assertEqual(np.frombuffer(out,dtype='<i2').tolist(),[500]*640)
        self.assertEqual(r.stats['captured_samples'],640);self.assertEqual(r.stats['submitted_samples'],640)
        self.assertTrue(r.eof);self.assertFalse(r.receipt()['speaker_identity_verified'])
    def test_non_target_fake_suppression_not_raw_fallback(self):
        r=self.reader(ex=Extractor(zero=True));self.assertEqual(r.read(640),bytes(640))
    def test_failure_no_raw_fallback(self):
        r=self.reader(ex=Extractor(fail=True))
        with self.assertRaisesRegex(RuntimeError,'model failure'):r.read(640)
        self.assertEqual(r.stats['submitted_samples'],0)
        with self.assertRaises(InputBlocked):r.read(640)
    def test_gate_change_during_source_read(self):
        class Raw:
            def read(inner,n):self.g.output_started();return bytes(640)
        r=ZoomPCMReader(Raw(),Extractor(),self.g,self.arm())
        with self.assertRaises(InputBlocked):r.read(640)
        self.assertEqual(r.stats['submitted_samples'],0)
    def test_expiry_alignment_and_no_clip(self):
        r=self.reader();self.time[0]=11
        with self.assertRaises(InputBlocked):r.read(640)
        r=self.reader()
        with self.assertRaises(ValueError):r.read(3)
        class Loud(Extractor):
            def push(self,x):return np.ones_like(x)*2
        r=self.reader(ex=Loud())
        with self.assertRaisesRegex(ValueError,'clip'):r.read(640)
    def test_asr_seam_extracted_coverage_and_origin(self):
        r=self.reader();source=source_contract(reference_sha256='a'*64,capture_source_sha256='b'*64)
        def asr(*,pcm,completion_policy):
            stream=pcm();out=b''
            while data:=stream.read(640):out+=data
            self.assertEqual(np.frombuffer(out,dtype='<i2').tolist(),[500]*640)
            return dict(final=True,errors=[],cleanup_complete=True,input_committed=True,completion_policy=completion_policy,
                        audio_sent_seconds=.04,finalized_audio_seconds=.04,text='テスト')
        result=transcribe_extracted(reader=r,source=source,asr=asr)
        self.assertEqual(result['origin'],'zoom_process_voxlock_endpoint_asr')
        source['speaker_identity_verified']=True
        with self.assertRaises(ValueError):transcribe_extracted(reader=r,source=source,asr=lambda **k:self.fail('must not call'))
    def test_asr_failure_disposes_reader_without_masking(self):
        r=self.reader();source=source_contract(reference_sha256='a'*64,capture_source_sha256='b'*64)
        def asr(**kw):raise OSError('primary network failure')
        with self.assertRaisesRegex(OSError,'primary network failure'):
            transcribe_extracted(reader=r,source=source,asr=asr)
        self.assertTrue(r.failed);self.assertFalse(r.pending)

    def test_asr_late_echo_invalidates_final(self):
        r=self.reader();source=source_contract(reference_sha256='a'*64,capture_source_sha256='b'*64)
        def asr(**kw):self.g.output_started();return {'text':'stale'}
        with self.assertRaises(InputBlocked):transcribe_extracted(reader=r,source=source,asr=asr)
