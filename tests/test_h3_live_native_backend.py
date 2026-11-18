from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest

from research.h3_personal.accepted_profiles import choose_profile
from research.h3_personal.conversation_conditioning import prompt_sha,role_prompt
from research.h3_personal.live_native_backend import LiveNativeBackend
from research.h3_personal.live_conversation import LiveConversation
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.streaming_windows import ResamplingContext
from research.h3_personal.streaming_pcm import PCMIngress,PCMStreamBinding,PCMChunk


def profile():return choose_profile(ordered_source_sha256=('a'*64,),prompt_sha256=prompt_sha('speech'))


class FakeRuntime:
    """Protocol fixture only: never pretends to load real H3 or prove quality."""
    def __init__(self,root):
        self._state='ready';self._continuation_enabled=True;self.conversation_roles=True
        self.quality_profile=profile();self.config=NS(prompt=role_prompt('speech'))
        self._lock=threading.RLock();self.coverage=Path(root)/'coverage';self.coverage.mkdir()
        self.continuation_file=self.coverage/'current.json';self.controls=[];self.manifests=[]
        self.owner=None;self.sequence=0;self.final=False;self.retired=set();self.start_calls=0

    def start(self):raise AssertionError('adapter must never start/reload H3')

    def _switch_role(self,role):
        assert json.loads(self.continuation_file.read_text())=={'mode':'independent'}
        self.config=NS(prompt=role_prompt(role));self.quality_profile=replace(self.quality_profile,prompt_sha256=prompt_sha(role))

    def _generate(self,*,continuation):
        assert continuation is True
        m=json.loads(self.continuation_file.read_text())
        if 'conversation_control' in m:
            c=m['conversation_control'];assert c['conversation_id']==self.owner
            if c['action']=='retire':assert self.final
            assert len(self.retired)<4 and self.owner not in self.retired
            self.controls.append(c);self.retired.add(self.owner)
            self.owner=None;self.sequence=0;self.final=False
            return NS(frames=[]),(1.,2.)
        c=m['streaming_audio'];assert c['sequence']==self.sequence
        assert c['id'] not in self.retired
        if self.owner is None:self.owner=c['id']
        assert self.owner==c['id']
        assert self.config.prompt==role_prompt(c['role'])
        self.manifests.append(m);self.sequence+=1;self.final=c['conversation_final']
        end=c['committed_before']+(124 if c['sequence']==0 else 102)
        emitted=end-(0 if c['conversation_final'] else 5)
        canonical=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
        for rank in (0,1):
            audio=c['audio_offer']
            row=dict(schema=c['schema'],streaming=c,native_window_id=m['session_id'],committed_frame=end,
                forwards=4,steps_held=4,waveform_samples=165600,window_context_encodes=1,
                received_source_samples=m['source_samples'],canonical_sha256=canonical,guide_digests=[None]*3,
                prefix_pcm_sha256=None if audio is None else audio['prefix_sha256'],
                source_eos=True if audio is None else audio['eos'],
                padded_samples=0 if audio is None else audio['eos_padding_native_samples'])
            p=self.coverage/f'streaming-rank-{rank}-window-{c["sequence"]}.json'
            with p.open('x') as f:json.dump(row,f)
            (self.coverage/f'h3-continuation-decode-rank-{rank}-window-0.json').write_text(json.dumps(dict(
                session_id=m['session_id'],worker_rank=rank,committed_frames=end,emitted_frames=emitted,final=self.final)))
            p=self.coverage/f'conditioning-experiment-{m["session_id"]}-window-0-rank-{rank}.json'
            with p.open('x') as f:json.dump(dict(current_role=c['role'],streaming_sequence=c['sequence'],
                parameter_identity_preserved=True,cache_hit=True),f)
        return NS(sequence=c['sequence']),(1.,2.)


def export_stub(result,path,start,stop,index):
    body=b'generated-video-'+str(index).encode();path.write_bytes(body);return body


def driver(root,session='live-test'):
    p=profile()
    return LiveConversation(Path(root)/session,binding=LiveAVBinding(session,p.digest,1,max_segments=24),
                            context=ResamplingContext(),max_idle_windows=12)


def source(d,epoch,samples=70000):
    b=PCMStreamBinding(d.binding.session,f'turn-{epoch}',epoch,d.binding.profile_sha256)
    s=PCMIngress(b);pcm=epoch.to_bytes(2,'little')*samples
    for start in range(0,samples,32000):
        s.append(PCMChunk(b,s.snapshot()['chunks'],start,pcm[start*2:min(start+32000,samples)*2]))
    s.seal(total_samples=samples,sha256=hashlib.sha256(pcm).hexdigest())
    return s,pcm


def two_turn_packets(d,backend,originals):
    yield d.step(backend)
    for epoch in (1,2):
        s,pcm=source(d,epoch);originals.append(pcm);d.attach(s)
        while d.inspect().status=='ready':yield d.step(backend)
        assert d.inspect().status=='end';d.retire_speech()
        # A real nonfinal return-to-listening window between fresh turns.
        yield d.step(backend)
    d.request_stop();yield d.step(backend)


class NativeAdapterTests(unittest.TestCase):
    def test_dynamic_turns_use_one_owner_and_real_role_variant_digest(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=FakeRuntime(root);d=driver(root)
            b=LiveNativeBackend(runtime,Path(root)/'native',exporter=export_stub)
            originals=[];packets=list(two_turn_packets(d,b,originals))
            self.assertEqual(len(packets),6)
            self.assertEqual([m['streaming_audio']['role'] for m in runtime.manifests],
                             ['listening','speech','listening','speech','listening','listening'])
            self.assertEqual(len({m['streaming_audio']['id'] for m in runtime.manifests}),1)
            self.assertEqual(runtime.start_calls,0)
            self.assertNotEqual(runtime.quality_profile.digest,d.binding.profile_sha256)
            b.retire();self.assertEqual(runtime.controls[0]['action'],'retire')
            d.close();self.assertFalse(list(d.root.glob('*.wav')))
            # Fresh owner on SAME warm runtime; generic immutable receipts archived.
            with driver(root,'second-conversation') as next_driver:
                next_backend=LiveNativeBackend(runtime,Path(root)/'second-native',exporter=export_stub)
                next_driver.request_stop();next_driver.step(next_backend);next_backend.retire()
            self.assertEqual(len(runtime.controls),2);self.assertEqual(runtime.start_calls,0)

    def test_encoder_failure_abandons_owner_without_starting_weights(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=FakeRuntime(root);d=driver(root)
            def broken(*a):raise OSError('synthetic encoding failure')
            b=LiveNativeBackend(runtime,Path(root)/'native',exporter=broken)
            with self.assertRaises(OSError):d.step(b)
            self.assertEqual(runtime._state,'degraded')
            with self.assertRaises(ValueError):LiveNativeBackend(runtime,Path(root)/'blocked',exporter=export_stub)
            with self.assertRaises(RuntimeError):d.prepare()
            b.abandon(reason='local-preparation-failure')
            self.assertEqual(runtime.controls[0]['action'],'abandon')
            self.assertEqual(runtime._state,'ready')
            self.assertEqual(runtime.start_calls,0);d.close()

    def test_uncertain_release_stays_quarantined_and_does_not_publish_independent(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=FakeRuntime(root);d=driver(root)
            b=LiveNativeBackend(runtime,Path(root)/'native',exporter=export_stub)
            d.step(b)
            original=runtime._generate
            def broken(*,continuation):
                if 'conversation_control' in json.loads(runtime.continuation_file.read_text()):
                    raise OSError('unknown cleanup outcome')
                return original(continuation=continuation)
            runtime._generate=broken
            with self.assertRaises(OSError):b.abandon(reason='local-sink-failure')
            self.assertEqual(runtime._state,'degraded')
            self.assertIn('conversation_control',json.loads(runtime.continuation_file.read_text()))
            with self.assertRaises(ValueError):LiveNativeBackend(runtime,Path(root)/'blocked',exporter=export_stub)
            d.close()

    def test_local_archive_repair_does_not_replay_native_release(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            runtime=FakeRuntime(root);d=driver(root);d.request_stop()
            b=LiveNativeBackend(runtime,Path(root)/'native',exporter=export_stub)
            d.step(b)
            with patch.object(Path,'rename',side_effect=OSError('synthetic local archive error')):
                with self.assertRaises(OSError):b.retire()
            self.assertEqual(runtime._state,'degraded');self.assertEqual(len(runtime.controls),1)
            b.retire()
            self.assertEqual(len(runtime.controls),1);self.assertEqual(runtime._state,'ready');d.close()

    def test_exclusive_adapter_claim_and_unused_release(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=FakeRuntime(root)
            b=LiveNativeBackend(runtime,Path(root)/'first',exporter=export_stub)
            with self.assertRaises(RuntimeError):LiveNativeBackend(runtime,Path(root)/'second',exporter=export_stub)
            b.close_unused()
            again=LiveNativeBackend(runtime,Path(root)/'third',exporter=export_stub);again.close_unused()
            self.assertFalse(runtime.controls)

    def test_real_runtime_quarantine_gate_permits_only_matching_control(self):
        from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
        class ReachedValidation(Exception):pass
        with tempfile.TemporaryDirectory() as root:
            r=LiveVoiceRuntime.__new__(LiveVoiceRuntime)
            r.continuation_file=Path(root)/'current.json';r._continuation_enabled=True
            r._h3_live_native_quarantine=NS(_id='owned')
            def reached():raise ReachedValidation()
            r._validate_quality_binding=reached
            for command in (dict(mode='independent'),dict(conversation_control=dict(action='abandon',conversation_id='foreign'))):
                r.continuation_file.write_text(json.dumps(command))
                with self.assertRaises(RuntimeError):r._generate(continuation=True)
            r.continuation_file.write_text(json.dumps(dict(conversation_control=dict(action='abandon',conversation_id='owned'))))
            with self.assertRaises(ReachedValidation):r._generate(continuation=True)

    def test_no_retire_before_final_and_no_replay_after_decode_uncertainty(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=FakeRuntime(root);d=driver(root)
            b=LiveNativeBackend(runtime,Path(root)/'native',exporter=export_stub)
            with self.assertRaises(RuntimeError):b.retire()
            p=d.prepare();result=b(p.manifest)
            with self.assertRaises(ValueError):b(p.manifest)
            def bad(*args):raise ValueError('missing both-rank receipt')
            # Next distinct request fails after native publication; cannot repeat.
            d.complete_native(p,result)
            p=d.prepare();b._read=bad
            with self.assertRaises(ValueError):b(p.manifest)
            with self.assertRaises(RuntimeError):b(p.manifest)
            b.abandon(reason='local-preparation-failure');d.close()
