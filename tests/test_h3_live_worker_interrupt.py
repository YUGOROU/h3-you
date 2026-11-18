"""Opt-in worker interruption CPU tests, real scheduling and fake native math.

No provider, GPU, network, device, browser or acoustic playback is used.
"""
import hashlib
import importlib.util
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research.h3_personal.live_relay_commands import Commands
from research.h3_personal.live_relay_worker import (
    PI_MODEL, TurnInterruptionGate, run_worker)
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.streaming_windows import (
    ResamplingContext, NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256)
from research.h3_personal.tts_preroll import PCMPrerollPolicy


def turn(epoch):
    return dict(kind='turn', epoch=epoch, payload=dict(
        schema='h3-relay-typed-turn-v1', origin='fresh_typed_text', asr_used=False,
        director_model=PI_MODEL, reasoning_effort='none', reply=f'fresh reply {epoch}',
        previous_audio_reused=False))


def interrupt(epoch=1, generation=1):
    return dict(kind='interrupt_turn', epoch=epoch,
                payload=dict(schema='h3-turn-interruption-v1', presentation_epoch=generation))


class MemoryInput:
    def __init__(self):
        self.data=[]
    def send_input(self, data):
        self.data.append(data); return len(data)
    def read_input(self):
        return self.data.pop(0)


class CommandTests(unittest.TestCase):
    def test_explicit_monotonic_turn_interrupt_not_terminal_stop(self):
        transport=MemoryInput()
        sender=Commands(transport, 'session', turn_interruptions=True)
        receiver=Commands(transport, 'session', turn_interruptions=True)
        rows=[('turn',1,{}), ('interrupt_turn',1,interrupt()['payload']),
              ('turn',2,{}), ('interrupt_turn',2,interrupt(2,2)['payload']), ('stop',2,{})]
        for kind, epoch, payload in rows:
            sender.send(kind,epoch,payload)
            actual=receiver.receive()
            self.assertEqual((actual['kind'],actual['epoch'],actual['payload']),
                             (kind,epoch,payload))
        self.assertEqual(receiver._epoch,2)  # input epoch isn't presentation generation

    def test_opt_in_and_exact_payload_required(self):
        for enabled, epoch, payload in (
            (False,1,interrupt()['payload']), (True,0,interrupt()['payload']),
            (True,1,{'schema':'h3-turn-interruption-v1','presentation_epoch':True}),
            (True,1,{'schema':'h3-turn-interruption-v1','presentation_epoch':0}),
            (True,1,dict(interrupt()['payload'],extra=True)), (True,1,{})):
            transport=MemoryInput(); commands=Commands(transport,'session',turn_interruptions=enabled)
            commands.send('turn',1,{})
            with self.subTest(enabled=enabled,epoch=epoch,payload=payload), self.assertRaises(ValueError):
                commands.send('interrupt_turn',epoch,payload)
        transport=MemoryInput(); commands=Commands(transport,'session',turn_interruptions=True)
        commands.send('turn',1,{})
        commands.send('interrupt_turn',1,interrupt()['payload'])
        with self.assertRaises(ValueError):
            commands.send('interrupt_turn',1,interrupt(1,2)['payload'])
        commands.send('turn',2,{})
        with self.assertRaises(ValueError):
            commands.send('interrupt_turn',2,interrupt(2,1)['payload'])

    def test_gate_bounds_stale_queued_jobs_and_exactly_once_completion(self):
        gate=TurnInterruptionGate(); gate.accept(1); gate.invalidate(interrupt())
        self.assertFalse(gate.current(1))
        gate.complete(1, skipped=True)
        gate.accept(2); self.assertTrue(gate.current(2)); gate.complete(2)
        view=gate.snapshot()
        self.assertTrue(view['synthesis_drained'])
        self.assertEqual(view['skipped_tts_requests'],1)
        self.assertEqual(view['quarantined_tts_results'],0)
        with self.assertRaises(ValueError):gate.complete(2)
        with self.assertRaises(ValueError):gate.accept(3)

    def test_interrupt_mode_refuses_rtc_before_runtime_or_tts(self):
        with self.assertRaisesRegex(ValueError,'HTTPS paired'):
            run_worker(None,None,None,binding=None,context=None,root=None,
                       deadline=time.monotonic()+10,turn_interruptions=True,
                       media_transport='webrtc-rtp-v1')


class WorkerInterruptTests(unittest.TestCase):
    def run_case(self, *, during_forward, visual_intents=False, paired_cuts=False,
                 native_cancel=False, reject_cancel=False, stall_terminal=False):
        started=threading.Event(); forward=threading.Event(); applied=threading.Event()
        prefix_ready=threading.Event(); drained=threading.Event(); fresh_done=threading.Event()
        closed=threading.Event(); cancel_requested=threading.Event()
        calls=[]; manifests=[]; packets=[]
        case=self
        def selected_turn(epoch):
            row=turn(epoch)
            if visual_intents:
                row['payload'].update(schema='h3-relay-typed-turn-v2',visual_intent=dict(
                    schema='h3-visual-intent-v1',instruction='Hold the surroundings in view.' if epoch==1 else 'Return to self-facing view.'))
            return row
        class CommandFixture:
            def __init__(self,*args,**kwargs):
                case.assertEqual(kwargs,{'turn_interruptions':True}); self.index=0
            def receive(self):
                self.index+=1
                if self.index==1:return selected_turn(1)
                if self.index==2:
                    trigger=forward if during_forward else started
                    if not trigger.wait(3):raise TimeoutError('test trigger')
                    return interrupt()
                if self.index==3:
                    applied.set()  # previous return was processed by worker reader
                    return selected_turn(2)
                return dict(kind='stop',epoch=2,payload={})
            def close(self):pass
        class TTS:
            cancel_protocol='h3-voxcpm-cancel-v2'
            last_completion=None
            def request_cancel(self, epoch, deadline):
                case.assertEqual(epoch,1)
                cancel_requested.set()
                if reject_cancel:raise RuntimeError('test control unavailable')
                return dict(epoch=epoch,state='requested')
            def stream(self, text, epoch, deadline):
                self.last_completion=None
                calls.append(epoch)
                if epoch==1:
                    started.set()
                    if during_forward:
                        # Native first window becomes ready while source is OPEN.
                        for count in (64000,64000,64000,12128):
                            yield struct.pack('<f',.125)*count
                        prefix_ready.set()
                        if not forward.wait(3):raise TimeoutError('test forward')
                    if not applied.wait(3):raise TimeoutError('test interrupt')
                    if native_cancel:
                        if not cancel_requested.wait(3):raise TimeoutError('test control')
                        if stall_terminal:
                            if not closed.wait(3):raise TimeoutError('test close missing')
                            raise RuntimeError('terminal deliberately unavailable')
                        if not reject_cancel:
                            count=204128 if during_forward else 0
                            self.last_completion=dict(samples=count,chunks=4 if during_forward else 0,
                                sha256=hashlib.sha256(struct.pack('<f',.125)*count).hexdigest(),
                                finish_reason='cancelled',model_eos_verified=False,
                                cancel_protocol=self.cancel_protocol,epoch=epoch,native_closed=True)
                            drained.set()
                            return
                    yield struct.pack('<f',.25)*1024
                    drained.set()  # genuine iterator exhaustion occurs next
                else:
                    case.assertTrue(drained.is_set())  # one serial Gate, no early close
                    yield struct.pack('<f',.5)*40000
                    fresh_done.set()
            def close(self):closed.set()
        from research.h3_personal.live_native_backend import LiveNativeBackend as RealBackend
        class Backend:
            accept_visual_intent=RealBackend.accept_visual_intent
            def __init__(self,*args,**kwargs):
                self.visual_intents=visual_intents;self._intent_lock=threading.Lock()
                self._intent_epoch=self._intent_target=0;self._intent=None
                self.visual_events=[];self.receipts=[]
            def __call__(self, manifest):
                c=manifest['streaming_audio']; manifests.append(c)
                if visual_intents:
                    from research.h3_personal.conversation_conditioning import role_record
                    with self._intent_lock:
                        visual=role_record(c['role'],visual_intents=True,active=self._intent,
                            intent_epoch=self._intent_epoch,target_epoch=self._intent_target)
                    c['visual_conditioning']=visual
                    self.receipts.append(dict(sequence=c['sequence'],visual_conditioning=dict(
                        stage='native_conditioning_applied',visual_achievement='unverified',
                        presentation_outcome='unverified',**{k:visual[k] for k in ('target_epoch','intent_epoch','prompt_sha256')})))
                if c['sequence']==0:
                    ready=prefix_ready if during_forward else fresh_done
                    if not ready.wait(3):raise TimeoutError('test prefix')
                if c['role']=='speech' and c['logical_source_id']=='typed-1':
                    forward.set()
                    if not applied.wait(3):raise TimeoutError('test applied')
                end=c['committed_before']+(124 if c['sequence']==0 else 102)
                emitted=end-(0 if c['conversation_final'] else 5)
                before=0 if c['sequence']==0 else c['committed_before']-5
                if visual_intents:
                    self.receipts[-1].update(role=c['role'],committed_before=c['committed_before'],
                        committed_frame=end,emitted_frame=emitted,receipt_sha256='b'*64)
                return dict(video=b'fake encoded video',count=emitted-before,
                    committed_frame=end,emitted_frame=emitted,
                    conditioning_prefix_sha256=None if c['audio_offer'] is None else c['audio_offer']['prefix_sha256'],
                    receipt_sha256='b'*64)
            def retire(self):pass
            def close_unused(self):pass
        class Pump:
            def __init__(self,*args,**kwargs):pass
            def run(self, source):
                packets.extend(source)
                return dict(final=packets[-1].envelope.conversation_final)
        with tempfile.TemporaryDirectory() as root, \
             patch('research.h3_personal.live_relay_worker.Commands',CommandFixture), \
             patch('research.h3_personal.live_relay_worker.LiveNativeBackend',Backend), \
             patch('research.h3_personal.live_relay_worker.ProducerPump',Pump), \
             patch('research.h3_personal.live_relay_worker.role_profile_digest',return_value='a'*64), \
             patch('builtins.print'):
            result=run_worker(SimpleNamespace(quality_profile={},visual_intents=visual_intents),TTS(),Mock(),
                binding=LiveAVBinding('test','a'*64,7,max_segments=60),
                context=ResamplingContext('native-48k32k-prefix128',
                    NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256),
                root=root,deadline=time.monotonic()+12,tts_mode='native_voxcpm',
                tts_preroll=PCMPrerollPolicy(reserve_samples=1),turn_interruptions=True,
                **({'visual_intents':True} if visual_intents else {}),
                **({'paired_cuts':True} if paired_cuts else {}),
                **({'native_tts_cancel':True} if native_cancel else {}))
        self.assertTrue(result['final'])
        self.assertEqual(calls,[1,2])
        self.assertTrue(drained.is_set() and fresh_done.is_set())
        self.assertFalse(closed.is_set())  # turn interruption never closes healthy TTS
        self.assertEqual(len(result['native_tts_receipts']),2)
        self.assertEqual(result['turn_interruption']['interrupts'],1)
        self.assertEqual(result['turn_interruption']['quarantined_tts_results'],1)
        self.assertTrue(result['turn_interruption']['synthesis_drained'])
        self.assertFalse(any(t.name in ('h3-live-command-reader','h3-live-tts','h3-live-tts-cancel')
                             for t in threading.enumerate()))
        self.assertTrue(all(p.envelope.binding.epoch==7 for p in packets))
        return result, manifests, packets

    def test_native_cancel_retains_inflight_tail_and_all_fresh_pcm(self):
        if importlib.util.find_spec('httpx') is None:
            self.skipTest('worker serial transport needs httpx')
        if importlib.util.find_spec('av') is None:
            self.skipTest('worker AV sink needs PyAV')
        result, manifests, packets=self.run_case(during_forward=True, native_cancel=True,
                                                 visual_intents=True, paired_cuts=True)
        self.assertEqual(result['native_tts_receipts'][0]['completion'],'interrupted_protocol_END')
        self.assertEqual(result['native_tts_receipts'][0]['native_samples'],204128)
        old=[c for c in manifests if c['logical_source_id']=='typed-1']
        self.assertEqual(len(old),1)
        self.assertFalse(old[0]['source_final'])
        self.assertFalse(old[0]['audio_offer']['eos'])
        cut=[p.envelope for p in packets if p.envelope.presentation_cut is not None]
        self.assertEqual(len(cut),1)
        fresh=[r for p in packets for r in p.envelope.regions if r.source_epoch==2]
        self.assertEqual(min(r.source_offset for r in fresh),0)
        self.assertEqual(sum(r.sample_stop-r.sample_start for r in fresh),40000)
        exact=b''.join(p.pcm[(r.sample_start-p.envelope.frame_start*2000)*2:
                            (r.sample_stop-p.envelope.frame_start*2000)*2]
                      for p in packets for r in p.envelope.regions if r.source_epoch==2)
        self.assertEqual(exact,struct.pack('<h',16384)*40000)
        self.assertEqual(hashlib.sha256(exact).hexdigest(),
                         result['native_tts_receipts'][1]['converted_sha256'])
        self.assertEqual(result['native_tts_cancel_receipts'][0]['status'],'requested')

    def test_native_cancel_before_publication_does_not_attach_empty_source(self):
        result, manifests, packets=self.run_case(during_forward=False,native_cancel=True)
        self.assertEqual(result['native_tts_receipts'][0]['completion'],'interrupted_protocol_END')
        self.assertEqual(result['native_tts_receipts'][0]['native_samples'],0)
        self.assertFalse(any(c['logical_source_id']=='typed-1' for c in manifests))
        self.assertEqual(result['turns'],1)

    def test_terminal_timeout_fails_closed_and_joins_all_workers(self):
        with patch('research.h3_personal.live_relay_worker.NATIVE_CANCEL_TERMINAL_SECONDS',.01):
            with self.assertRaises(RuntimeError):
                self.run_case(during_forward=True,native_cancel=True,stall_terminal=True)
        self.assertFalse(any(t.name in ('h3-live-command-reader','h3-live-tts','h3-live-tts-cancel')
                             for t in threading.enumerate()))

    def test_control_rejection_drains_verified_end_without_close(self):
        result, manifests, packets=self.run_case(during_forward=True,native_cancel=True,reject_cancel=True)
        self.assertEqual(result['native_tts_receipts'][0]['completion'],'protocol_END')
        self.assertEqual(result['native_tts_cancel_receipts'][0]['status'],'bounded_drain')
        self.assertTrue(result['turn_interruption']['synthesis_drained'])

    def test_partial_quarantine_is_new_generation_without_skipping_native_forward(self):
        if importlib.util.find_spec('httpx') is None:
            self.skipTest('worker serial transport needs httpx')
        if importlib.util.find_spec('av') is None:
            self.skipTest('worker AV sink needs PyAV')
        result,manifests,packets=self.run_case(during_forward=True,visual_intents=True,paired_cuts=True)
        cut_packets=[p for p in packets if p.envelope.presentation_cut is not None]
        self.assertEqual(len(cut_packets),1)
        e=cut_packets[0].envelope;cut=e.presentation_cut
        self.assertEqual(e.presentation_epoch,1)
        self.assertEqual(cut.previous_epoch,0)
        self.assertEqual(cut.interrupted_source_epoch,1)
        self.assertEqual(cut.frame-e.frame_start,5)
        self.assertEqual(manifests[e.index]['role'],'listening')
        self.assertEqual(cut.frame,manifests[e.index]['committed_before'])
        self.assertTrue(all(r.sample_stop<=cut.frame*2000 for r in e.regions if r.kind=='speech'))
        fresh=[r for p in packets for r in p.envelope.regions if r.source_epoch==2]
        self.assertEqual(min(r.source_offset for r in fresh),0)
        self.assertEqual(sum(r.sample_stop-r.sample_start for r in fresh),40000)
        proof=result['visual_windows'][e.index]
        self.assertEqual(proof['presentation_epoch'],1)
        self.assertEqual(proof['visual_achievement'],'unverified')

    def test_visual_return_during_forward_replaces_intent_not_native_owner(self):
        if importlib.util.find_spec('httpx') is None:
            self.skipTest('worker serial transport needs httpx')
        if importlib.util.find_spec('av') is None:
            self.skipTest('worker AV sink needs PyAV')
        result, manifests, packets=self.run_case(during_forward=True,visual_intents=True)
        self.assertEqual([r['target_epoch'] for r in result['visual_intent_events']],[1,2])
        self.assertTrue(any(r['target_epoch']==2 for r in result['visual_windows']))
        self.assertTrue(all(r['visual_achievement']=='unverified' for r in result['visual_windows']))
        old=[c for c in manifests if c['role']=='speech' and c['logical_source_id']=='typed-1']
        self.assertEqual(len(old),1)
        self.assertEqual(old[0]['visual_conditioning']['intent_epoch'],1)
        returned=[c for c in manifests if c['visual_conditioning']['intent_epoch']==2]
        self.assertTrue(returned)
        self.assertTrue(all('Return' in c['visual_conditioning']['visual_intent']['instruction'] for c in returned))

    def test_started_tts_drains_end_and_stale_callback_never_attaches(self):
        result, manifests, packets=self.run_case(during_forward=False)
        self.assertEqual(result['turns'],1)
        self.assertFalse(any(c['logical_source_id']=='typed-1' for c in manifests))
        self.assertFalse(any(r.source_epoch==1 for p in packets for r in p.envelope.regions))
        fresh=[p for p in packets if any(r.source_epoch==2 for r in p.envelope.regions)]
        self.assertTrue(fresh)
        self.assertTrue(all(p.envelope.presentation_epoch==1 for p in fresh))

    def test_inflight_forward_once_then_one_old_epoch_quarantine_before_fresh_speech(self):
        result, manifests, packets=self.run_case(during_forward=True)
        self.assertEqual(result['turns'],2)
        old=[c for c in manifests if c['logical_source_id']=='typed-1']
        self.assertEqual(len(old),1)
        self.assertFalse(old[0]['audio_offer']['eos'])
        self.assertFalse(old[0]['source_final'])
        old_index=old[0]['sequence']
        quarantine=manifests[old_index+1]
        self.assertEqual(quarantine['role'],'listening')
        self.assertEqual(quarantine['previous_speech_interrupted']['source_id'],'typed-1')
        self.assertEqual(packets[old_index+1].envelope.presentation_epoch,0)
        self.assertEqual(sum(r.sample_stop-r.sample_start
            for r in packets[old_index+1].envelope.regions if r.source_epoch==1),10000)
        fresh_manifests=[c for c in manifests if c['logical_source_id']=='typed-2']
        first_fresh=fresh_manifests[0]['sequence']
        self.assertGreaterEqual(first_fresh,old_index+2)
        # TTS may still be draining; any additional native listening is CLEAN,
        # not another old-epoch quarantine or invented audio padding.
        for index in range(old_index+2, first_fresh):
            self.assertEqual(manifests[index]['role'],'listening')
            self.assertEqual(packets[index].envelope.presentation_epoch,1)
            self.assertFalse(any(r.source_epoch==1 for r in packets[index].envelope.regions))
        fresh=[p for p in packets if any(r.source_epoch==2 for r in p.envelope.regions)]
        self.assertTrue(all(p.envelope.presentation_epoch==1 for p in fresh))
        self.assertFalse(any(r.source_epoch==1 for p in fresh for r in p.envelope.regions))
        self.assertEqual(min(r.source_offset for p in fresh for r in p.envelope.regions
                             if r.source_epoch==2),0)
        self.assertEqual(sum(r.sample_stop-r.sample_start for p in fresh
                             for r in p.envelope.regions if r.source_epoch==2),40000)


if __name__=='__main__':unittest.main()
