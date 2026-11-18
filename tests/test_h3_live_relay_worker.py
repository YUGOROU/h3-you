import hashlib
from pathlib import Path
import tempfile
import unittest
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch, Mock

from research.h3_personal.live_relay_worker import validate_turn, complete_pcm, PI_MODEL, run_worker, _conversation
from research.h3_personal.director_irodori import SOURCE, MODEL, CODEC
from research.h3_personal.live_conversation import LiveConversation
from research.h3_personal.paired_av import LiveAVBinding
from research.h3_personal.streaming_windows import ResamplingContext
from research.h3_personal.streaming_pcm import PCMStreamBinding, PCMIngress, PCMChunk


class WorkerTests(unittest.TestCase):
    def test_native_bridge_worker(self):
        import struct
        from research.h3_personal.tts_preroll import PCMPrerollPolicy
        for short, cancel in ((False, False), (True, False), (False, True)):
            with self.subTest(short=short, cancel=cancel), tempfile.TemporaryDirectory() as tmp:
                attached=threading.Event(); waited=threading.Event(); closed=threading.Event()
                done=threading.Event(); pending=threading.Event()
                turn=dict(self.row(), kind='turn', epoch=1)
                class Commands:
                    def __init__(self,*a): self.n=0
                    def receive(self):
                        self.n+=1
                        if self.n==1:return turn
                        if cancel:
                            if not waited.wait(2):raise TimeoutError()
                            return dict(kind='cancel_pending',epoch=1)
                        return dict(kind='stop',epoch=1)
                    def close(self):pass
                class TTS:
                    def stream(self,text,epoch,deadline):
                        self_args.append((text,epoch))
                        try:
                            if not pending.wait(2):raise TimeoutError()
                            yield struct.pack('<ff', .25, -.25)
                            if not short:
                                if not waited.wait(2):raise TimeoutError()
                                if cancel:
                                    # Let command reader publish cancellation before next chunk.
                                    time.sleep(.05)
                                yield struct.pack('<ff', .5, -.5)
                            done.set()
                        finally:closed.set()
                    def close(self):closed.set()
                self_args=[]
                driver=Mock(); driver._stream=None
                driver.inspect.side_effect=lambda:SimpleNamespace(status='waiting')
                observed=[]; consumed=[]; final=threading.Event()
                def attach(stream):
                    observed.append(done.is_set()); driver._stream=stream; attached.set()
                driver.attach.side_effect=attach
                driver.request_stop.side_effect=final.set
                def step(backend):
                    stream=driver._stream
                    if stream is None:
                        pending.set(); time.sleep(.001)
                    else:
                        result=stream.read(0, 4)
                        if result.status=='wait':
                            self.assertEqual(result.pcm,b''); waited.set(); return None
                        consumed.append(result.pcm)
                        driver._stream=None
                    return SimpleNamespace(envelope=SimpleNamespace(conversation_final=final.is_set()))
                driver.step.side_effect=step
                class Pump:
                    def __init__(self,*a,**kw):pass
                    def run(self,packets):
                        list(packets);return dict(final=True)
                with patch('research.h3_personal.live_relay_worker.Commands',Commands), \
                     patch('research.h3_personal.live_relay_worker.role_profile_digest',return_value='a'*64), \
                     patch('research.h3_personal.live_relay_worker.LiveNativeBackend'), \
                     patch('research.h3_personal.live_relay_worker.LiveConversation',return_value=driver), \
                     patch('research.h3_personal.live_relay_worker.ProducerPump',Pump), \
                     patch('builtins.print') as output:
                    def run():
                        return run_worker(SimpleNamespace(quality_profile={}),TTS(),Mock(),
                            binding=SimpleNamespace(profile_sha256='a'*64,session='test',max_segments=24),
                            context=None,root=tmp,deadline=time.monotonic()+5,
                            tts_mode='native_voxcpm',tts_preroll=PCMPrerollPolicy(reserve_samples=4 if short else 2))
                    if cancel:
                        with self.assertRaises(InterruptedError):run()
                        self.assertFalse(consumed)
                    else:
                        result=run()
                        self.assertIn('no-watermark',result['tts_availability'])
                        self.assertEqual(observed,[short])
                        receipts=result['native_tts_receipts']
                        self.assertEqual(len(receipts),1)
                        receipt=receipts[0]
                        self.assertEqual((receipt['source_id'],receipt['epoch']),('typed-1',1))
                        self.assertFalse(receipt['watermark_applied'] or receipt['model_eos_verified'])
                        native=struct.pack('<ff',.25,-.25)+(b'' if short else struct.pack('<ff',.5,-.5))
                        self.assertEqual(receipt['native_sha256'],hashlib.sha256(native).hexdigest())
                        self.assertEqual(receipt['converted_sha256'],hashlib.sha256(consumed[0]).hexdigest())
                        self.assertEqual(receipt['preroll']['short_sealed'],short)
                        saved=__import__('json').loads((Path(tmp)/'native-tts-receipts.json').read_text())
                        self.assertEqual(saved,receipts)
                        self.assertEqual(consumed,[struct.pack('<hh',8192,-8192) if short else
                            struct.pack('<hhhh',8192,-8192,16384,-16384)])
                    import json
                    events=[json.loads(call.args[0]) for call in output.call_args_list
                        if call.args and isinstance(call.args[0],str)
                        and '"h3_native_tts_receipts"' in call.args[0]]
                    saved=json.loads((Path(tmp)/'native-tts-receipts.json').read_text())
                    self.assertEqual(events,[dict(type='h3_native_tts_receipts',receipts=saved)])
                    self.assertLessEqual(len(saved),2)
                    self.assertNotIn('new reply',json.dumps(events))
                    if cancel:self.assertEqual(saved,[])
                    else:self.assertEqual(saved,result['native_tts_receipts'])
                    event_call=next(call for call in output.call_args_list
                        if call.args and '"h3_native_tts_receipts"' in str(call.args[0]))
                    self.assertTrue(event_call.kwargs['flush'])
                    self.assertTrue(closed.is_set())
                    self.assertEqual(self_args,[('new reply',1)])
                    self.assertTrue(pending.is_set())

    def test_native_mode_contract(self):
        from research.h3_personal.tts_preroll import PCMPrerollPolicy
        base=dict(binding=None,context=None,root=None,deadline=time.monotonic()+5)
        for mode, policy in (('native_voxcpm',None),('complete',PCMPrerollPolicy()),
                             ('sentence_incremental',PCMPrerollPolicy()),('native_voxcpm',{})):
            with self.subTest(mode=mode,policy=policy), self.assertRaises(ValueError):
                run_worker(None,None,None,tts_mode=mode,tts_preroll=policy,**base)
        with self.assertRaises(TypeError):
            run_worker(None,SimpleNamespace(synthesize=lambda:None),None,
                tts_mode='native_voxcpm',tts_preroll=PCMPrerollPolicy(),**base)

    def row(self):
        return dict(payload=dict(schema='h3-relay-typed-turn-v1',origin='fresh_typed_text',
            asr_used=False,reply='new reply',director_model=PI_MODEL,
            reasoning_effort='none',previous_audio_reused=False))

    def test_blocked_tts_stop_drains_but_cancel_discards(self):
        for kind, count, asr in (('stop',1,False),('stop',2,False),
                                ('cancel_pending',1,False),('cancel_pending',2,False),
                                ('stop',2,True),('cancel_pending',1,True)):
            with self.subTest(kind=kind,count=count), tempfile.TemporaryDirectory() as tmp:
                started=threading.Event(); release=threading.Event(); consumed=threading.Event()
                closed=threading.Event()
                turn=dict(self.row(),kind='turn',epoch=1)
                metadata=None
                if asr:
                    metadata=dict(policy='endpoint-native-v1',endpoint_name='h3-asr-t4-7c926aae18',
                        revision='a'*40,endpoint_source='native_completed',
                        completion_policy='input-finalized',capture_stopped=True,cleanup_complete=True)
                    turn['payload'].update(schema='h3-relay-asr-turn-v1',
                        origin='comet_process_audio_endpoint_asr',asr_used=True,asr=metadata.copy())
                class CommandFixture:
                    def __init__(self,*args):self.count=0
                    def receive(self):
                        self.count+=1
                        if self.count==1:return turn
                        if self.count==2 and count==2:
                            if not started.wait(2):raise TimeoutError()
                            return dict(turn,epoch=2)
                        if self.count==count+1:
                            if not started.wait(2):raise TimeoutError()
                            consumed.set()
                            return dict(kind=kind,epoch=count)
                        closed.wait(2)
                        raise EOFError()
                    def close(self):pass
                tts=Mock()
                def synthesize(*args):
                    started.set()
                    if not release.wait(3):raise TimeoutError()
                    return {}  # Receipt seam is checked independently below.
                tts.synthesize.side_effect=synthesize
                adapter=Mock();adapter.close.side_effect=closed.set
                driver=Mock();driver._stream=None
                driver.inspect.side_effect=lambda:SimpleNamespace(
                    status='end' if driver._stream is not None else 'waiting')
                driver.attach.side_effect=lambda stream:setattr(driver,'_stream',stream)
                driver.retire_speech.side_effect=lambda:setattr(driver,'_stream',None)
                final=threading.Event()
                driver.request_stop.side_effect=final.set
                def step(backend):
                    time.sleep(.001)
                    return SimpleNamespace(envelope=SimpleNamespace(conversation_final=final.is_set()))
                driver.step.side_effect=step
                class PumpFixture:
                    def __init__(self,*args,**kwargs):pass
                    def run(self,packets):
                        if not consumed.wait(2):raise AssertionError('reader blocked by TTS')
                        # Wait until interrupt publication (receive returns first).
                        time.sleep(.02)
                        release.set()
                        emitted=list(packets)
                        self.final=emitted[-1].envelope.conversation_final
                        return dict(final=self.final)
                converter=Mock()
                completed=Mock(return_value=b'\x01\x00'*100)

                with patch('research.h3_personal.live_relay_worker.Commands',CommandFixture), \
                     patch('research.h3_personal.live_relay_worker.role_profile_digest',return_value='a'*64), \
                     patch('research.h3_personal.live_relay_worker.LiveNativeBackend'), \
                     patch('research.h3_personal.live_relay_worker.LiveConversation',return_value=driver), \
                     patch('research.h3_personal.live_relay_worker.ProducerPump',PumpFixture), \
                     patch('research.h3_personal.live_relay_worker.complete_pcm',completed):
                    def run():
                        return run_worker(SimpleNamespace(quality_profile={}),tts,adapter,
                            binding=SimpleNamespace(profile_sha256='a'*64,session='test',max_segments=24),
                            context=None,root=tmp,deadline=time.monotonic()+10,canonical_pcm=converter,
                            expected_asr=metadata)
                    if kind=='cancel_pending':
                        with self.assertRaises(InterruptedError):run()
                        driver.attach.assert_not_called();completed.assert_not_called()
                        driver.request_stop.assert_not_called()
                    else:
                        result=run()
                        self.assertEqual(result['turns'],count)
                        self.assertEqual(result['origin'],turn['payload']['origin'])
                        prefix='turn' if asr else 'typed'
                        self.assertEqual([c.args[1] for c in tts.synthesize.call_args_list],
                            [f'{prefix}-{i}' for i in range(1,count+1)])
                        self.assertEqual([c.args[0].binding.source_id for c in driver.attach.call_args_list],
                            [f'{prefix}-{i}' for i in range(1,count+1)])
                        self.assertTrue(result['final'])
                        self.assertEqual(driver.attach.call_count,count)
                        self.assertEqual(completed.call_count,count)
                        self.assertEqual([call.args[0].binding.epoch for call in driver.attach.call_args_list],
                            list(range(1,count+1)))
                        driver.request_stop.assert_called_once()
                converter.assert_not_called()
                self.assertEqual(tts.synthesize.call_count,count if kind=='stop' else 1)
                self.assertFalse(any(t.name in ('h3-live-command-reader','h3-live-tts')
                                     for t in threading.enumerate()))

    def test_ready_only_after_both_constructors_before_threads_and_pump(self):
        for failure in ('backend', 'conversation', None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                events=[]
                backend=Mock(); driver=Mock(); adapter=Mock()
                def construct(name, result):
                    events.append(name)
                    if failure==name:raise RuntimeError('constructor failed')
                    return result
                def make_thread(*args, **kwargs):
                    thread=Mock(ident=None)
                    thread.start.side_effect=lambda:events.append(kwargs['name'])
                    thread.is_alive.return_value=False
                    return thread
                def pump(*args, **kwargs):
                    events.append('pump')
                    result=Mock();result.run.return_value={}
                    return result
                with patch('research.h3_personal.live_relay_worker.role_profile_digest',return_value='a'*64), \
                     patch('research.h3_personal.live_relay_worker.Commands'), \
                     patch('research.h3_personal.live_relay_worker.threading.Thread',side_effect=make_thread), \
                     patch('research.h3_personal.live_relay_worker.LiveNativeBackend',side_effect=lambda *a:construct('backend',backend)), \
                     patch('research.h3_personal.live_relay_worker._conversation',side_effect=lambda *a,**k:construct('conversation',driver)), \
                     patch('research.h3_personal.live_relay_worker.ProducerPump',side_effect=pump):
                    def run():
                        return run_worker(SimpleNamespace(quality_profile={}),Mock(),adapter,
                            binding=SimpleNamespace(profile_sha256='a'*64,session='test'),
                            context=None,root=tmp,deadline=time.monotonic()+10,
                            on_ready=lambda:events.append('ready'))
                    if failure:
                        with self.assertRaisesRegex(RuntimeError,'constructor failed'):run()
                        self.assertNotIn('ready',events)
                        self.assertNotIn('pump',events)
                    else:
                        run()
                        self.assertEqual(events,['backend','conversation','ready',
                            'h3-live-command-reader','h3-live-tts','pump'])
                adapter.close.assert_called_once()
                if failure=='conversation':
                    backend.close_unused.assert_called_once()
                    backend.abandon.assert_not_called()

    def test_worker_initialization_real_admission_bounds(self):
        binding=LiveAVBinding('worker-init','a'*64,0,
            max_segments=16,max_total_frames=1654)
        context=ResamplingContext()
        with tempfile.TemporaryDirectory() as tmp:
            # Reproduce both independent pre-fix constructor failures, without
            # replacing LiveConversation or its StreamingAdmission controller.
            for idle, timeout in ((16,300),(15,700),(16,700)):
                with self.subTest(idle=idle,timeout=timeout):
                    with self.assertRaises(ValueError):
                        LiveConversation(Path(tmp)/'old',binding=binding,context=context,
                            max_turns=2,max_idle_windows=idle,timeout=timeout)
            # The worker module imports time normally, so the patch must target that
            # exact object; `time.monotonic` is immutable and cannot be patched.
            # Admission clamps the timeout to the remaining lease, so the frozen clock
            # must move forward once the driver exists (clock granularity varies by OS).
            import research.h3_personal.live_relay_worker as worker_module
            for remaining in (700,42):
                with self.subTest(remaining=remaining):
                    driver=_conversation(Path(tmp)/str(remaining),binding=binding,
                        context=context,deadline=time.monotonic()+remaining)
                    # The remaining lease is sampled from the real clock, so allow sub-ms drift.
                    self.assertAlmostEqual(driver.control._deadline-driver.control._last_clock,
                        min(300,remaining), delta=0.01)
                    self.assertEqual(driver.inspect().status,'ready')
                    try:
                        self.assertIsInstance(driver,LiveConversation)
                        self.assertEqual(driver.control._max_windows,16)
                        self.assertEqual(driver.control._max_idle,15)
                        self.assertEqual(driver.control._max_turns,2)
                    finally:driver.close()
            with patch.object(worker_module.time,'monotonic',return_value=1000):
                with self.assertRaisesRegex(ValueError,'finite admission deadline'):
                    _conversation(Path(tmp)/'expired',binding=binding,
                        context=context,deadline=999)

    def test_native_asr_receipt_requires_exact_pin_and_no_transcript(self):
        metadata=dict(policy='endpoint-native-v1',endpoint_name='h3-asr-t4-7c926aae18',
            revision='a'*40,endpoint_source='native_completed',
            completion_policy='input-finalized',capture_stopped=True,cleanup_complete=True)
        row=self.row()
        row['payload'].update(schema='h3-relay-asr-turn-v1',
            origin='comet_process_audio_endpoint_asr',asr_used=True,asr=metadata.copy())
        self.assertEqual(validate_turn(row,metadata),'new reply')
        with self.assertRaises(ValueError):validate_turn(row)
        for key in metadata:
            bad=dict(row,payload=dict(row['payload'],asr=dict(metadata,**{key:'wrong'})))
            with self.subTest(key=key), self.assertRaises(ValueError):validate_turn(bad,metadata)
            with self.subTest(invalid_pin=key), self.assertRaises(ValueError):
                validate_turn(bad,bad['payload']['asr'])
        for key,value in [('transcript','private'),('asr_used',False),('origin','fresh_typed_text'),
                          ('previous_audio_reused',True),('director_model','fallback')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_turn(dict(row,payload=dict(row['payload'],**{key:value})),metadata)
        for revision in ('a'*39,'g'*40,'A'*40):
            bad=dict(metadata,revision=revision)
            with self.assertRaises(ValueError):
                validate_turn(dict(row,payload=dict(row['payload'],asr=bad)),bad)
        self.assertEqual(validate_turn(self.row(),metadata),'new reply')

    def test_truthful_origin_and_model(self):
        row=self.row();self.assertEqual(validate_turn(row),'new reply')
        for key,value in [('origin','asr'),('director_model','fallback'),('reply','x'*161),
                          ('asr_used',True),('previous_audio_reused',True),('text','private')]:
            bad=self.row();bad['payload'][key]=value
            with self.assertRaises(ValueError):validate_turn(bad)

    def test_missing_tts_fails_before_runtime_or_network(self):
        with self.assertRaisesRegex(TypeError,'missing interface'):
            run_worker(None,None,None,binding=None,context=None,root=None,deadline=0)

    def test_watermark_hash_and_complete_source_actual_driver(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'tts';directory.mkdir()
            wav=directory/'reply.wav';wav.write_bytes(b'fixture complete wav')
            reply='new reply'
            receipt=dict(watermark_applied=True,source_revision=SOURCE,model_revision=MODEL,
                codec_revision=CODEC,text_sha256=hashlib.sha256(reply.encode()).hexdigest(),
                sha256=hashlib.sha256(wav.read_bytes()).hexdigest())
            result=dict(turn_id='typed-1',wav=str(wav),receipt=receipt)
            original=b'\x01\x00'*70000
            pcm=complete_pcm(result,reply,'typed-1',directory,lambda p,d:original,100)
            binding=LiveAVBinding('worker-test','a'*64,0,max_segments=24)
            driver=LiveConversation(Path(tmp)/'driver',binding=binding,context=ResamplingContext())
            try:
                for epoch in (1,2):
                    b=PCMStreamBinding(binding.session,f'typed-{epoch}',epoch,binding.profile_sha256)
                    stream=PCMIngress(b)
                    for offset in range(0,len(pcm),64000):
                        stream.append(PCMChunk(b,offset//64000,offset//2,pcm[offset:offset+64000]))
                    stream.seal(total_samples=len(pcm)//2,sha256=hashlib.sha256(pcm).hexdigest())
                    driver.attach(stream)
                    while driver.inspect().status=='ready':
                        prepared=driver.prepare();m=prepared.manifest['streaming_audio']
                        end=m['committed_before']+(124 if m['sequence']==0 else 102)
                        emitted=end-5;before=0 if m['sequence']==0 else m['committed_before']-5
                        driver.complete_native(prepared,dict(video=b'fake',count=emitted-before,
                            committed_frame=end,emitted_frame=emitted,
                            conditioning_prefix_sha256=None if m['audio_offer'] is None else m['audio_offer']['prefix_sha256'],
                            receipt_sha256='b'*64))
                    self.assertEqual(driver.inspect().status,'end');driver.retire_speech()
            finally:driver.close()
            receipt['watermark_applied']=False
            with self.assertRaises(ValueError):complete_pcm(result,reply,'typed-1',directory,lambda p,d:original,100)
            receipt['watermark_applied']=True;wav.write_bytes(b'changed')
            with self.assertRaises(ValueError):complete_pcm(result,reply,'typed-1',directory,lambda p,d:original,100)


if __name__=='__main__':unittest.main()

