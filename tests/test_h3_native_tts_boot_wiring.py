"""CPU-only backend selection and real run_session wiring with inert dependencies."""
from concurrent.futures import Future
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_h3_direct_conversation_spec import spec
from research.hf_h3_direct_conversation import run_session, validate_spec
from research.bootstrap_h3_fast_ref2va import prepare_session_tts, session_tts_ownership, finish_bootstrap
from research.h3_personal.voxcpm_endpoint_protocol import PINS


def native_spec():
    value = spec()
    value['live_direct_session'].update(tts_mode='native_voxcpm', native_tts=dict(
        schema='h3-native-voxcpm-backend-v1', url='https://test.endpoints.huggingface.cloud',
        session='b'*32, boot='c'*32, expires_at=1500,
        pins=dict(PINS), preroll=dict(sample_rate=48000, reserve_samples=48000,
            max_reserve_samples=96000, max_wait_seconds=2.0)))
    return value


class NativeBootTests(unittest.TestCase):
    def test_spec_pure_distinct_session_and_strict_config(self):
        value = native_spec()
        with patch('time.time', side_effect=AssertionError):
            validate_spec(value)
        for mode in ('complete', 'sentence_incremental'):
            value = native_spec(); value['live_direct_session']['tts_mode'] = mode
            with self.assertRaises(ValueError): validate_spec(value)
        value = native_spec(); value['native_tts'] = value['live_direct_session']['native_tts']
        with self.assertRaises(ValueError): validate_spec(value)
        value = native_spec(); value['live_direct_session']['native_tts']['secret'] = 'forbidden'
        with self.assertRaises(ValueError): validate_spec(value)

    def test_rtc_requires_explicit_fixed_profile_and_disables_http_stream(self):
        value=native_spec();row=value['live_direct_session']
        row.update(media_transport='webrtc-rtp-v1',ice_profile='hf-cloudflare-turn-tls-v1')
        validate_spec(value)
        for field,bad in (('ice_profile','host-only-v1'),('continuous_av',True),('media_transport','auto')):
            broken=json.loads(json.dumps(value));broken['live_direct_session'][field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):validate_spec(broken)
        value['media_transport']='webrtc-rtp-v1'
        with self.assertRaises(ValueError):validate_spec(value)

    def test_factory_never_constructs_local_pool(self):
        future = Future(); future.set_result(object())
        with patch('research.h3_personal.native_tts_backend.prepare_direct_tts', return_value=future) as factory, patch('research.bootstrap_h3_fast_ref2va.ThreadPoolExecutor', side_effect=AssertionError):
            self.assertEqual(prepare_session_tts(Path('/unused'), native_spec()), (future, None))
            self.assertTrue(callable(factory.call_args.args[2]))

    def test_local_submission_failure_shuts_pool(self):
        pool = Mock(); pool.submit.side_effect = RuntimeError('submit')
        with patch('research.bootstrap_h3_fast_ref2va.ThreadPoolExecutor', return_value=pool):
            with self.assertRaises(RuntimeError):
                prepare_session_tts(Path('/unused'), {'live_voice_session': {}})
        pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_non_direct_still_submits_local(self):
        pool = Mock()
        with patch('research.bootstrap_h3_fast_ref2va.ThreadPoolExecutor', return_value=pool):
            future, owner = prepare_session_tts(Path('/unused'), {'live_voice_session': {}})
        self.assertIs(owner, pool)
        self.assertIs(future, pool.submit.return_value)
        self.assertEqual(pool.submit.call_count, 1)

    def test_early_native_failure_closes_completed_owner_once(self):
        client = Mock()
        future = Future(); future.set_result(client)
        with self.assertRaises(FileNotFoundError):
            with session_tts_ownership(future, None) as owner:
                finish_bootstrap(Path('/unused/missing-bootstrap'), {}, {}, None,
                                 None, None, 0, None, None, None, future, owner)
        client.close.assert_called_once_with()

    def test_early_legacy_failure_closes_late_owner_and_shuts_pool(self):
        client = Mock(); pool = Mock(); future = Future()
        future.set_running_or_notify_cancel()
        with self.assertRaises(RuntimeError):
            with session_tts_ownership(future, pool):
                raise RuntimeError('asset download')
        client.close.assert_not_called()
        pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        future.set_result(client)
        client.close.assert_called_once_with()

    def test_successful_handoff_does_not_close_twice(self):
        client = Mock(); pool = Mock(); future = Future(); future.set_result(client)
        with session_tts_ownership(future, pool) as owner:
            owner['transferred'] = True
            future.result().close()  # run_session owns cleanup after handoff
        client.close.assert_called_once_with()
        pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_cancellation_always_shuts_pool(self):
        pool = Mock(); future = Future()
        with self.assertRaises(KeyboardInterrupt):
            with session_tts_ownership(future, pool):
                raise KeyboardInterrupt()
        self.assertTrue(future.cancelled())
        pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def exercise(self, failure=None, expiry=1500, rtc=False):
        value = native_spec(); value['live_direct_session']['native_tts']['expires_at'] = expiry
        if rtc:value['live_direct_session'].update(media_transport='webrtc-rtp-v1',ice_profile='hf-cloudflare-turn-tls-v1')
        runtime = Mock(); runtime.quality_profile = SimpleNamespace(name='reference1024')
        client = Mock()
        future = Future(); future.set_result(client)
        worker = Mock()
        def work(*args, **kwargs):
            kwargs['on_ready']()
            if failure == 'worker': raise RuntimeError('worker')
            return dict(source_complete=True,media_complete=False,complete=False,
                        closing_tail_unverified=True,local_ipc_source_verified=True,
                        remote_original_pcm_verified=False) if rtc else 'done'
        worker.side_effect = work
        if failure == 'ready': client.read.side_effect = RuntimeError('ready')
        constructor = Mock(return_value=runtime)
        if failure == 'factory': constructor.side_effect = RuntimeError('factory')
        server = Mock(); server.__enter__ = Mock(return_value=server); server.__exit__ = Mock(return_value=False)
        binding = lambda session, digest, epoch, **kw: SimpleNamespace(session=session, profile_sha256=digest, epoch=epoch, **kw)
        modules = {
            'job_av_server': dict(JobAVServer=Mock(return_value=server)),
            'accepted_profiles': dict(choose_profile=lambda **kw: runtime.quality_profile),
            'conversation_conditioning': dict(prompt_sha=lambda x: 'p', role_profile_digest=lambda *a: 'd'),
            'ref2va_runtime': dict(sha=lambda p: 's'),
            'live_voice_runtime': dict(LiveVoiceRuntime=constructor),
            'streaming_windows': dict(ResamplingContext=Mock(), NATIVE_TORCH='t', NATIVE_TORCHAUDIO='a', REFERENCE_SHA256='r'),
            'live_relay_worker': dict(run_worker=worker),
            'paired_av': dict(LiveAVBinding=binding),
            'live_relay': dict(RelayConfig=Mock()),
        }
        if rtc:
            sidecar=Mock();sidecar.expires_at=1450;sidecar.deadline=550;sidecar.boot='e'*32;sidecar.port=7861
            sidecar.__enter__=Mock(return_value=sidecar)
            if failure=='rtc':sidecar.__enter__.side_effect=RuntimeError('sidecar')
            modules['webrtc_sidecar']=dict(LiveRTCSidecar=Mock(return_value=sidecar))
            modules['webrtc_job_gateway']=dict(RTCGateway=Mock())
        fake = {}
        for name, attrs in modules.items():
            fullname = 'research.h3_personal.' + name
            module = ModuleType(fullname); module.__dict__.update(attrs); fake[fullname] = module
        stdout = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, fake))
            stack.enter_context(patch.dict('os.environ', {'H3_DIRECT_PRODUCER_KEY':'p', 'H3_DIRECT_CONSUMER_KEY':'c','HF_TOKEN':'test-token'}, clear=True))
            stack.enter_context(patch('time.time', return_value=1000))
            stack.enter_context(patch('time.monotonic', return_value=100))
            stack.enter_context(patch('research.h3_personal.native_tts_backend.config_sha256', return_value='digest', create=True))
            stack.enter_context(redirect_stdout(stdout))
            paths = {name: Path('/unused') for name in ('reference', 'personal', 'base', 'smd')}
            if failure or expiry < 1120:
                with self.assertRaises((RuntimeError, TimeoutError)):
                    run_session('/unused/root', paths, value, {}, future, 1300)
            else:
                result=run_session('/unused/root', paths, value, {}, future, 1300)
                if rtc:self.assertTrue(result['source_complete'])
                else:self.assertEqual(result,'done')
        client.close.assert_called_once_with()
        if failure != 'factory': runtime.close.assert_called_once_with()
        if rtc:sidecar.close.assert_called_once_with()
        return worker, stdout, modules, client

    def test_ready_preroll_and_clamped_deadline(self):
        worker, stdout, modules, client = self.exercise()
        ready = json.loads(stdout.getvalue())
        self.assertEqual(ready['tts_mode'], 'native_voxcpm')
        self.assertEqual(ready['native_tts_config_sha256'], 'digest')
        self.assertEqual(ready['expires_at'], 1500)
        self.assertEqual(worker.call_args.kwargs['deadline'], 600)
        self.assertEqual(worker.call_args.kwargs['tts_preroll'].reserve_samples, 48000)
        client.read.assert_called_once_with('tts_ready', 600)

    def test_rtc_real_entry_wires_local_media_and_clamps_lease(self):
        worker,out,modules,_=self.exercise(rtc=True)
        ready,delivery=map(json.loads,out.getvalue().splitlines())
        self.assertFalse(delivery['media_complete'])
        self.assertEqual(ready['media_transport'],'webrtc-rtp-v1')
        self.assertEqual(ready['rtc_boot'],'e'*32)
        self.assertEqual(ready['expires_at'],1450)
        self.assertEqual(worker.call_args.kwargs['media_transport'],'webrtc-rtp-v1')
        sidecar=modules['webrtc_sidecar']['LiveRTCSidecar'].return_value
        self.assertIs(worker.call_args.kwargs['media_adapter'],sidecar.media_adapter)
        self.assertEqual(worker.call_args.kwargs['deadline'],550)
        self.assertIn('rtc_gateway',modules['job_av_server']['JobAVServer'].call_args.kwargs)
        self.exercise(rtc=True,failure='rtc')

    def test_cleanup_errors_and_short_expiry(self):
        for failure in ('factory', 'ready', 'worker'):
            with self.subTest(failure=failure): self.exercise(failure)
        worker, _, _, client = self.exercise(expiry=1100)
        worker.assert_not_called(); client.read.assert_not_called()


if __name__ == '__main__': unittest.main()
