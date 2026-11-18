"""CPU/offline launcher contracts only; no provider or realtime acceptance."""
import contextlib
from dataclasses import asdict
import hashlib
import io
import json
import inspect
import signal
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch, create_autospec

from research import launch_h3_direct_conversation as launch
from research.hf_h3_direct_conversation import validate_spec
from research.h3_personal.fast_ref2va import config_for_layout
from research.launch_h3_fast_ref2va import IDENTITY_PREFIX, REFERENCE_SHA


def spec(session='a'*32):
    return dict(fast_config=asdict(config_for_layout('portrait')), layout='portrait',
        attention_mode='fa4-packed', runtime_policy='resident-visual-cached',
        identity_prefix=IDENTITY_PREFIX, reference_sha256=REFERENCE_SHA,
        live_voice_session=dict(id=session,max_turns=2,fresh_input_required=True),
        live_direct_session=dict(schema='h3-live-direct-session-v1',session=session,
            port=7860,max_windows=16,max_frames=1654,lease_seconds=1200))


def ready():
    return dict(type='h3_direct_ready',status='ready',session='a'*32,
        profile_sha256='b'*64,epoch=1,max_segments=16,max_total_frames=1654,
        port=7860,expires_at=time.time()+100)


class LaunchTests(unittest.TestCase):
    def test_offline_default_no_input_or_run(self):
        with tempfile.TemporaryDirectory() as d, patch.object(launch,'run') as run, patch('builtins.input',side_effect=AssertionError('input offline')), contextlib.redirect_stdout(io.StringIO()):
            launch.main(['--directory',d])
            run.assert_not_called()

    def test_offline_existing_stage_inspected(self):
        with tempfile.TemporaryDirectory() as d, patch.object(launch,'inspect') as inspect, contextlib.redirect_stdout(io.StringIO()):
            (Path(d)/'deployment').mkdir()
            launch.main(['--directory',d])
            inspect.assert_called_once_with(Path(d)/'deployment')

    def test_run_reads_at_most_two_ram_lines(self):
        with patch.object(launch,'run') as run, patch('builtins.input',side_effect=['fresh one','fresh two',AssertionError('third read')]), contextlib.redirect_stdout(io.StringIO()):
            launch.main(['--directory','/unused','--run'])
            self.assertEqual(run.call_args.args[1],['fresh one','fresh two'])

    def test_asr_cli_offline_and_capture_skips_typed_input(self):
        with tempfile.TemporaryDirectory() as d, patch.object(launch, 'run') as run, patch('builtins.input', side_effect=AssertionError('typed input')), contextlib.redirect_stdout(io.StringIO()):
            args = ['--directory', d, '--capture-comet', '--asr-parent', '/parent']
            launch.main(args)
            run.assert_not_called()
            launch.main(args + ['--run'])
            run.assert_called_once_with(Path(d), capture_comet=True, asr_parent=Path('/parent'), allow_obs=False, obs_port=None)

    def test_asr_mode_rejections_before_operations(self):
        for kwargs, turns in (({'capture_comet': True}, None),
                              ({'asr_parent': '/parent'}, None),
                              ({'capture_comet': True, 'asr_parent': '/parent'}, []),
                              ({'capture_comet': True, 'asr_parent': '/parent'}, ['typed']),
                              ({'capture_comet': True, 'asr_parent': '/parent'}, None)):
            with patch.object(launch, 'canonical'), self.assertRaises(ValueError):
                launch.run('/outside-endpoint-private', turns, **kwargs)

    def test_input_and_obs_fail_before_dependencies(self):
        for turns in ([], [''], ['x'*2001], ['a','b','c'], [1]):
            with self.assertRaises(ValueError):
                launch.run('/unused',turns)
        with self.assertRaises(PermissionError):
            launch.run('/unused',['fresh'],obs_port=48124)

    def test_current_profile_and_direct_schema(self):
        s = spec()
        self.assertEqual(validate_spec(s)['lease_seconds'],1200)
        for key,value in [('port',True),('max_windows',17),('max_frames',1655),('lease_seconds',1201)]:
            broken = spec(); broken['live_direct_session'][key]=value
            with self.assertRaises(ValueError):
                validate_spec(broken)
        s['fast_config']['width'] = -1
        with self.assertRaises(ValueError):
            validate_spec(s)

    def test_scalar_ready_strict_and_expiry(self):
        self.assertEqual(launch.validate_ready(ready(),'a'*32)['port'],7860)
        for key,value in [('port',True),('epoch',1.0),('expires_at',float('nan')),('expires_at',time.time()-1),('profile_sha256','bad'),('session','c'*32),('extra','secret')]:
            row=ready(); row[key]=value
            with self.assertRaises(ValueError):
                launch.validate_ready(row,'a'*32)

    def test_ready_poll_finite_and_no_log_retention(self):
        api=SimpleNamespace(fetch_job_logs=lambda **kw: ['not json',json.dumps(ready())])
        result=launch.await_ready(api,SimpleNamespace(id='job'),'a'*32,time.monotonic()+1)
        self.assertEqual(result['type'],'h3_direct_ready')
        with self.assertRaises(TimeoutError):
            launch.await_ready(api,SimpleNamespace(id='job'),'a'*32,time.monotonic()-1)
        api=SimpleNamespace(fetch_job_logs=lambda **kw: [],inspect_job=lambda **kw: SimpleNamespace(status=SimpleNamespace(stage='ERROR')))
        with self.assertRaises(RuntimeError):
            launch.await_ready(api,SimpleNamespace(id='job'),'a'*32,time.monotonic()+1)

    def test_ready_recovers_partial_log_transport_failure(self):
        import httpx
        def partial():
            yield 'irrelevant log'
            raise httpx.RemoteProtocolError('incomplete chunked read')
        api=SimpleNamespace(
            fetch_job_logs=unittest.mock.Mock(side_effect=[partial(), [json.dumps(ready())]]),
            inspect_job=unittest.mock.Mock(return_value=SimpleNamespace(status=SimpleNamespace(stage='RUNNING'))))
        with patch.object(launch.time,'sleep'):
            result=launch.await_ready(api,SimpleNamespace(id='same-job'),'a'*32,time.monotonic()+10)
        self.assertEqual(result['status'],'ready')
        self.assertEqual(api.fetch_job_logs.call_args_list,
                         [unittest.mock.call(job_id='same-job',follow=False)]*2)
        api.inspect_job.assert_called_once_with(job_id='same-job')

    def test_ready_transport_failures_bounded_and_clean_poll_resets(self):
        import httpx
        for reset in (False, True):
            error=httpx.RemoteProtocolError('incomplete read')
            polls=[error, error, [], error, error, error] if reset else [error]*3
            api=SimpleNamespace(
                fetch_job_logs=unittest.mock.Mock(side_effect=polls),
                inspect_job=unittest.mock.Mock(return_value=SimpleNamespace(status=SimpleNamespace(stage='RUNNING'))))
            with patch.object(launch.time,'sleep'), self.assertRaises(httpx.RemoteProtocolError) as raised:
                launch.await_ready(api,SimpleNamespace(id='job'),'a'*32,time.monotonic()+10)
            self.assertIs(raised.exception,error)
            self.assertEqual(api.fetch_job_logs.call_count,len(polls))
            self.assertEqual(api.inspect_job.call_count,len(polls))

    def test_ready_retry_preserves_deadline_and_terminal_checks(self):
        import httpx
        api=SimpleNamespace(
            fetch_job_logs=unittest.mock.Mock(side_effect=httpx.RemoteProtocolError('read')),
            inspect_job=unittest.mock.Mock(return_value=SimpleNamespace(status=SimpleNamespace(stage='RUNNING'))))
        with patch.object(launch.time,'monotonic',side_effect=[0, 9, 10]), patch.object(launch.time,'sleep') as sleep:
            with self.assertRaises(TimeoutError):
                launch.await_ready(api,SimpleNamespace(id='job'),'a'*32,10)
            sleep.assert_called_once_with(1)
        self.assertEqual(api.fetch_job_logs.call_count,1)
        api.inspect_job.return_value.status.stage='ERROR'
        with self.assertRaisesRegex(RuntimeError,'Job ended'):
            launch.await_ready(api,SimpleNamespace(id='job'),'a'*32,time.monotonic()+10)

    def test_ready_does_not_retry_http_status_or_validation_errors(self):
        import httpx
        for status in (400,401,403,422,503):
            error=httpx.HTTPStatusError('status',request=httpx.Request('GET','https://offline.invalid'),
                                        response=httpx.Response(status))
            api=SimpleNamespace(fetch_job_logs=unittest.mock.Mock(side_effect=error))
            with self.assertRaises(httpx.HTTPStatusError):
                launch.await_ready(api,SimpleNamespace(id='job'),'a'*32,time.monotonic()+10)
            self.assertEqual(api.fetch_job_logs.call_count,1)
        row=ready(); row['session']='wrong'
        api=SimpleNamespace(fetch_job_logs=unittest.mock.Mock(return_value=[json.dumps(row)]))
        with self.assertRaises(ValueError):
            launch.await_ready(api,SimpleNamespace(id='job'),'a'*32,time.monotonic()+10)
        self.assertEqual(api.fetch_job_logs.call_count,1)

    def test_repo_creation_unknown_outcome_verified_without_post_retry(self):
        import httpx
        from unittest.mock import Mock
        repo = 'YUGOROU/h3-direct-' + 'a'*32
        for case in ('empty', 'attributes', 'public', 'wrong_id', 'wrong_owner',
                     'occupied', 'missing_siblings', 'lookup_error'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as d:
                stage = Path(d)/'deployment'; stage.mkdir()
                original = httpx.ReadTimeout('unknown creation outcome')
                stop = RuntimeError('fresh upload reached')
                info = SimpleNamespace(id=repo, private=True, siblings=[])
                if case == 'attributes':
                    info.siblings = [SimpleNamespace(rfilename='.gitattributes')]
                elif case == 'public': info.private = False
                elif case == 'wrong_id': info.id = repo + '-other'
                elif case == 'wrong_owner': info.id = 'other/' + repo.split('/')[1]
                elif case == 'occupied': info.siblings = [SimpleNamespace(rfilename='runtime.py')]
                elif case == 'missing_siblings': del info.siblings
                api = SimpleNamespace(
                    whoami=Mock(return_value=dict(name='YUGOROU', isPro=True)),
                    list_jobs=Mock(return_value=[]),
                    repo_info=Mock(side_effect=RuntimeError('lookup') if case == 'lookup_error' else None,
                                   return_value=info),
                    create_commit=Mock(side_effect=stop))
                hf = ModuleType('huggingface_hub'); hf.CommitOperationAdd = lambda **kw: kw
                helper = ModuleType('research.h3_personal.direct_job_api'); helper.submit_job = Mock()
                control = Mock(); control.post.side_effect = original
                control.__enter__ = Mock(return_value=control)
                control.__exit__ = Mock(return_value=False)
                with patch.dict(sys.modules, {'huggingface_hub': hf, helper.__name__: helper}), \
                     patch.object(launch, 'canonical'), \
                     patch.object(launch, 'inspect', return_value=({'publication_repo': repo}, {'files': []})), \
                     patch('research.launch_h3_live_voice.pre_submit'), \
                     patch.object(httpx, 'Client', return_value=control):
                    with self.assertRaises(Exception) as raised:
                        launch.submit(api, stage, 'token', 'producer', 'consumer')
                accepted = case in ('empty', 'attributes')
                self.assertIs(raised.exception, stop if accepted else original)
                control.post.assert_called_once()
                self.assertEqual(api.repo_info.call_args_list[0],
                                 unittest.mock.call(repo, repo_type='dataset'))
                self.assertEqual(api.create_commit.call_count, int(accepted))
                helper.submit_job.assert_not_called()
                self.assertFalse((Path(d)/'submission_guard.json').exists())

    def test_durable_guard_exclusive_and_private(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'guard.json'
            launch.durable_json(p,{'state':'unknown_no_replay'})
            self.assertEqual(p.stat().st_mode & 0o777,0o600)
            with self.assertRaises(FileExistsError):
                launch.durable_json(p,{})

    def test_run_ready_attestation_open_consumer_and_finally_cancel(self):
        from research import run_h3_relay_conversation as controller
        from research.h3_personal.job_av_endpoint import attest_job
        for scenario in ('success', 'ready', 'inspect', 'cleanup', 'sigterm', 'asr', 'asr_failure'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as d:
                events=[]; r=ready(); job=SimpleNamespace(id='job')
                failure = TimeoutError('unready')
                def inspect_job(**kw):
                    if scenario in ('inspect', 'cleanup'):
                        raise OSError('inspection unavailable')
                    return SimpleNamespace(status=SimpleNamespace(stage='RUNNING'))
                def cancel(**kw):
                    events.append('cancel')
                    if scenario == 'cleanup': raise OSError('cancel unavailable')
                api=SimpleNamespace(inspect_job=inspect_job, cancel_job=cancel)
                hf=ModuleType('huggingface_hub'); hf.HfApi=lambda **kw:api; hf.set_client_factory=lambda f:None
                http=ModuleType('httpx')
                helper=ModuleType('research.h3_personal.direct_job_api'); helper.submit_job=lambda *a, **kw:None
                endpoint=ModuleType('research.h3_personal.job_av_endpoint')
                def attest(*args, **kwargs):
                    bound = inspect.signature(attest_job).bind(*args, **kwargs)
                    self.assertEqual(bound.arguments, dict(api=api, job_id='job', port=7860,
                        expected_image='image', expected_command=['command']))
                    events.append('attest')
                    return SimpleNamespace(base_url='https://attested.invalid')
                endpoint.attest_job=create_autospec(attest_job, side_effect=attest)
                token=ModuleType('research.provision_modal_identity_secret'); token.cached_token=lambda:'token'
                client=ModuleType('research.h3_personal.live_relay_client')
                class Consumer:
                    def __init__(self,url,**kwargs):
                        self.kw=kwargs
                        self.opened=False
                        self_outer.assertEqual(kwargs['not_after'],r['expires_at'])
                        self_outer.assertEqual(kwargs['role'],'consumer')
                    def open_claim(self): self.opened=True; events.append('open')
                    def close(self):
                        events.append('consumer_close')
                        if scenario == 'cleanup': raise ValueError('close failed')
                self_outer=self; client.SocketAdapter=Consumer
                director=SimpleNamespace(close=lambda:events.append('director_close'))
                asr_mode = scenario.startswith('asr')
                lease = SimpleNamespace(close=lambda: events.append('asr_close'))
                source = SimpleNamespace(expected_asr={'pinned': 'metadata'})
                def prep(directory,row,api):
                    if asr_mode:
                        self.assertEqual(row['expected_asr'], source.expected_asr)
                    else:
                        self.assertNotIn('expected_asr', row)
                    self.assertEqual(row['lease_seconds'],1200)
                    self.assertEqual(row['port'],7860)
                    r['session']=row['session']
                    return directory/'deployment'
                def wait(*a):
                    events.append('ready')
                    if scenario in ('ready', 'asr_failure'): raise failure
                    if scenario == 'sigterm':
                        handler = signal.getsignal(signal.SIGTERM)
                        try:
                            handler(signal.SIGTERM, None)
                        finally:
                            # Repeated termination does not interrupt cleanup.
                            handler(signal.SIGTERM, None)
                    return r
                def conversation(consumer,binding,root,turns,pi,**kw):
                    self.assertTrue(consumer.opened)
                    self.assertEqual(binding.session,r['session'])
                    self.assertEqual(turns, None if asr_mode else ['fresh'])
                    if asr_mode:
                        self.assertIs(kw['input_source'], source)
                    else:
                        self.assertNotIn('input_source', kw)
                    self.assertIs(pi,director)
                    self.assertFalse(kw['allow_obs'])
                    events.append('conversation')
                    if scenario == 'cleanup': raise failure
                    return 'done'
                modules={m.__name__:m for m in (hf,http,helper,endpoint,token,client)}
                from research.h3_personal import resident_endpoint, direct_asr_input
                with patch.object(resident_endpoint, 'PRIVATE', Path(d)), patch.object(resident_endpoint.ResidentEndpoint, 'open', return_value=lease) as open_asr, patch.object(direct_asr_input, 'DirectASRInput', return_value=source), patch.dict(sys.modules,modules), patch.object(launch,'canonical'), patch.object(launch.shutil,'which',return_value='/existing'), patch.object(controller,'PiDirector',return_value=director), patch.object(controller,'run_conversation',side_effect=conversation), patch.object(launch,'prepare',side_effect=prep), patch.object(launch,'submit',return_value=(job,'image',['command'])), patch.object(launch,'await_ready',side_effect=wait):
                    previous = signal.getsignal(signal.SIGTERM)
                    args = (Path(d)/'fresh', None if asr_mode else ['fresh'])
                    kwargs = dict(capture_comet=True, asr_parent=Path(d)/'parent') if asr_mode else {}
                    if scenario in ('ready', 'cleanup', 'sigterm', 'inspect', 'asr_failure'):
                        expected = {'ready': TimeoutError, 'cleanup': TimeoutError,
                                    'sigterm': InterruptedError, 'inspect': OSError, 'asr_failure': TimeoutError}[scenario]
                        with self.assertRaises(expected) as raised:
                            launch.run(*args, **kwargs)
                        if scenario in ('ready', 'cleanup'):
                            self.assertIs(raised.exception, failure)
                        if scenario == 'cleanup':
                            self.assertEqual(len(raised.exception.__notes__), 3)
                    else:
                        self.assertEqual(launch.run(*args, **kwargs),'done')
                    if asr_mode:
                        open_asr.assert_called_once_with(Path(d).resolve()/'fresh/asr', Path(d)/'parent', accounting='operator-managed')
                        self.assertEqual(events.count('asr_close'), 1)
                        events.remove('asr_close')
                    else:
                        open_asr.assert_not_called()
                    self.assertIs(signal.getsignal(signal.SIGTERM), previous)
                    if scenario in ('ready', 'sigterm', 'asr_failure'):
                        self.assertEqual(events,['ready','director_close','cancel'])
                    else:
                        self.assertEqual(events,['ready','attest','open','conversation','consumer_close','director_close','cancel'])

    def test_submit_interruption_after_callback_cancels_owned_job(self):
        self._submit_case('after_callback')

    def test_submit_interrupted_receipt_cancels_owned_job(self):
        self._submit_case('receipt')

    def test_submit_exact_payload_guard_before_single_post(self):
        self._submit_case()

    def test_backup_receipt_failure_confirms_terminal_in_compute_namespace(self):
        self._submit_case('receipt', backup=True)

    def _submit_case(self, interruption=None, *, backup=False):
        # Fake control plane only. Neither httpx nor HF makes a network call.
        with tempfile.TemporaryDirectory() as d:
            stage=Path(d)/'deployment'; stage.mkdir()
            names=['research/bootstrap_h3_fast_ref2va.py','upload-allowlist.json']
            for n in names:
                p=stage/n; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('code only')
            entries=[]
            for n in names:
                body=(stage/n).read_bytes()
                entries.append(SimpleNamespace(rfilename=n,size=len(body),lfs=None,
                    blob_id=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()))
            s=spec(); s.update(publication_repo='YUGOROU/private-direct',image='pinned-image')
            api=SimpleNamespace(whoami=lambda:dict(name='YUGOROU',isPro=True),
                list_jobs=lambda **kw:[],
                repo_info=lambda *a,**kw:SimpleNamespace(private=True,sha='revision',siblings=entries),
                create_commit=lambda *a,**kw:SimpleNamespace(oid='revision'))
            calls=[]
            cancelled=[]
            api.cancel_job=lambda **kw:cancelled.append(kw['job_id'])
            inspected=[]
            def inspect_terminal(**kw):
                inspected.append(kw)
                return SimpleNamespace(status=SimpleNamespace(stage='CANCELED'))
            api.inspect_job=inspect_terminal
            compute_api=SimpleNamespace(cancel_job=lambda **kw:cancelled.append(kw['job_id']),
                                        inspect_job=inspect_terminal)
            compute=SimpleNamespace(api=compute_api, token='backup-token', namespace='2nd2nd',
                                    assert_idle=lambda:None) if backup else None
            if backup:
                s['live_direct_session']=dict(stable_selfie=True,tts_mode='native_voxcpm',
                    native_tts=dict(schema='h3-native-voxcpm-backend-v1'))
            def submit(client,token,payload,on_created=None,**kwargs):
                self.assertTrue((Path(d)/'submission_guard.json').exists())
                self.assertFalse((Path(d)/'submission.json').exists())
                calls.append(payload)
                job=SimpleNamespace(id='job-id')
                self.assertEqual(token,'backup-token' if backup else 'token')
                self.assertEqual(kwargs,{'namespace':'2nd2nd'} if backup else {})
                on_created(job)
                self.assertTrue((Path(d)/'submission.json').exists())
                if interruption == 'after_callback':
                    raise KeyboardInterrupt('after callback')
                return job
            helper=ModuleType('research.h3_personal.direct_job_api'); helper.submit_job=submit
            hf=ModuleType('huggingface_hub'); hf.CommitOperationAdd=lambda **kw:kw
            class Control:
                def __init__(self,**kw):
                    self.kw=kw
                def __enter__(self): return self
                def __exit__(self,*a): pass
                def post(self,url,**kw):
                    self_url='https://huggingface.co/api/repos/create'
                    if url != self_url: raise AssertionError(url)
                    return SimpleNamespace(status_code=201)
            http=ModuleType('httpx'); http.Client=Control
            with patch.dict(sys.modules,{'huggingface_hub':hf,'httpx':http,helper.__name__:helper}), patch.object(launch,'inspect',return_value=(s,{'files':[{'path':names[0]}]})), patch('research.launch_h3_live_voice.pre_submit') as gates, patch.object(launch,'_backup_compute_binding',return_value=compute), patch('research.h3_personal.native_tts_backend.native_job_secrets',return_value={}), contextlib.redirect_stdout(io.StringIO()):
                if interruption:
                    original_durable=launch.durable_json
                    def durable(path, value):
                        if interruption == 'receipt' and Path(path).name == 'submission.json':
                            raise KeyboardInterrupt('receipt interrupted')
                        return original_durable(path,value)
                    with patch.object(launch,'durable_json',side_effect=durable):
                        with self.assertRaises(KeyboardInterrupt):
                            launch.submit(api,stage,'token','producer','consumer',compute_profile=compute)
                    self.assertEqual(inspected,[dict(job_id='job-id',**({'namespace':'2nd2nd'} if backup else {}))])
                    terminal=json.loads((Path(d)/'job-terminal.json').read_text())
                    self.assertTrue(terminal['independent_confirmation'])
                    self.assertEqual(terminal['job_namespace'],'2nd2nd' if backup else 'YUGOROU')
                    self.assertEqual(cancelled,['job-id'])
                    self.assertEqual(len(calls),1)
                    if interruption == 'after_callback':
                        self.assertEqual(json.loads((Path(d)/'submission.json').read_text())['job_id'],'job-id')
                    return
                job,image,command=launch.submit(api,stage,'token','producer','consumer')
                self.assertEqual(gates.call_count,2)
                with self.assertRaises(ValueError):
                    launch.submit(api,stage,'token','producer','consumer')
            self.assertEqual(job.id,'job-id')
            self.assertEqual(len(calls),1)
            p=calls[0]
            self.assertEqual(set(p),{'dockerImage','command','arguments','environment','secrets','flavor','timeoutSeconds','attempts','expose','labels'})
            self.assertEqual(p['expose'],{'ports':[7860],'portsPublic':[]})
            self.assertEqual((p['timeoutSeconds'],p['attempts'],p['flavor']),(1200,1,'h200x2'))
            self.assertEqual(p['secrets'],{'HF_TOKEN':'token','H3_DIRECT_PRODUCER_KEY':'producer','H3_DIRECT_CONSUMER_KEY':'consumer'})
            self.assertEqual(p['command'],command)
            self.assertEqual(p['arguments'],[])
            self.assertEqual(p['labels'],{'deployrev':'revision'})
            receipt=(Path(d)/'submission.json').read_text()
            self.assertEqual(json.loads(receipt)['job_id'],'job-id')
            self.assertNotIn('consumer',receipt)
            self.assertNotIn('producer',receipt)


if __name__ == '__main__':
    unittest.main()
