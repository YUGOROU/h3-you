"""Paired owner failure contracts; cloud/conversation mocked, TLS cert CPU only."""
import hashlib
import json
from pathlib import Path
import ssl
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from research import run_h3_network_pair as owner
from research.h3_personal.voxcpm_endpoint_protocol import PINS

class PairOwnerTests(unittest.TestCase):
    def exercise(self,mode='normal',*,h3_first=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)/'run';root.mkdir();fence=Path(temp.name)/'fence'
        api=Mock();client=Mock();canceled=set()
        def inspect(job_id,**kw):return SimpleNamespace(status=SimpleNamespace(stage='CANCELED' if job_id in canceled else 'RUNNING'))
        api.inspect_job.side_effect=inspect
        api.cancel_job.side_effect=lambda job_id,**kw:canceled.add(job_id)
        client.post.return_value=SimpleNamespace(status_code=201,json=lambda:{'id':'a'*24})
        if mode=='unknown':client.post.side_effect=TimeoutError('create response unknown')
        if mode=='rejected':client.post.return_value.status_code=403
        receipts=[dict(epoch=i,pins=dict(PINS),native_sha256=str(i)*64,native_samples=3) for i in (1,2)]
        runtime=dict(device_name='NVIDIA L4',device_count=1,optimize=False,sample_rate=48000,torch='2.8.0+cu128')
        def metadata(api,id,deadline,kind):
            if kind=='candidate.completion':
                return [dict(epoch=i,native_float_sha256=str(i)*64,samples=3,completion=dict(reason='natural_stop',natural_stop=True,
                    stream_exhausted=True,stop_index=2,effective_max_len=10,decision_monotonic_ns=1)) for i in (1,2)]
            p=client.post.call_args.kwargs['json'];s=p['secrets']
            exp=json.loads((root/'source-deployment.json').read_text())['expires_at']
            config=dict(schema='h3-native-voxcpm-network-v2',connection=dict(group=p['network']['group'],alias='voxcpm',port=17861,
                certificate_sha256=hashlib.sha256(ssl.PEM_cert_to_DER_cert(s['H3_NATIVE_TLS_CERT'])).hexdigest(),server_job_id='a'*24),
                session=s['H3_TTS_SESSION'],boot=s['H3_TTS_BOOT'],expires_at=exp,pins=dict(PINS),
                preroll=dict(sample_rate=48000,reserve_samples=48000,max_reserve_samples=96000,max_wait_seconds=2.0))
            raw=dict(id='a'*24,owner={'name':'YUGOROU'},flavor='l4x1',dockerImage=p['dockerImage'],network=p['network'],
                labels=p['labels'],command=p['command'],retry=0,status={'stage':'RUNNING'})
            client.get.return_value=SimpleNamespace(status_code=200,json=lambda:raw)
            return [dict(config=config,runtime=runtime)]
        def conversation(directory,config,**kwargs):
            if h3_first:self.assertIs(kwargs['adoption'],deferred)
            h3=root/'h3';h3.mkdir()
            if mode=='unknown-h3':
                (h3/'submission_guard.json').write_text('{}');raise RuntimeError('unknown H3 creation')
            (h3/'submission.json').write_text(json.dumps({'job_id':'b'*24}) if mode!='bad-h3-receipt' else 'malformed')
            if mode=='conversation-failed':raise RuntimeError('conversation failed')
            (h3/'native-worker-evidence.json').write_text(json.dumps({'receipts':receipts}))
            return dict(source_complete=True,turns=2,media_complete=False,closing_tail_unverified=True,remote_original_pcm_verified=False)
        from research.h3_personal.pair_placement import gate
        placement=Mock(side_effect=gate(root))
        if mode=='bad-placement':placement.side_effect=[True,False]
        if mode=='bad-preplacement':placement.side_effect=None;placement.return_value=False
        deferred=Mock();deferred.job=None;deferred.remaining.return_value=1100
        def start():deferred.job=SimpleNamespace(id='b'*24)
        deferred.start.side_effect=start
        if h3_first:
            previous_post=client.post.side_effect
            def post(*args,**kwargs):
                self.assertIsNotNone(deferred.job)
                if previous_post is not None:raise previous_post
                return client.post.return_value
            client.post.side_effect=post
        with patch('research.h3_personal.h3_first_owner.DeferredH3Owner',return_value=deferred),\
             patch.object(owner,'FENCE',fence),patch.object(owner,'bundle',return_value=root/'archive'),\
             patch.object(owner,'publish',return_value=('YUGOROU/fresh','a'*40,'b'*64)),patch.object(owner,'metadata',side_effect=metadata):
            if mode=='normal':result=owner.owned(api,client,'token',root,{},conversation,placement,h3_first=h3_first)
            else:
                with self.assertRaises(Exception):owner.owned(api,client,'token',root,{},conversation,placement,h3_first=h3_first)
                result=json.loads((root/'result.json').read_text())
        return root,fence,canceled,result,placement

    def test_h3_first_order_and_no_duplicate_cleanup(self):
        root,fence,canceled,result,_=self.exercise(h3_first=True)
        self.assertEqual(canceled,{'a'*24,'b'*24})
        self.assertFalse(fence.exists());self.assertTrue(result['cleanup_confirmed'])

    def test_h3_first_source_rejection_releases_h200(self):
        _,fence,canceled,result,_=self.exercise('rejected',h3_first=True)
        self.assertEqual(canceled,{'b'*24});self.assertFalse(fence.exists())
        self.assertTrue(result['cleanup_confirmed'])

    def test_h3_first_unknown_source_create_retains_fence_but_releases_h200(self):
        _,fence,canceled,result,_=self.exercise('unknown',h3_first=True)
        self.assertEqual(canceled,{'b'*24});self.assertTrue(fence.exists())
        self.assertFalse(result['cleanup_confirmed'])

    def test_source_and_h3_end_confirmed_but_partial_tail_not_complete(self):
        root,fence,canceled,result,placement=self.exercise()
        self.assertEqual(canceled,{'a'*24,'b'*24});self.assertFalse(fence.exists())
        self.assertTrue(result['cleanup_confirmed']);self.assertFalse(result['complete']);self.assertEqual(placement.call_count,2)
        self.assertEqual(json.loads((root/'source-terminal.json').read_text())['status'],'CANCELED')

    def test_real_gate_receives_created_identity_only_after_creation(self):
        root,_,_,_,placement=self.exercise()
        pre=placement.call_args_list[0].args[1]['network_service']
        running=placement.call_args_list[1].args[1]['network_service']
        self.assertNotIn('server_job_id',pre)
        self.assertEqual(running['server_job_id'],'a'*24)
        self.assertEqual(json.loads((root/'source-placement.json').read_text())[-1]['phase'],'running_source')

    def test_conversation_failure_cannot_skip_dual_cleanup(self):
        _,fence,canceled,_,_=self.exercise('conversation-failed')
        self.assertEqual(canceled,{'a'*24,'b'*24});self.assertFalse(fence.exists())

    def test_unknown_source_create_keeps_fence_no_replay(self):
        _,fence,canceled,result,_=self.exercise('unknown')
        self.assertTrue(fence.exists());self.assertEqual(canceled,set());self.assertFalse(result['cleanup_confirmed'])

    def test_definite_rejection_clears_only_own_fence(self):
        _,fence,canceled,_,_=self.exercise('rejected')
        self.assertFalse(fence.exists());self.assertEqual(canceled,set())

    def test_unknown_or_malformed_h3_never_skips_source_cancel(self):
        for mode in ('unknown-h3','bad-h3-receipt'):
            with self.subTest(mode=mode):
                _,fence,canceled,result,_=self.exercise(mode)
                self.assertIn('a'*24,canceled);self.assertTrue(fence.exists());self.assertFalse(result['cleanup_confirmed'])

    def test_bad_preplacement_never_creates_source(self):
        _,fence,canceled,_,_=self.exercise('bad-preplacement')
        self.assertEqual(canceled,set());self.assertFalse(fence.exists())

    def test_bad_placement_never_starts_h3(self):
        _,fence,canceled,_,_=self.exercise('bad-placement')
        self.assertEqual(canceled,{'a'*24});self.assertFalse(fence.exists())

    def test_inspect_error_still_attempts_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            api=Mock();api.inspect_job.side_effect=[OSError(),SimpleNamespace(status=SimpleNamespace(stage='CANCELED'))]
            owner.terminal(api,'a'*24,Path(tmp)/'terminal.json')
            api.cancel_job.assert_called_once_with(job_id='a'*24,namespace='YUGOROU')


class SchedulingWindowTests(unittest.TestCase):
    """A long SCHEDULING wait must not consume the runtime lease."""

    def test_window_adds_queue_budget_to_unchanged_lease(self):
        import time
        before=int(time.time());expires,wait=owner.window(3600)
        self.assertEqual(wait,3600);self.assertEqual(expires-before,3600+owner.LEASE_SECONDS)
        self.assertEqual(owner.LEASE_SECONDS,1200)

    def test_window_rejects_negative_unbounded_or_non_int(self):
        for bad in (-1,owner.SCHEDULING_WAIT_MAX+1,True,1.5,'600',None):
            with self.subTest(wait=bad):
                with self.assertRaises(ValueError):owner.window(bad)

    def test_default_owner_window_is_unchanged(self):
        import inspect
        self.assertEqual(inspect.signature(owner.run).parameters['scheduling_wait'].default,600)
        self.assertEqual(inspect.signature(owner.owned).parameters['scheduling_wait'].default,600)

    def test_long_wait_still_runs_and_reports_clamped_evidence(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        result=PairOwnerTests().exercise()  # default 600 path unchanged
        self.assertTrue(result[3]['schema'])

    def test_worker_clamps_runtime_lease_and_bounds_queue(self):
        import time
        from research.hf_h3_voxcpm_network_worker import lease,LEASE_SECONDS,QUEUE_SECONDS_MAX
        now=time.time()
        later=now+LEASE_SECONDS+30
        self.assertAlmostEqual(lease(later,now)-now,LEASE_SECONDS,places=3)
        self.assertEqual(lease(now+30,now),now+30)
        for bad in (now-1,now,now+QUEUE_SECONDS_MAX+LEASE_SECONDS+1,float('inf')):
            with self.subTest(expires=bad):
                with self.assertRaises(ValueError):lease(bad,now)

if __name__=='__main__':unittest.main()
