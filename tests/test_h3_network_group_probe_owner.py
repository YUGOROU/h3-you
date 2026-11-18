"""Synthetic CPU probe ownership only; no actual Hub calls."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from research import run_h3_network_group_probe as owner

class CPUOwnerTests(unittest.TestCase):
    def exercise(self,mode):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)/'run';root.mkdir();fence=Path(temp.name)/'fence';api=Mock();client=Mock();canceled=set()
        def response(job):return SimpleNamespace(status_code=201,json=lambda:{'id':job})
        client.post.side_effect=[response('a'*24),TimeoutError('unknown')] if mode=='unknown-client' else [response('a'*24),response('b'*24)]
        if mode=='reject':client.post.side_effect=[SimpleNamespace(status_code=403)]
        api.inspect_job.side_effect=lambda job_id,**kw:SimpleNamespace(status=SimpleNamespace(stage='CANCELED' if job_id in canceled else 'RUNNING'))
        api.cancel_job.side_effect=lambda job_id,**kw:canceled.add(job_id)
        proof=dict(type='h3_network_cpu_result',gpu=False,exact_match=True,tls_pinned=True,turns=2)
        with patch.object(owner,'FENCE',fence),patch.object(owner,'publish',return_value=('YUGOROU/fresh','a'*40,'b'*64)),\
             patch.object(owner,'log_event',side_effect=[{'type':'h3_network_cpu_ready'},proof]):
            if mode=='normal':result=owner.owned(api,client,'token',root)
            else:
                with self.assertRaises(Exception):owner.owned(api,client,'token',root)
                result=None
        for call in client.post.call_args_list:
            p=call.kwargs['json'];self.assertEqual(p['flavor'],'cpu-basic');self.assertEqual(p['attempts'],1)
            self.assertEqual(p['timeoutSeconds'],600);self.assertNotIn('expose',p)
        return fence,canceled,result

    def test_exact_two_jobs_end_confirmed(self):
        fence,canceled,result=self.exercise('normal')
        self.assertEqual(canceled,{'a'*24,'b'*24});self.assertFalse(fence.exists());self.assertTrue(result['cleanup_confirmed'])
    def test_unknown_second_job_keeps_fence_but_stops_known_first(self):
        fence,canceled,_=self.exercise('unknown-client')
        self.assertTrue(fence.exists());self.assertEqual(canceled,{'a'*24})
    def test_definite_rejection_has_no_lease(self):
        fence,canceled,_=self.exercise('reject')
        self.assertFalse(fence.exists());self.assertEqual(canceled,set())

if __name__=='__main__':unittest.main()
