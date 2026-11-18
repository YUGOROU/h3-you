import asyncio
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as N
import unittest
from research.h3_personal.discord_bridge import DiscordScope,DiscordTurnBridge


class DiscordBridgeTests(unittest.IsolatedAsyncioTestCase):
    def make(self,root,*,gate=None,fail=False):
        threads=[]
        class Producer:
            history=[]
            h3=N(session=None)
            def produce(self,text,target,is_current):
                threads.append(threading.get_ident())
                if gate is not None:gate.wait(2)
                if fail:raise RuntimeError('secret/transcript must never escape')
                if not is_current():raise RuntimeError('stale')
                self.history.append({'content':text});target.mkdir();return target/'manifest.json'
        def factory():threads.append(threading.get_ident());return Producer()
        b=DiscordTurnBridge(DiscordScope(1,2,3),factory,root,'a'*64,mux=lambda m,a,o:o)
        return b,threads

    async def test_scope_and_backpressure_one_thread(self):
        with tempfile.TemporaryDirectory() as d:
            gate=threading.Event();b,threads=self.make(d,gate=gate);sent=[]
            async def deliver(p):sent.append(p);return N(delete=None)
            self.assertEqual(b.submit(1,2,4,'hi',deliver),'forbidden')
            self.assertEqual(b.submit(1,2,3,'hi',deliver,is_bot=True),'forbidden')
            self.assertEqual(b.submit(1,2,3,'hi',deliver),'accepted')
            self.assertEqual(b.submit(1,2,3,'other',deliver),'busy')
            gate.set();self.assertEqual(await b.wait_result(),'delivered')
            self.assertEqual(len(set(threads)),1);self.assertEqual(len(sent),1)
            await b.close()

    async def test_cancel_discards_late_generation(self):
        with tempfile.TemporaryDirectory() as d:
            gate=threading.Event();b,_=self.make(d,gate=gate);sent=[]
            async def deliver(p):sent.append(p)
            b.submit(1,2,3,'hi',deliver)
            self.assertEqual(await b.stop(1,2,4),'forbidden')
            await b.stop(1,2,3);gate.set()
            self.assertEqual(await b.wait_result(),'interrupted');self.assertFalse(sent)
            await b.close()

    async def test_delivery_failure_rolls_back_history_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            b,threads=self.make(d)
            async def deliver(p):raise RuntimeError('remote secret')
            b.submit(1,2,3,'hi',deliver)
            self.assertEqual(await b.wait_result(),'failed')
            self.assertEqual(b._producer.history,[]);self.assertEqual(len(threads),2)
            await b.close()

    async def test_cancel_during_upload_deletes_own_post(self):
        with tempfile.TemporaryDirectory() as d:
            b,_=self.make(d);entered=asyncio.Event();finish=asyncio.Event();deleted=[]
            async def delete():deleted.append(True)
            async def deliver(p):entered.set();await finish.wait();return N(delete=delete)
            b.submit(1,2,3,'hi',deliver);await asyncio.wait_for(entered.wait(),3)
            await b.stop(1,2,3);finish.set()
            self.assertEqual(await b.wait_result(),'interrupted');self.assertEqual(deleted,[True])
            self.assertEqual(b._producer.history,[]);await b.close()

    async def test_invalid_text_and_closed_bridge(self):
        with tempfile.TemporaryDirectory() as d:
            b,_=self.make(d)
            self.assertEqual(b.submit(1,2,3,'',None),'invalid')
            await b.close()
            with self.assertRaises(RuntimeError):b.submit(1,2,3,'hi',None)

    def test_invalid_scope(self):
        with self.assertRaises(ValueError):DiscordScope(1,2,0)
