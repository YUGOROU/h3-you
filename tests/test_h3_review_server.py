import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from research.h3_personal.review_server import ReviewStore, make_server, PAGE


class ReviewServerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.candidate={'id':'s_0000','source_sha256':'a'*64,'source':'/original.mov',
            'session_id':'s','start_seconds':0.,'end_seconds':5.166667,'pool':'review','reasons':[]}
        (self.root/'candidate_manifest.json').write_text(json.dumps([self.candidate]))
        self.store=ReviewStore(self.root)
        self.server=make_server(self.store,0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()
        self.addCleanup(self.stop)
        self.url=f'http://127.0.0.1:{self.server.server_port}'

    def stop(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def post(self,payload,token=None,origin=None):
        headers={'Content-Type':'application/json','X-Review-Token':token or self.store.token}
        if origin:headers['Origin']=origin
        return urlopen(Request(self.url+'/label',data=json.dumps(payload).encode(),headers=headers),timeout=3)

    def test_page_data_and_persistent_labels(self):
        with urlopen(self.url) as r:
            text=r.read().decode();self.assertIn('ArrowUp',text);self.assertNotIn('__TOKEN__',text)
        with self.post({'id':'s_0000','label':'identity'}) as r:
            self.assertEqual(json.load(r)['labels']['s_0000']['label'],'identity')
        restored=ReviewStore(self.root)
        self.assertEqual(restored.labels['s_0000']['review_scope'],'sampled_color_frames')
        self.assertFalse((self.root/'selected_manifest.json').exists())
        with self.post({'id':'s_0000','label':None}) as r:
            self.assertEqual(json.load(r)['labels'],{})
        with urlopen(self.url+'/data') as r:
            self.assertEqual(json.load(r)['labels'],{})

    def test_unknown_label_host_and_csrf_rejected(self):
        for payload,token,origin,status in [({'id':'../secret','label':'reject'},None,None,400),
            ({'id':'s_0000','label':'bogus'},None,None,400),
            ({'id':'s_0000','label':'reject'},'wrong',None,403),
            ({'id':'s_0000','label':'reject'},None,'https://external.example',403)]:
            with self.assertRaises(HTTPError) as e:self.post(payload,token,origin)
            self.assertEqual(e.exception.code,status)
        with self.assertRaises(HTTPError) as e:
            urlopen(Request(self.url,headers={'Host':'attacker.example'}))
        self.assertEqual(e.exception.code,403)
        self.assertEqual(self.store.labels,{})

    def test_file_allowlist(self):
        for path in ('/CURRENT.md','/frame/../secret','/reference/-1','/reference/100'):
            with self.assertRaises(HTTPError) as e:urlopen(self.url+path)
            self.assertEqual(e.exception.code,404)
        file=self.root/'preview.jpg';file.write_bytes(b'test-image')
        with patch.object(self.store,'frame',return_value=file):
            with urlopen(self.url+'/frame/s_0000') as r:
                self.assertEqual(r.read(),b'test-image')

    def test_proposals_reload_without_changing_human_judgments(self):
        self.store.label('s_0000','identity')
        (self.root/'quality_proposals.json').write_text(json.dumps({'schema_version':1,'cpu_only':True,
            'proposals':{'s_0000':{'source_sha256':'a'*64,'suggested_label':None,
                'decision':'review','needs_review':True,'vote_score':.5,'reasons':['uncertain']}}}))
        with urlopen(self.url+'/data') as r:body=json.load(r)
        self.assertEqual(body['labels']['s_0000']['label'],'identity')
        self.assertTrue(body['proposals']['s_0000']['needs_review'])

    def test_initial_group_membership_does_not_shrink_after_approval(self):
        extra={**self.candidate,'id':'s_0001','pool':'identity'}
        (self.root/'candidate_manifest.json').write_text(json.dumps([self.candidate,extra]))
        store=ReviewStore(self.root)
        self.assertEqual(store.initial_ids,['s_0001'])
        store.label('s_0001','reject')
        restored=ReviewStore(self.root)
        self.assertEqual(restored.initial_ids,['s_0001'])
        self.assertEqual(restored.labels['s_0001']['label'],'reject')

    def test_query_view_pages_and_navigation_are_read_only(self):
        for query in ('?view=teacher','?view=approval'):
            with urlopen(self.url+'/'+query) as r:
                self.assertEqual(r.status,200)
        self.assertEqual(self.store.labels,{})
        self.assertIn("if(e.key==='ArrowRight')navigate(1)",PAGE)
        self.assertNotIn("if(e.key==='ArrowRight')annotate",PAGE)
        self.assertIn("if(e.key==='Enter')approve()",PAGE)

    def test_changed_source_blocks_old_labels(self):
        self.store.label('s_0000','motion')
        (self.root/'candidate_manifest.json').write_text(json.dumps([{**self.candidate,'source_sha256':'b'*64}]))
        with self.assertRaises(ValueError):ReviewStore(self.root)

    def test_save_failure_does_not_change_memory(self):
        with patch('research.h3_personal.review_server.dump',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.store.label('s_0000','reject')
        self.assertEqual(self.store.labels,{})

    @unittest.skipUnless(shutil.which('node'), 'Node needed for UI logic simulation')
    def test_keyboard_navigation_and_approval_dom_simulation(self):
        root=Path(__file__).resolve().parent.parent
        subprocess.run(['node',str(root/'tests/review_ui_test.cjs'),
                        str(root/'research/h3_personal/review_page.html')],check=True,
                       timeout=10,capture_output=True,text=True)

    def test_keys_wait_for_image_and_no_auto_training_approval(self):
        self.assertIn('if(!imageReady)',PAGE)
        self.assertIn('e.repeat',PAGE)
        self.store.label('s_0000','identity')
        payload=json.loads((self.root/'image_labels.json').read_text())
        self.assertFalse(payload['full_clip_review_complete'])

if __name__=='__main__':unittest.main()
