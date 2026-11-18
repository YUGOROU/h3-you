"""Private generated-only transfer with fake Hub, no credentials/network/media."""
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from research.h3_personal import generated_transfer as transfer
from research.h3_personal.buffered_stream import BufferedStream


def digest(b):return hashlib.sha256(b).hexdigest()


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.prefix='generated-playback/turn1/'
        self.manifest={'schema':'h3-generated-playback-transfer-v1','ai_generated':True,
            'audio_source':'voxcpm2_zero_shot_generated','video_source':'h3_ref2va_identity',
            'h3_revision':transfer.H3_REVISION,'adapter_sha256':'a'*64,
            'assets':{'video':{'file':'video.mp4','bytes':1,'sha256':digest(b'v')},
                      'original_tts':{'file':'tts.wav','bytes':1,'sha256':digest(b'a')}}}
        self.calls=[];self.private=True;self.current=True
        self.params={'api':self,'download':self.download,'repo':'YUGOROU/generated-test',
            'revision':'b'*40,'bundle':'turn1','adapter_sha256':'a'*64,
            'destination':self.root/'turn1','validator':lambda root,c:Fraction(1,4)}
        self.reset()

    def reset(self):
        b=json.dumps(self.manifest).encode()
        self.files={self.prefix+'manifest.json':b,self.prefix+'video.mp4':b'v',self.prefix+'tts.wav':b'a'}
        self.params['manifest_sha256']=digest(b)

    def repo_info(self,*args,**kwargs):
        return N(private=self.private,sha='b'*40,siblings=[N(rfilename=n,size=len(b),
            lfs=None if n.endswith('json') else N(sha256=digest(b))) for n,b in self.files.items()])

    def download(self,repo,name,revision,target):
        self.calls.append(name);target.write_bytes(self.files[name])

    def test_verified_atomic_private_publication_no_overwrite(self):
        path=transfer.fetch(**self.params)
        self.assertEqual(path,self.root/'turn1/clips.json')
        self.assertEqual(len(self.calls),3)
        self.assertEqual(json.loads(path.read_text())['clips'][0]['original_tts'],'tts.wav')
        self.assertEqual(path.parent.stat().st_mode & 0o077,0)
        receipt=json.loads((path.parent/'RECEIPT.json').read_text())
        self.assertFalse(receipt['producer_claims_independently_verified']);self.assertFalse(receipt['autoplay'])
        with self.assertRaises(ValueError):transfer.fetch(**self.params)

    def test_raw_audio_or_wrong_identity_refused_before_media(self):
        for field,bad in (('audio_source','frozen clean recorded target'),('video_source','raw_camera'),
                          ('adapter_sha256','c'*64),('h3_revision','c'*40),('ai_generated',False)):
            with self.subTest(field=field):
                original=copy.deepcopy(self.manifest);self.manifest[field]=bad;self.reset();self.calls=[]
                with self.assertRaises(ValueError):transfer.fetch(**self.params)
                self.assertEqual(self.calls,[self.prefix+'manifest.json'])
                self.assertFalse(self.params['destination'].exists())
                self.manifest=original

    def test_paths_sizes_and_metadata_fail_closed(self):
        for key,value in (('file','../../raw.wav'),('bytes',81*1024*1024),('sha256','c'*64)):
            original=copy.deepcopy(self.manifest);self.manifest['assets']['original_tts'][key]=value
            self.reset();self.calls=[]
            with self.assertRaises(ValueError):transfer.fetch(**self.params)
            self.assertEqual(len(self.calls),1);self.manifest=original

    def test_public_or_unpinned_refused_without_download(self):
        self.private=False
        with self.assertRaises(ValueError):transfer.fetch(**self.params)
        self.assertFalse(self.calls)
        self.private=True
        for key,value in (('revision','main'),('repo','Other/raw'),('bundle','../raw')):
            p=dict(self.params);p[key]=value
            with self.assertRaises(ValueError):transfer.fetch(**p)
        self.assertFalse(self.calls)

    def test_corrupt_download_and_mid_transfer_cancel_cleanup(self):
        original=self.download
        def corrupt(repo,name,rev,target):
            original(repo,name,rev,target)
            if name.endswith('video.mp4'):target.write_bytes(b'x')
        with self.assertRaises(ValueError):transfer.fetch(**dict(self.params,download=corrupt))
        self.assertEqual(list(self.root.iterdir()),[])
        def cancel(repo,name,rev,target):
            original(repo,name,rev,target);self.current=False
        with self.assertRaises(transfer.TransferInterrupted):
            transfer.fetch(**dict(self.params,download=cancel,is_current=lambda:self.current))
        self.assertEqual(list(self.root.iterdir()),[])

    def test_queue_epoch_binding_no_autoplay(self):
        q=BufferedStream(self.root,validator=lambda root,c:Fraction(1,4));t=q.begin()
        params=dict(self.params);params.pop('destination')
        self.assertTrue(transfer.fetch_into_queue(q,t,**params))
        item=q.take();self.assertEqual(item.clip.video,'turn1/video.mp4')
        q.interrupt();calls=len(self.calls)
        self.assertFalse(transfer.fetch_into_queue(q,t,**params));self.assertEqual(len(self.calls),calls)


if __name__=='__main__':unittest.main()
