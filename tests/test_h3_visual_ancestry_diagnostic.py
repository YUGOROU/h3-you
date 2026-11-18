"""CPU-only bounded ancestry contract/security tests, no model imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import wave
from research.h3_personal.visual_ancestry_diagnostic import speech, digest, inventory, rank_evidence, SECTION


class Contracts(unittest.TestCase):
    def test_exact_pcm_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            d=str(Path(temporary).resolve())
            p=Path(d)/'diagnostic-speech.wav';pcm=b'\x01\x00'*480
            with wave.open(str(p),'wb') as w:
                w.setparams((1,2,48000,0,'NONE','not compressed'));w.writeframes(pcm)
            c=dict(filename=p.name,wav_sha256=digest(p.read_bytes()),pcm_sha256=digest(pcm),samples=480,sample_rate=48000,channels=1)
            self.assertEqual(speech(d,c),pcm)
            for change in ({'samples':True},{'samples':1440001},{'channels':2},{'sample_rate':24000},{'pcm_sha256':'0'*64},{'filename':'../input.wav'},{'extra':1}):
                with self.subTest(change=change), self.assertRaises(ValueError): speech(d,dict(c,**change))
            p.unlink();p.symlink_to('/dev/null')
            with self.assertRaises(ValueError): speech(d,c)

    def test_inventory_never_uploads_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            d=str(Path(temporary).resolve())
            root=Path(d);(root/'arm-0').mkdir();p=root/'arm-0/evidence.json';p.write_text('{}')
            result=inventory(root,['arm-0/evidence.json'])
            self.assertEqual(result[0]['sha256'],digest(b'{}'))
            for names in (['../secret'],['arm-0/driver.wav'],['arm-0/evidence.json']*2,['arm-0/temporary.pcm']):
                with self.assertRaises(ValueError): inventory(root,names)
            p.unlink();p.symlink_to('/etc/hosts')
            with self.assertRaises(ValueError): inventory(root,['arm-0/evidence.json'])

    def test_both_rank_fail_closed_and_clocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            d=str(Path(temporary).resolve())
            c=dict(sequence=1,role='listening',logical_index=1,logical_source_id='idle-epoch-0',conversation_final=False,source_final=False)
            row=dict(streaming=c,committed_frame=226,canonical_sha256='a'*64,
                     guide_digests=['b'*64,'c'*64,'a'*64],visual_ancestry_evidence=dict(policy=SECTION['arms'][0],sequence=1,kind='latest',
                         anchor_sha256='b'*64,selected_sha256='b'*64,anchor_sequence=1))
            paths=[Path(d)/f'streaming-rank-{r}-window-1.json' for r in (0,1)]
            for p in paths:p.write_text(json.dumps(row))
            manifest={'session_id':'test-session','streaming_audio':c,'visual_ancestry_diagnostic':{'policy':SECTION['arms'][0]}}
            decode_paths=[Path(d)/f'h3-continuation-decode-rank-{r}-window-0.json' for r in (0,1)]
            decoded=dict(session_id=manifest['session_id'],worker_rank=1,committed_frames=226,emitted_frames=221,final=False)
            for rank,p in enumerate(decode_paths):p.write_text(json.dumps(dict(decoded,worker_rank=rank)))
            self.assertEqual(rank_evidence(d,manifest)['emitted_frame'],221)
            for key,value in [('session_id','stale'),('worker_rank',0),('worker_rank',True),
                              ('committed_frames',225),('committed_frames',True),
                              ('emitted_frames',226),('emitted_frames',True),('final',True),('final',0)]:
                decode_paths[1].write_text(json.dumps(dict(decoded,**{key:value})))
                with self.subTest(decode_key=key,value=value),self.assertRaises(ValueError):rank_evidence(d,manifest)
            decode_paths[1].write_text(json.dumps(decoded))
            for key in ('source_final','conversation_final'):
                bad_manifest=copy.deepcopy(manifest);bad_manifest['streaming_audio'][key]=0
                with self.subTest(key=key),self.assertRaises(ValueError):rank_evidence(d,bad_manifest)
            c['conversation_final']=True
            for rank,p in enumerate(decode_paths):p.write_text(json.dumps(dict(decoded,worker_rank=rank,emitted_frames=226,final=True)))
            self.assertEqual(rank_evidence(d,manifest)['emitted_frame'],226)
            c['conversation_final']=False
            for rank,p in enumerate(decode_paths):p.write_text(json.dumps(dict(decoded,worker_rank=rank)))
            decode_paths[1].unlink()
            with self.assertRaises(ValueError):rank_evidence(d,manifest)
            decode_paths[1].write_text(json.dumps(decoded))
            for key,value in [('canonical_sha256','d'*64),('committed_frame',227),('visual_ancestry_evidence',{})]:
                bad=copy.deepcopy(row);bad[key]=value;paths[1].write_text(json.dumps(bad))
                with self.subTest(key=key),self.assertRaises(ValueError):rank_evidence(d,manifest)
            paths[1].unlink()
            with self.assertRaises(ValueError):rank_evidence(d,manifest)

    def test_actual_driver_schedule_cpu(self):
        import os
        import time
        from types import SimpleNamespace
        from unittest.mock import patch
        import sys
        class Pixels:
            dtype='uint8'
            def __init__(self,count):self.count=count;self.shape=(count,448,256,3)
            def __len__(self):return self.count
            def __getitem__(self,index):return self
            def tobytes(self):return bytes(self.count*448*256*3)
        np=SimpleNamespace(uint8='uint8',asarray=lambda f:f,stack=lambda fs:Pixels(len(fs)))
        class Sheet:
            def paste(self,*args):pass
            def save(self,out,**kw):out.write(b'png')
        image=SimpleNamespace(new=lambda *args:Sheet(),fromarray=lambda p:p)
        from research.h3_personal import visual_ancestry_diagnostic as diagnostic
        from research.h3_personal.live_conversation import LiveConversation
        from research.h3_personal import live_native_backend
        from research.h3_personal import conversation_conditioning
        pcm = b'\x01\x02'*240000  # multiple speech windows, nonzero exact PCM
        completed = []
        class Backend:
            def __init__(self, runtime, root, *, exporter, evidence_reader):
                self.root=Path(root);self.root.mkdir();self.exporter=exporter
                self.reader=evidence_reader;self._retired=False;self._id='fake'
                self.emitted=0;self.epoch_anchor=None
            def __call__(self, m):
                c=m['streaming_audio'];seq=c['sequence'];policy=m['visual_ancestry_diagnostic']['policy']
                selected=None if seq==0 else digest(str(seq).encode())
                if c['role']=='listening' and c['logical_index']==1:self.epoch_anchor=selected
                if c['role']=='speech' or c['logical_index']==0:self.epoch_anchor=None
                anchor=digest(b'1') if policy==SECTION['arms'][1] and seq else self.epoch_anchor
                refresh=seq>0 and (seq%3==0 if policy==SECTION['arms'][1] else
                    c['role']=='listening' and c['logical_index']>=3 and c['logical_index']%3==0)
                if refresh:selected=anchor
                end=124+seq*102;emitted=end-(0 if c['conversation_final'] else 5)
                row=dict(streaming=c,committed_frame=end,
                    canonical_sha256='a'*64,guide_digests=[selected,None if seq==0 else 'c'*64,'a'*64],
                    visual_ancestry_evidence=dict(policy=policy,sequence=seq,
                        kind='unconditioned' if seq==0 else 'anchor' if refresh else 'latest',
                        anchor_sha256=anchor,selected_sha256=selected,
                        anchor_sequence=((1 if policy==SECTION['arms'][1] else
                                          seq-c['logical_index']+1) if anchor else None)))
                for rank in (0,1):
                    (self.root/f'streaming-rank-{rank}-window-{seq}.json').write_text(json.dumps(row))
                    (self.root/f'h3-continuation-decode-rank-{rank}-window-0.json').write_text(json.dumps(dict(
                        session_id=m['session_id'],worker_rank=rank,committed_frames=end,
                        emitted_frames=emitted,final=c['conversation_final'])))
                self.reader(self.root,m)
                frames=[None]*(emitted-self.emitted)
                video=self.exporter(SimpleNamespace(frames=frames),self.root/f'segment-{seq:03d}.mp4',self.emitted,emitted,seq)
                count=emitted-self.emitted;self.emitted=emitted
                return dict(video=video,count=count,committed_frame=end,emitted_frame=emitted,
                    conditioning_prefix_sha256=None if c['audio_offer'] is None else c['audio_offer']['prefix_sha256'],receipt_sha256='b'*64)
            def retire(self):self._retired=True
            def abandon(self,**kw):self._retired=True
        # Codec process is simulated; real driver/ingress/manifest/PCM composition run.
        audio=[None]
        def process(args,**kw):
            if args[0]=='ffprobe':
                frames=len(audio[0])//4000
                return SimpleNamespace(stdout=json.dumps({'frames':[
                    {'best_effort_timestamp_time':str(i/24)} for i in range(frames)]}).encode())
            target=args[-1]
            if target=='pipe:1':return SimpleNamespace(stdout=audio[0])
            Path(target).write_bytes(b'generated')
            if target.endswith('review.mov'):
                with wave.open(str(Path(target).parent/'temporary.wav'),'rb') as w:audio[0]=w.readframes(w.getnframes())
            return SimpleNamespace(stdout=b'')
        def export(result,path,*args):path.write_bytes(b'video');return b'video'
        def read(*args):return {}
        def driver(*args,**kw):
            d=LiveConversation(*args,**kw);completed.append(d);return d
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{'numpy':np,'PIL':SimpleNamespace(Image=image)}), \
                patch.dict(os.environ,{diagnostic.ENV:diagnostic.SCHEMA}), \
                patch.object(live_native_backend,'LiveNativeBackend',Backend), \
                patch.object(live_native_backend,'_export',export), \
                patch.object(conversation_conditioning,'role_profile_digest',return_value='a'*64), \
                patch('research.hf_h3_streaming_trial.read_evidence',read), \
                patch('research.h3_personal.live_conversation.LiveConversation',driver), \
                patch.object(diagnostic.subprocess,'run',side_effect=process):
            result=diagnostic.run_diagnostic(SimpleNamespace(quality_profile=None),Path(tmp)/'output',pcm=pcm,deadline=time.monotonic()+60)
            self.assertTrue(result['completed'])
            self.assertEqual(len(completed),2)
            for arm in result['arms']:
                self.assertGreater(arm['windows'],21)
                self.assertEqual(arm['samples'],arm['receipts'][-1]['emitted_frame']*2000)
            self.assertEqual(sum('/native-' in f['path'] for f in result['files']),6)

    def test_partial_publication_has_no_input_paths(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from research.hf_h3_visual_ancestry_diagnostic import publish
        class Op:
            def __init__(self,**kw):self.__dict__.update(kw)
        class API:
            def repo_info(self,*a,**kw):return SimpleNamespace(private=True)
            def create_commit(self,*a,**kw):self.operations=kw['operations']
        api=API()
        with patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(CommitOperationAdd=Op)}):
            publish(api,'private','/nonexistent',{'secret':'input'},partial=True)
        self.assertEqual(len(api.operations),1)
        self.assertNotIn(b'secret',api.operations[0].path_or_fileobj)


if __name__=='__main__':unittest.main()
