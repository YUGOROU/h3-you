import ast
import copy
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from research import launch_h3_irodori_turn as launcher
from research.h3_personal.director_tts_job import validate,inspect,digest,FILES,SCOPE,REFERENCE_SHA,LOCK_SHA,SOURCE,MODEL,CODEC


def request():
    return dict(text='こんにちは。',scope=SCOPE,max_syntheses=1,timeout_seconds=1200,reference_sha256=REFERENCE_SHA,
                source_revision=SOURCE,uv_lock_sha256=LOCK_SHA,uv_version='0.13.1',user_approved_private_reference_upload=True,
                output_repo='YUGOROU/h3-director-irodori-turn-test',license='approved own voice / Gemma',
                director_model='Qwen/Qwen3.8-27B:cerebras',input_origin='user_text_chat_not_discord_asr',
                models=[dict(repo=r,revision=v,files={}) for r,v in [('Aratako/Irodori-TTS-v4-Large',MODEL),('Aratako/Semantic-DACVAE-Japanese-32dim',CODEC),('sony/silentcipher','a1c4d021905e0dc5b24be5f68db5fc4dba410ee1')]])

class TTSJobTests(unittest.TestCase):
    def test_pins_single_turn_and_no_fallback(self):
        self.assertIs(validate(request())['max_syntheses'],1)
        for key,value in [('text','x'*161),('reference_sha256','changed'),('source_revision','main'),('max_syntheses',2),('timeout_seconds',3600),('director_model','other'),('input_origin','discord_voice_received')]:
            r=request();r[key]=value
            with self.assertRaises(ValueError,msg=key):validate(r)

    def test_timeout_and_origin(self):
        r=request();r.update(timeout_seconds=600,input_origin='discord_received_audio_ondevice_asr')
        self.assertEqual(validate(r),r)
        for value in (0,1201,True,False,600.0,'600'):
            bad=copy.deepcopy(r);bad['timeout_seconds']=value
            with self.assertRaises(ValueError):validate(bad)
        r['input_origin']='wrong'
        with self.assertRaises(ValueError):validate(r)

    def test_prepare_600_asr_and_offline_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);original=root/'gpu-hf-irodori-diagnostic-v1/request.json'
            original.parent.mkdir();original.write_text(json.dumps(request()))
            proposal=root/'proposal.json';proposal.write_text(json.dumps(dict(type='turn',model=launcher.MODEL,status='proposal_only',reply='こんにちは。')))
            real_digest=launcher.digest
            def fixture_digest(path):
                return 'b1279cf430f00314f8ac6f61728341fb174ad765fd4ea3bd5528c43c80800002' if path==original else real_digest(path)
            with patch.object(launcher,'PRIVATE',root),patch.object(launcher,'digest',side_effect=fixture_digest):
                stage=launcher.prepare(root/'turn',proposal,timeout_seconds=600,input_origin='discord_received_audio_ondevice_asr')
            prepared,_=inspect(stage)
            self.assertEqual(prepared['timeout_seconds'],600)
            self.assertEqual(prepared['input_origin'],'discord_received_audio_ondevice_asr')
            result=subprocess.run([sys.executable,'-B','-m','research.launch_h3_irodori_turn','--directory',str(stage.parent)],check=True,capture_output=True,text=True,timeout=15)
            self.assertEqual(json.loads(result.stdout)['timeout_seconds'],600)

    def test_prepare_rejects_options_before_reading_files(self):
        for timeout,origin in ((0,'user_text_chat_not_discord_asr'),(1201,'user_text_chat_not_discord_asr'),(True,'user_text_chat_not_discord_asr'),(600,'wrong')):
            with self.assertRaises(ValueError):launcher.prepare(None,None,timeout_seconds=timeout,input_origin=origin)

    def test_inventory_rejects_extra_and_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            stage=Path(folder)
            for name in FILES:
                p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(request()) if name=='request.json' else '# CPU fixture\n')
            rows=[dict(path=n,bytes=(stage/n).stat().st_size,sha256=digest(stage/n)) for n in sorted(FILES)]
            inv=stage/'upload-allowlist.json';inv.write_text(json.dumps(dict(files=rows)))
            inspect(stage)
            (stage/'raw.wav').write_bytes(b'not approved')
            with self.assertRaises(ValueError):inspect(stage)
            (stage/'raw.wav').unlink();inv.write_text(json.dumps(dict(files=rows+[rows[0]])))
            with self.assertRaises(ValueError):inspect(stage)

    def test_remote_import_does_not_dirty_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            stage=Path(folder)
            for name in FILES:
                p=stage/name;p.parent.mkdir(parents=True,exist_ok=True)
                if name=='request.json':p.write_text(json.dumps(request()))
                else:shutil.copyfile(name,p)
            rows=[dict(path=n,bytes=(stage/n).stat().st_size,sha256=digest(stage/n)) for n in sorted(FILES)]
            (stage/'upload-allowlist.json').write_text(json.dumps(dict(files=rows)))
            code='import sys;sys.dont_write_bytecode=True;from pathlib import Path;from research.h3_personal.director_tts_job import inspect;inspect(Path.cwd())'
            subprocess.run([sys.executable,'-c',code],cwd=stage,check=True,capture_output=True,timeout=15)
            self.assertFalse(list(stage.rglob('*.pyc')))

    def test_inline_bootstrap_is_valid_python(self):
        tree=ast.parse(Path('research/launch_h3_irodori_turn.py').read_text())
        values=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bootstrap' for t in n.targets)]
        self.assertEqual(len(values),1);bootstrap=ast.literal_eval(values[0]);ast.parse(bootstrap)
        self.assertLess(bootstrap.index('sys.dont_write_bytecode=True'),bootstrap.index('from research.'))
        self.assertIn("'ffmpeg','libsndfile1','git'",bootstrap)
        self.assertIn("for name in ('ffmpeg','ffprobe','git','uv')",bootstrap)
        self.assertLess(bootstrap.index("for name in ('ffmpeg','ffprobe','git','uv')"),bootstrap.index('runpy.run_path'))

if __name__=='__main__':unittest.main()
