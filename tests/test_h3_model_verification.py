import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research.h3_personal.model_verification import hash_asset, encoder_inventory, GROUPS


class ModelVerificationTests(unittest.TestCase):
    def fixture(self):
        files={'model_index.json':'a'*64}
        for group in GROUPS:
            files[group+'/config.json']='b'*64
            if group in ('vae','audio_vae','text_encoder'):
                files[group+'/weights.safetensors']='c'*64
        return files,dict(schema='h3-local-model-lock-v1',revision='d'*40,files=dict(files))

    def test_one_open_computes_sha_and_git_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'asset';data=b'abc'*500000;path.write_bytes(data)
            original=Path.open
            calls=[]
            def opened(p,*args,**kwargs):
                calls.append(p);return original(p,*args,**kwargs)
            with patch.object(Path,'open',opened):
                sha,git=hash_asset(path,git=True)
            self.assertEqual(calls,[path])
            self.assertEqual(sha,hashlib.sha256(data).hexdigest())
            self.assertEqual(git,hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest())
            self.assertEqual(hash_asset(path),(sha,None))

    def test_changing_size_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'asset';p.write_bytes(b'abc')
            with patch('research.h3_personal.model_verification.os.fstat',return_value=SimpleNamespace(st_size=4)):
                with self.assertRaises(ValueError):hash_asset(p)

    def test_encoder_selection_no_io_or_input_mutation(self):
        files,lock=self.fixture();files['transformer_ref/weights.safetensors']='e'*64
        before=copy.deepcopy(files)
        with patch.object(Path,'open',side_effect=AssertionError('must not re-read bytes')):
            self.assertEqual(encoder_inventory(files,lock),lock['files'])
        self.assertEqual(files,before)

    def test_inventory_and_lock_drift_rejected(self):
        files,lock=self.fixture()
        for name in files:
            bad=dict(files);bad[name]='f'*64
            with self.assertRaises(ValueError):encoder_inventory(bad,lock)
        bad=dict(files);bad['processor/unexpected.json']='f'*64
        with self.assertRaises(ValueError):encoder_inventory(bad,lock)
        with self.assertRaises(ValueError):encoder_inventory(files,dict(lock,revision='main'))

    def test_missing_encoder_weights_rejected_even_if_lock_agrees(self):
        files,lock=self.fixture();del files['vae/weights.safetensors'];lock['files']=dict(files)
        with self.assertRaises(ValueError):encoder_inventory(files,lock)
