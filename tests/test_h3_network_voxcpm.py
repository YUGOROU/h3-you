"""Real loopback TLS/native protocol; no Jobs, model, media devices or installs."""
import hashlib
import os
from pathlib import Path
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock,patch
from research.h3_personal.network_voxcpm import SCHEMA,RemoteNetworkVox,validate_config,address
from research.h3_personal.network_voxcpm_service import Server
from research.h3_personal.native_tts_backend import prepare_direct_tts,native_job_secrets,config_sha256
from research.h3_personal.voxcpm_endpoint_protocol import PINS
from research.h3_personal.job_network import network,assert_source
from research.h3_personal.director_tts_job import IMAGE

class NetworkNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory();cls.root=Path(cls.directory.name)
        cls.cert=cls.root/'cert.pem';cls.key=cls.root/'key.pem'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(cls.key),'-out',str(cls.cert),
                        '-days','1','-subj','/CN=h3-native-test'],check=True,timeout=10,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        cls.fp=hashlib.sha256(ssl.PEM_cert_to_DER_cert(cls.cert.read_text())).hexdigest()
    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def config(self):
        return dict(schema=SCHEMA,connection=dict(group='h3-pair-unit',alias='voxcpm',port=17861,
            certificate_sha256=self.fp,server_job_id='c'*24),session='a'*32,boot='b'*32,expires_at=time.time()+120,
            pins=dict(PINS),preroll=dict(sample_rate=48000,reserve_samples=48000,max_reserve_samples=96000,max_wait_seconds=2.0))

    def start(self):
        config=self.config()
        class Candidate:
            sample_rate=48000
            def stream(self,text):yield [.125,-.5,1.0];yield [.0,.25]
            def export_evidence(self):return dict(runtime={'synthetic':True},completions=[])
        server=Server(('127.0.0.1',0),Candidate(),certificate=str(self.cert),key_file=str(self.key),
            key='k'*40,session=config['session'],boot=config['boot'],expires_at=config['expires_at'])
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01});thread.start()
        def stop():server.shutdown();server.server_close();thread.join(2);self.assertFalse(thread.is_alive())
        self.addCleanup(stop)
        config['connection']['port']=server.server_address[1]
        def connector(addr,timeout):
            self.assertEqual(addr,('prefix-voxcpm',server.server_address[1]))
            return socket.create_connection(('127.0.0.1',server.server_address[1]),timeout=timeout)
        return server,config,connector

    def test_real_tls_ready_two_native_streams_exact_and_evidence(self):
        server,config,connector=self.start()
        client=RemoteNetworkVox(config,key='k'*40,connector=connector,environ={'HF_NETWORK_GROUP_PREFIX':'prefix-'})
        self.addCleanup(client.close)
        with patch('http.client.HTTPSConnection',side_effect=AssertionError('no HTTP/TURN media')):
            self.assertEqual(client.read('tts_ready',time.monotonic()+3)['epoch'],0)
            for epoch in (1,2):
                body=b''.join(client.stream('fresh',epoch,time.monotonic()+3))
                self.assertEqual(body,struct.pack('<5f',.125,-.5,1.0,.0,.25))
            evidence=client.evidence(time.monotonic()+3)
        self.assertEqual(server.gate.count,2);self.assertEqual(evidence['binding']['boot'],config['boot'])

    def test_bad_certificate_key_or_boot_fail_before_generation(self):
        for change in ('certificate','key','boot'):
            with self.subTest(change=change):
                server,config,connector=self.start()
                if change=='certificate':config['connection']['certificate_sha256']='0'*64
                if change=='boot':config['boot']='d'*32
                client=RemoteNetworkVox(config,key='wrong'*8 if change=='key' else 'k'*40,connector=connector,environ={'HF_NETWORK_GROUP_PREFIX':'prefix-'})
                self.addCleanup(client.close)
                with self.assertRaises(RuntimeError):client.read('tts_ready',time.monotonic()+2)
                self.assertEqual(server.gate.count,0)

    def test_replay_fails_without_second_generation(self):
        server,config,connector=self.start()
        client=RemoteNetworkVox(config,key='k'*40,connector=connector,environ={'HF_NETWORK_GROUP_PREFIX':'prefix-'})
        self.addCleanup(client.close);client.read('tts_ready',time.monotonic()+3)
        list(client.stream('fresh',1,time.monotonic()+3))
        with self.assertRaises(RuntimeError):list(client.stream('fresh',1,time.monotonic()+3))
        self.assertEqual(server.gate.count,1)

    def test_manifest_prefix_and_digest_are_strict(self):
        config=self.config();self.assertEqual(validate_config(config),config)
        self.assertEqual(len(config_sha256(config)),64)
        self.assertEqual(address(config,{'HF_NETWORK_GROUP_PREFIX':'prefix-'}),('prefix-voxcpm',17861))
        for env in ({},{'HF_NETWORK_GROUP_PREFIX':'https://evil/'},{'HF_NETWORK_GROUP_PREFIX':'-bad.'}):
            with self.assertRaises(ValueError):address(config,env)
        for key,bad in (('group','../escape'),('alias','evil:443'),('port',True),('certificate_sha256','no-pin')):
            value=self.config();value['connection'][key]=bad
            with self.assertRaises(ValueError):validate_config(value)

    def test_backend_selection_precedes_local_or_https_factory(self):
        config=self.config();spec={'live_direct_session':dict(tts_mode='native_voxcpm',native_tts=config)}
        with patch.dict(os.environ,{'H3_NATIVE_TTS_KEY':'k'*40},clear=True),\
             patch('research.h3_personal.native_tts_backend.RemoteVoxCPM',side_effect=AssertionError):
            self.assertEqual(set(native_job_secrets(spec)),{'H3_NATIVE_TTS_KEY'})
            future=prepare_direct_tts('/unused',spec,Mock(side_effect=AssertionError))
            self.assertIsInstance(future.result(),RemoteNetworkVox)
            self.assertNotIn('H3_NATIVE_TTS_KEY',os.environ);future.result().close()

    def test_offline_source_bundle_imports_without_models_or_installs(self):
        import zipfile
        from research.prepare_h3_network_tts_job import bundle,payload
        config=self.config()
        spec=dict(candidate='voxcpm2',native_pins=dict(PINS),network_service={k:v for k,v in config['connection'].items() if k!='server_job_id'})
        with tempfile.TemporaryDirectory() as tmp:
            artifact=bundle(Path(tmp).resolve()/'fresh',spec)
            stage=Path(tmp)/'unpacked';stage.mkdir()
            with zipfile.ZipFile(artifact) as z:
                self.assertLess(len(z.infolist()),32);z.extractall(stage)
                self.assertFalse(any('key.pem' in n or '.venv' in n for n in z.namelist()))
            result=subprocess.run([__import__('sys').executable,'-c',
                'import research.hf_h3_voxcpm_network_worker;import research.h3_personal.network_voxcpm_service;print("ok")'],
                cwd=stage,timeout=5,capture_output=True,text=True,env={'PATH':'/usr/bin:/bin'})
            self.assertEqual(result.returncode,0,result.stderr)
            p=payload('YUGOROU/fresh','a'*40,'b'*64,group='new-pair',session='a'*32,boot='b'*32,
                      token='secret-token',key='k'*40,certificate='public-cert',private_key='private-secret')
            self.assertEqual(p['flavor'],'l4x1');self.assertEqual(p['attempts'],1);self.assertNotIn('expose',p)
            self.assertNotIn('private-secret',repr(p['command']));self.assertNotIn('private-secret',repr(p['environment']))
            self.assertIn('hf_h3_voxcpm_network_worker.py',p['command'][2])

    def test_rest_network_fields_and_source_identity(self):
        config=self.config();self.assertEqual(network('fresh-pair',['voxcpm']),dict(group='fresh-pair',aliases=['voxcpm']))
        row=dict(id='c'*24,owner={'name':'YUGOROU'},flavor='l4x1',dockerImage=IMAGE,
            network=network(config['connection']['group'],['voxcpm']),retry=0,
            labels=dict(h3session=config['session'],h3boot=config['boot']),status={'stage':'RUNNING'})
        assert_source(row,config)
        for change in ({'flavor':'h200x2'},{'network':network('other',['voxcpm'])},{'retry':True},{'expose':{'ports':[7860]}}):
            with self.subTest(change=change),self.assertRaises(ValueError):assert_source(dict(row,**change),config)

if __name__=='__main__':unittest.main()


class PayloadCommandTests(unittest.TestCase):
    """The pinned pytorch image has `python` on PATH, not /opt/venv."""

    def test_command_uses_the_image_python(self):
        from research.prepare_h3_network_tts_job import payload
        row = payload('YUGOROU/fresh', 'a'*40, 'b'*64, group='h3-pair-abc', session='a'*32, boot='b'*32,
                      token='t', key='k'*40, certificate='-----BEGIN CERTIFICATE-----\n',
                      private_key='-----BEGIN PRIVATE KEY-----\n')
        self.assertEqual(row['command'][:2], ['python', '-c'])
        self.assertEqual(row['flavor'], 'l4x1')
        self.assertEqual(row['timeoutSeconds'], 1200)
        self.assertEqual(row['attempts'], 1)
        self.assertEqual(row['network'], dict(group='h3-pair-abc', aliases=['voxcpm']))
        self.assertIsNone(row.get('expose'))
