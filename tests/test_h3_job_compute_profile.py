"""Offline split-compute tests. Fake token strings and fake SDK/HTTP only."""
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from research.h3_personal.job_compute_profile import (
    ComputeProfile, load_compute_profile, validate_namespace, _read_backup_token,
)
from research.h3_personal.direct_job_api import submit_job
from research.h3_personal.job_av_endpoint import attest_job
from research.h3_personal import job_network

MAIN = 'hf_MainFixtureSecret'
BACKUP = 'hf_BackupFixtureSecret'
JOB = 'a' * 24
IMAGE = 'image@sha256:' + 'b' * 64


def fake_api(name='YUGOROU', **identity):
    api=Mock(endpoint='https://huggingface.co')
    api.whoami.return_value=dict(name=name, **identity)
    api.list_jobs.return_value=[]
    return api


def body():
    return dict(flavor='h200x2', timeoutSeconds=1200, attempts=1,
                expose={'ports':[7860], 'portsPublic':[]},
                secrets={'HF_TOKEN':MAIN})


class ComputeProfileTests(unittest.TestCase):
    def test_default_keeps_main_without_reading_backup_or_changing_env(self):
        api=fake_api()
        with patch('research.h3_personal.job_compute_profile._read_backup_token') as read, patch.dict(os.environ, {'HF_TOKEN':'unrelated'}):
            result=load_compute_profile(api, MAIN)
            self.assertIs(result.api,api)
            self.assertEqual(result.token,MAIN)
            self.assertEqual(result.namespace,'YUGOROU')
            self.assertEqual(os.environ['HF_TOKEN'],'unrelated')
            read.assert_not_called()
        self.assertEqual(result.metadata(),{'namespace':'YUGOROU'})
        self.assertNotIn(MAIN,repr(result))
        with self.assertRaises(FrozenInstanceError):result.namespace='other'

    def load_backup(self, identity=None):
        main=fake_api()
        backup=fake_api('2nd2nd', isPro=False)
        if identity is not None:backup.whoami.return_value=identity
        with patch('research.h3_personal.job_compute_profile._read_backup_token',return_value=BACKUP), patch('huggingface_hub.HfApi',return_value=backup) as ctor:
            result=load_compute_profile(main,MAIN,backup=True)
        ctor.assert_called_once_with(endpoint='https://huggingface.co',token=BACKUP)
        return result, main, backup

    def test_explicit_backup_verifies_owner_without_pro_requirement(self):
        result,main,backup=self.load_backup()
        self.assertIs(result.api,backup)
        self.assertEqual(result.namespace,'2nd2nd')
        self.assertEqual(result.token,BACKUP)
        self.assertEqual(result.metadata(),{'namespace':'2nd2nd'})
        self.assertNotIn(BACKUP,repr(result))
        self.assertNotIn(MAIN,repr(result))
        main.whoami.assert_called_once_with()
        backup.whoami.assert_called_once_with()
        backup.list_jobs.assert_not_called()

    def test_idle_checks_bound_namespace_only_and_never_cancels(self):
        for namespace in ('YUGOROU','2nd2nd'):
            api=fake_api(namespace)
            profile=ComputeProfile(api,BACKUP,namespace)
            api.list_jobs.return_value=[NS(status=NS(stage='CANCELED')),NS(status=NS(stage='COMPLETED'))]
            profile.assert_idle()
            api.list_jobs.assert_called_once_with(namespace=namespace)
            for stage in ('RUNNING','SCHEDULING','UPDATING',None):
                api.list_jobs.return_value=[NS(status=NS(stage=stage))]
                with self.assertRaises(ValueError):profile.assert_idle()
            api.cancel_job.assert_not_called()

    def test_backup_identity_and_known_permission_denials_fail_closed(self):
        invalid=[{'name':'YUGOROU'}, {'name':'someone'}, {},
                 {'name':'2nd2nd','auth':{'accessToken':{'role':'read'}}},
                 {'name':'2nd2nd','auth':{'accessToken':{'role':'fineGrained','fineGrained':{'global':[]}}}},
                 {'name':'2nd2nd','auth':{'accessToken':{'role':'fineGrained','fineGrained':{'global':['job.read']}}}}]
        for identity in invalid:
            with self.subTest(identity=identity), self.assertRaises(ValueError):self.load_backup(identity)
        result,_,_=self.load_backup({'name':'2nd2nd','auth':{'accessToken':{
            'role':'fineGrained','fineGrained':{'global':['job.read','job.write']}}}})
        self.assertEqual(result.namespace,'2nd2nd')

    def test_invalid_main_selection_and_distinct_credential(self):
        for value in (1,None,'yes'):
            with self.assertRaises(ValueError):load_compute_profile(fake_api(),MAIN,backup=value)
        for api in (fake_api('other'),Mock(endpoint='http://huggingface.co')):
            with self.assertRaises(ValueError):load_compute_profile(api,MAIN,backup=True)
        with patch('research.h3_personal.job_compute_profile._read_backup_token',return_value=MAIN), self.assertRaises(ValueError):
            load_compute_profile(fake_api(),MAIN,backup=True)
        for value in ('',None,'has space','bad\n', 'x'*8193):
            with self.assertRaises(ValueError):load_compute_profile(fake_api(),value)

    def test_backup_file_bounded_regular_nonsymlink_and_not_logged(self):
        with tempfile.TemporaryDirectory() as tmp, patch('research.h3_personal.job_compute_profile.Path.home',return_value=Path(tmp)):
            path=Path(tmp)/'.cache/huggingface-backup/token-backup'
            path.parent.mkdir(parents=True)
            path.write_text(BACKUP+'\n')
            self.assertEqual(_read_backup_token(),BACKUP)
            path.write_bytes(b'x'*8193)
            with self.assertRaises(ValueError):_read_backup_token()
            path.write_bytes(b'\xff')
            with self.assertRaises(ValueError):_read_backup_token()
            path.unlink();path.symlink_to(Path(tmp)/'missing')
            with self.assertRaises(ValueError):_read_backup_token()

    def test_namespace_allowlist_is_exact(self):
        for value in ('2nd2nd/other','YUGOROU/','other','2nd2nd\n',True,None):
            with self.assertRaises(ValueError):validate_namespace(value)
        self.assertEqual(validate_namespace('YUGOROU'),'YUGOROU')
        self.assertEqual(validate_namespace('2nd2nd'),'2nd2nd')

    def test_job_post_bills_backup_but_payload_asset_secret_stays_main(self):
        client=Mock();client.post.return_value.status_code=201
        client.post.return_value.json.return_value={'id':JOB}
        payload=body()
        result=submit_job(client,BACKUP,payload,namespace='2nd2nd')
        self.assertEqual(result.id,JOB)
        client.post.assert_called_once_with('https://huggingface.co/api/jobs/2nd2nd',
            headers={'Authorization':'Bearer '+BACKUP},json=payload,follow_redirects=False)
        self.assertEqual(payload['secrets']['HF_TOKEN'],MAIN)
        client.reset_mock()
        with self.assertRaises(ValueError):submit_job(client,BACKUP,payload,namespace='other')
        client.post.assert_not_called()

    def test_backup_attestation_inspects_and_binds_expected_owner(self):
        api=fake_api('2nd2nd')
        job=NS(id=JOB,owner=NS(name='2nd2nd'),status=NS(stage='RUNNING'),
               docker_image=IMAGE,command=['worker'],flavor='h200x2')
        api.inspect_job.return_value=job
        with patch('research.h3_personal.job_av_endpoint.http.client.HTTPSConnection') as factory:
            factory.return_value.getresponse.return_value.status=401
            permit=attest_job(api,job_id=JOB,port=7860,expected_image=IMAGE,
                              expected_command=['worker'],namespace='2nd2nd')
            self.assertEqual(permit.namespace,'2nd2nd')
            api.inspect_job.assert_called_once_with(job_id=JOB,namespace='2nd2nd')
            factory.reset_mock();job.owner.name='YUGOROU'
            with self.assertRaises(ValueError):
                attest_job(api,job_id=JOB,port=7860,expected_image=IMAGE,
                           expected_command=['worker'],namespace='2nd2nd')
            factory.assert_not_called()

    def test_network_idle_uses_backup_and_rejects_network_group_before_io(self):
        api=fake_api('2nd2nd')
        spec={'live_direct_session':{'native_tts':{'schema':'h3-native-voxcpm-backend-v1'}}}
        job_network.assert_idle(api,spec,BACKUP,namespace='2nd2nd')
        api.list_jobs.assert_called_once_with(namespace='2nd2nd')
        api.reset_mock()
        with patch.object(job_network,'native_config',return_value={'connection':{}}), self.assertRaises(ValueError):
            job_network.assert_idle(api,spec,BACKUP,namespace='2nd2nd')
        api.list_jobs.assert_not_called()


if __name__=='__main__':unittest.main()
