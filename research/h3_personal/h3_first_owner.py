"""Finite actual H3 allocation with one authenticated deferred launch; no fallback."""
import hashlib
import json
import math
import re
import secrets
import time
from pathlib import Path
from types import SimpleNamespace

H3_IMAGE='ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:34ef8907bf0c4329d81be5fd4a6e3056b806e65fa0bba439733ab8a45bef31cf'
ROOT=Path(__file__).resolve().parents[2]


class DeferredH3Owner:
    def __init__(self,api,client,token,root,group):
        self.api=api;self.client=client;self.token=token;self.root=Path(root);self.group=group
        self.campaign=secrets.token_hex(16);self.key=secrets.token_hex(32)
        self.job=None;self.command=None;self.ready=None;self.claimed=False;self.expires_at=None
        self.bootstrap_sha=hashlib.sha256((ROOT/'research/bootstrap_h3_fast_ref2va.py').read_bytes()).hexdigest()
        self.snapshot=self.root/'h3-first-code';self.source_binding=None

    def remaining(self):
        return 0 if self.expires_at is None else self.expires_at-time.time()

    def _record(self,name,row):
        from research.run_h3_native_coupled import persist
        persist(self.root/name,row)

    def start(self):
        from research.h3_personal.direct_job_api import submit_job,JobCreationRejected
        from research.h3_personal.job_network import network
        from research.h3_personal.h3_deferred_boot import __file__ as module_file
        if self.job is not None:raise ValueError('no reservation replay')
        import shutil
        from research.launch_h3_direct_conversation import first_party_sources
        self.snapshot.mkdir(mode=0o700)
        inventory=[]
        for path in first_party_sources(ROOT):
            rel=path.relative_to(ROOT);target=self.snapshot/rel
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            inventory.append(dict(path=str(rel),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        self._record('h3-first-code.json',dict(files=inventory))
        self.expires_at=int(time.time())+1200
        source=(self.snapshot/'research/h3_personal/h3_deferred_boot.py').read_text()
        self.command=['/opt/venv/bin/python','-c',source]
        self._record('h3-first-intent.json',dict(campaign=self.campaign,group=self.group,
            image=H3_IMAGE,command=self.command,expires_at=self.expires_at,bootstrap_sha256=self.bootstrap_sha))
        self._record('h3-first-guard.json',dict(state='unknown-no-replay',campaign=self.campaign))
        payload=dict(dockerImage=H3_IMAGE,command=self.command,arguments=[],flavor='h200x2',
            timeoutSeconds=1200,attempts=1,expose={'ports':[7860],'portsPublic':[]},network=network(self.group),
            environment={'H3_DEFERRED_CAMPAIGN':self.campaign,'H3_DEFERRED_GROUP':self.group,
                'H3_DEFERRED_EXPIRES_AT':str(self.expires_at),'H3_DEFERRED_BOOTSTRAP_SHA256':self.bootstrap_sha,
                'HF_HUB_DISABLE_TELEMETRY':'1','PYTHONDONTWRITEBYTECODE':'1'},
            secrets={'HF_TOKEN':self.token,'H3_DEFERRED_KEY':self.key},labels={'scope':'h3-first-native','h3campaign':self.campaign})
        def created(job):
            self.job=job
            self._record('h3-first-submission.json',dict(job_id=job.id,campaign=self.campaign))
        try:submit_job(self.client,self.token,payload,on_created=created)
        except JobCreationRejected as error:
            self._record('h3-first-rejection.json',dict(status=error.status,detail=error.detail))
            raise
        deadline=time.monotonic()+min(300,self.remaining()-660)
        while time.monotonic()<deadline:
            raw=self._raw()
            if raw.get('status',{}).get('stage') in ('ERROR','CANCELED','COMPLETED','DELETED'):
                raise RuntimeError('H3-first ended before deferred readiness')
            if raw.get('status',{}).get('stage')=='RUNNING':
                self._identity(raw)
                import httpx
                try:response=self._request('GET','/ready')
                except (httpx.TimeoutException,httpx.NetworkError):
                    time.sleep(2);continue
                if response.status_code==200:
                    row=response.json()
                    if (type(row) is not dict or set(row)!={'schema','campaign','group','boot','job_id','expires_at','state'}
                            or row['schema']!='h3-deferred-ready-v1' or row['campaign']!=self.campaign
                            or row['group']!=self.group or row['job_id']!=self.job.id
                            or row['state']!='waiting' or row['expires_at']!=self.expires_at
                            or type(row['boot']) is not str or re.fullmatch('[a-f0-9]{32}',row['boot']) is None):
                        raise ValueError('H3-first readiness binding')
                    if self.remaining()<660:raise TimeoutError('H3-first remaining lease')
                    self.ready=row;self._record('h3-first-ready.json',row);return
            time.sleep(2)
        raise TimeoutError('H3-first readiness deadline')

    def _raw(self):
        response=self.client.get('https://huggingface.co/api/jobs/YUGOROU/'+self.job.id,
            headers={'Authorization':'Bearer '+self.token})
        if response.status_code!=200:raise RuntimeError('H3-first inspection unavailable')
        return response.json()

    def _identity(self,row):
        if (row.get('id')!=self.job.id or row.get('owner',{}).get('name')!='YUGOROU'
                or row.get('dockerImage')!=H3_IMAGE or row.get('command')!=self.command
                or row.get('flavor')!='h200x2' or row.get('network')!={'group':self.group}
                or row.get('resourceGroupId') is not None or row.get('retry')!=0
                or type(row.get('expose')) is not dict or row['expose'].get('ports')!=[7860]
                or row['expose'].get('portsPublic',[])!=[]
                or row.get('labels',{}).get('h3campaign')!=self.campaign):
            raise ValueError('H3-first immutable Job identity')

    def _request(self,method,path,body=None):
        return self.client.request(method,'https://'+self.job.id+'--7860.hf.jobs'+path,
            headers={'Authorization':'Bearer '+self.token,'X-H3-Launch-Key':self.key,'Content-Type':'application/json'},
            **({'json':body} if body is not None else {}))

    def bind_source(self,source_id,command,revision):
        if self.source_binding is not None:raise ValueError('no source rebinding')
        self.source_binding=dict(job_id=source_id,command=command,revision=revision)

    def admit(self,api,spec,token):
        from research.h3_personal.job_network import assert_source,native_config,TERMINAL
        config=native_config(spec)
        if (token!=self.token or self.ready is None or config is None
                or config['connection']['group']!=self.group or self.remaining()<180):
            raise ValueError('explicit H3-first/source admission required')
        jobs=[j for j in api.list_jobs(namespace='YUGOROU') if j.status.stage not in TERMINAL]
        if {j.id for j in jobs}!={self.job.id,config['connection']['server_job_id']} or len(jobs)!=2:
            raise ValueError('only exact H3-first and source may coexist')
        if self.source_binding is None or config['connection']['server_job_id']!=self.source_binding['job_id']:
            raise ValueError('unbound deferred source identity')
        raw=self._raw();self._identity(raw)
        if raw.get('status',{}).get('stage')!='RUNNING':raise ValueError('H3-first not running')
        response=self.client.get('https://huggingface.co/api/jobs/YUGOROU/'+config['connection']['server_job_id'],
            headers={'Authorization':'Bearer '+token})
        if response.status_code!=200:raise RuntimeError('source inspection unavailable')
        raw=response.json();assert_source(raw,config)
        if (raw.get('command')!=self.source_binding['command']
                or raw.get('labels',{}).get('deployrev')!=self.source_binding['revision']):
            raise ValueError('deferred source command/publication drift')

    def launch(self,api,stage,token,payload,revision,allow_sha):
        from research.h3_personal.job_network import native_config
        spec=json.loads((stage/'spec.json').read_text());self.admit(api,spec,token)
        if self.claimed:raise ValueError('no deferred launch replay')
        if payload['dockerImage']!=H3_IMAGE:raise ValueError('original H3 image changed')
        actual=hashlib.sha256((stage/'research/bootstrap_h3_fast_ref2va.py').read_bytes()).hexdigest()
        if actual!=self.bootstrap_sha:raise ValueError('deferred bootstrap pin changed')
        expires=min(self.expires_at,native_config(spec)['expires_at'])
        if expires-time.time()<180:raise TimeoutError('joint lease margin')
        s=payload['secrets']
        capsule=dict(schema='h3-deferred-launch-v1',campaign=self.campaign,group=self.group,
            boot=self.ready['boot'],job_id=self.job.id,expires_at=expires,repo=spec['publication_repo'],
            revision=revision,allow_sha256=allow_sha,bootstrap_sha256=self.bootstrap_sha,
            spec_sha256=hashlib.sha256((stage/'spec.json').read_bytes()).hexdigest(),
            producer_key=s['H3_DIRECT_PRODUCER_KEY'],consumer_key=s['H3_DIRECT_CONSUMER_KEY'],native_key=s['H3_NATIVE_TTS_KEY'])
        self.claimed=True  # Unknown response is never retried.
        response=self._request('POST','/launch',capsule)
        if response.status_code!=202:raise RuntimeError('deferred capsule not accepted; no replay')
        self._record('h3-first-launch.json',{k:v for k,v in capsule.items() if not k.endswith('_key')})
        return self.job,H3_IMAGE,self.command
