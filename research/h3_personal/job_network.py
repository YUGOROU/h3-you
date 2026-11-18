"""HF Network Groups REST contract, no SDK upgrade/automatic creation/retry.

Fields verified against upstream huggingface_hub._jobs_api._create_job_spec.
No assumption that a group implies co-location, encryption or country proof.
"""
import re
from .director_tts_job import IMAGE
from .network_voxcpm import validate_config,SCHEMA

TERMINAL={'CANCELED','COMPLETED','ERROR','DELETED'}

def network(group,aliases=()):
    if type(group) is not str or not re.fullmatch('[a-z0-9](?:[a-z0-9-]{0,44}[a-z0-9])?',group):raise ValueError('network group')
    if type(aliases) not in (tuple,list) or len(aliases)>1 or any(type(a) is not str or not re.fullmatch('[a-z0-9](?:[a-z0-9-]{0,32}[a-z0-9])?',a) for a in aliases):raise ValueError('single unique service alias required')
    return dict(group=group,**({'aliases':list(aliases)} if aliases else {}))

def native_config(spec):
    config=spec.get('live_direct_session',{}).get('native_tts',{})
    if type(config) is not dict:raise ValueError('native backend manifest required')
    return validate_config(config) if config.get('schema')==SCHEMA else None

def assert_idle(api,spec,token,*,namespace='YUGOROU'):
    """Permit only the exact manifest's owned L4 source, never other GPUs.

    SDK1.7.2 drops network metadata: read the raw control-plane JSON, not a
    fabricated SDK attribute. This performs GET only. Source owner must have
    separately confirmed immutable creation, finite timeout and placement.
    """
    from .job_compute_profile import validate_namespace
    namespace=validate_namespace(namespace)
    config=native_config(spec)
    if namespace != 'YUGOROU' and config is not None:
        raise ValueError('backup Jobs cannot share main account native network groups')
    jobs=[j for j in api.list_jobs(namespace=namespace) if j.status.stage not in TERMINAL]
    if config is None:
        if jobs:raise ValueError('HF not idle')
        return
    expected=config['connection']
    if len(jobs)!=1 or jobs[0].id!=expected['server_job_id'] or str(jobs[0].flavor)!='l4x1':
        raise ValueError('only exact owned L4 network source may coexist')
    import httpx
    with httpx.Client(timeout=httpx.Timeout(15,connect=5),follow_redirects=False,trust_env=False) as client:
        response=client.get('https://huggingface.co/api/jobs/YUGOROU/'+expected['server_job_id'],headers={'Authorization':'Bearer '+token})
        if response.status_code!=200:raise ValueError('source Job inspection unavailable')
        assert_source(response.json(),config)

def assert_source(row,config):
    config=validate_config(config);c=config['connection']
    if (type(row) is not dict or any(type(row.get(k)) is not dict for k in ('owner','network','labels','status'))
            or type(row.get('expose') or {}) is not dict):raise ValueError('source Job metadata shape')
    expose=row.get('expose') or {}
    if (type(row) is not dict or row.get('id')!=c['server_job_id'] or row.get('owner',{}).get('name')!='YUGOROU'
            or row.get('flavor')!='l4x1' or row.get('dockerImage')!=IMAGE
            or row.get('network')!=dict(group=c['group'],aliases=[c['alias']])
            or row.get('retry')!=0 or type(row.get('retry')) is not int
            or row.get('labels',{}).get('h3session')!=config['session']
            or row.get('labels',{}).get('h3boot')!=config['boot']
            or row.get('status',{}).get('stage')!='RUNNING'
            or expose.get('portsPublic',[]) or expose.get('ports',[])):
        raise ValueError('native network source Job contract mismatch')
