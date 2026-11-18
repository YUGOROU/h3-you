"""Single-request Jobs creation compatibility; no SDK install or retry.

Payload fields match upstream huggingface_hub._jobs_api._create_job_spec.
https://huggingface.co/docs/hub/en/jobs-configuration#expose-ports
Caller owns a canonical HTTPS client with redirects disabled for this POST.
"""
import re
from types import SimpleNamespace


class JobCreationRejected(RuntimeError):
    def __init__(self,status,detail):
        self.status=status;self.detail=detail
        super().__init__('direct Job creation rejected (HTTP '+str(status)+')')


def submit_job(client,token,payload,on_created=None,*,five_minute_test=False,long_session=False,namespace='YUGOROU'):
    from .job_compute_profile import validate_namespace
    namespace=validate_namespace(namespace)
    if (type(five_minute_test) is not bool or type(long_session) is not bool
            or (long_session and five_minute_test)
            or type(payload) is not dict or payload.get('flavor')!='h200x2'
            or payload.get('timeoutSeconds')!=(10800 if long_session else 300 if five_minute_test else 1200) or type(payload.get('timeoutSeconds')) is not int
            or payload.get('attempts')!=1 or type(payload.get('attempts')) is not int
            or payload.get('expose')!={'ports':[7860],'portsPublic':[]}
            or type(token) is not str or not token or any(ord(c)<33 or ord(c)>126 for c in token)):
        raise ValueError('finite authenticated direct Job payload required')
    # Deliberately one POST: unknown outcomes require inspection, never replay.
    response=client.post('https://huggingface.co/api/jobs/'+namespace,
        headers={'Authorization':'Bearer '+token},json=payload,follow_redirects=False)
    if response.status_code not in (200,201):
        # Detail is for a private receipt only, never exception text/stdout.
        detail=str(getattr(response,'text',''))[:8192]
        for secret in sorted({token,*payload.get('secrets',{}).values()},key=len,reverse=True):
            if isinstance(secret,str) and secret:detail=detail.replace(secret,'[REDACTED]')
        detail=re.sub(r'hf_[A-Za-z0-9]+','[REDACTED]',detail)[:2048]
        raise JobCreationRejected(response.status_code,detail)
    value=response.json()
    if type(value) is not dict or not re.fullmatch('[a-f0-9]{24}',str(value.get('id',''))):
        raise RuntimeError('unknown Job creation outcome; inspect namespace before any resubmission')
    job=SimpleNamespace(id=value['id'])
    if on_created is not None:
        on_created(job)
    return job
