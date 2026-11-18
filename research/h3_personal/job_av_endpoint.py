"""Attest a direct authenticated Jobs host before sending application keys.

The gateway is the HF provider, not another relay instance. No model/job launch.
"""
from dataclasses import dataclass,field
import http.client
import re

_CAP=object()


@dataclass(frozen=True)
class JobEndpoint:
    base_url:str
    job_id:str
    _capability:object=field(default=None,repr=False)
    namespace:str='YUGOROU'

    def permits(self,url):return self._capability is _CAP and self.base_url==url


def attest_job(api,*,job_id,port,expected_image,expected_command,timeout=10,namespace='YUGOROU'):
    from .job_compute_profile import validate_namespace
    namespace=validate_namespace(namespace)
    if (getattr(api,'endpoint',None)!='https://huggingface.co'
            or type(job_id) is not str or not re.fullmatch('[0-9a-f]{24}',job_id)
            or type(port) is not int or port!=7860
            or type(expected_image) is not str or not re.search(r'@sha256:[a-f0-9]{64}$',expected_image)
            or type(expected_command) is not list or not expected_command):
        raise ValueError('canonical immutable direct Job binding required')
    job=api.inspect_job(job_id=job_id,namespace=namespace)
    if (job.id!=job_id or job.owner.name!=namespace or job.status.stage!='RUNNING'
            or job.docker_image!=expected_image or job.command!=expected_command
            or str(job.flavor)!='h200x2'):
        raise ValueError('direct Job identity/runtime differs')
    url=f'https://{job_id}--{port}.hf.jobs'
    c=http.client.HTTPSConnection(f'{job_id}--{port}.hf.jobs',timeout=timeout)
    try:
        c.request('GET','/health',headers={'Accept-Encoding':'identity','Cache-Control':'no-store'})
        response=c.getresponse()
        if response.status not in (401,403,404):raise ValueError('anonymous Job gateway access not denied')
    finally:c.close()
    return JobEndpoint(url,job_id,_CAP,namespace)
