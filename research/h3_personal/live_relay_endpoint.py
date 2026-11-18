"""Private HF relay endpoint attestation, no provider actions at import.

Metadata must pin owner/revision and private visibility; anonymous gateway
must deny access before any session key or personal media is sent. No redirects.
"""
from dataclasses import dataclass,field
import http.client
import re
from urllib.parse import urlsplit

_CAP=object()


@dataclass(frozen=True)
class PrivateEndpoint:
    base_url: str
    space_id: str
    revision: str
    _capability: object = field(repr=False,default=None)

    def permits(self,url):
        return self._capability is _CAP and self.base_url==url


def attest(api,*,space_id,revision,timeout=10):
    if getattr(api,'endpoint',None)!='https://huggingface.co':
        raise ValueError('canonical HF API endpoint required')
    if not re.fullmatch(r'YUGOROU/h3-live-relay-[a-z0-9-]{8,64}',space_id):
        raise ValueError('explicit owner relay namespace required')
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('immutable deployment required')
    info=api.space_info(space_id,files_metadata=False)
    if info.id!=space_id or info.private is not True or info.sha!=revision:
        raise ValueError('private immutable relay differs')
    host=getattr(info,'host',None)
    url=host if type(host) is str and host.startswith('https://') else 'https://'+str(host)
    url=url.rstrip('/');u=urlsplit(url)
    if (u.scheme!='https' or not re.fullmatch('[a-z0-9-]+\\.hf\\.space',u.hostname or '')
            or u.path or u.port is not None or u.username is not None or u.password is not None or u.query or u.fragment):
        raise ValueError('provider-declared private HF HTTPS host required')
    c=http.client.HTTPSConnection(u.hostname,timeout=timeout)
    try:
        c.request('GET','/health',headers={'Accept-Encoding':'identity','Cache-Control':'no-store'})
        r=c.getresponse()
        if r.status not in (401,403,404):raise ValueError('anonymous relay access was not denied')
    finally:c.close()
    return PrivateEndpoint(url,space_id,revision,_CAP)
