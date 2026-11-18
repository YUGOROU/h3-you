"""Native Vox wire over certificate-pinned TLS/TCP in an HF Network Group.

No TURN, HTTP media, codec, reconnect-after-request, or local-model fallback.
Provider/group placement must be independently attested by the resource owner.
"""
import hashlib
import hmac
import json
import math
import os
import re
import socket
import ssl
import struct
import threading
import time

from .voxcpm_endpoint_protocol import PINS,decode,loads

SCHEMA='h3-native-voxcpm-network-v2'
CONNECTION={'group','alias','port','certificate_sha256','server_job_id'}

def validate_config(config):
    if type(config) is not dict or set(config)!={'schema','connection','session','boot','expires_at','pins','preroll'} or config['schema']!=SCHEMA:
        raise ValueError('immutable network native configuration required')
    c=config['connection']
    if (type(c) is not dict or set(c)!=CONNECTION
            or any(type(c[k]) is not str for k in CONNECTION-{'port'})
            or not re.fullmatch('[a-z0-9](?:[a-z0-9-]{0,44}[a-z0-9])?',c['group'])
            or not re.fullmatch('[a-z0-9](?:[a-z0-9-]{0,32}[a-z0-9])?',c['alias'])
            or type(c['port']) is not int or not 1024<=c['port']<=65535
            or not re.fullmatch('[a-f0-9]{64}',c['certificate_sha256'])
            or not re.fullmatch('[a-f0-9]{24}',c['server_job_id'])):
        raise ValueError('finite network group connection required')
    for name in ('session','boot'):
        if type(config[name]) is not str or not re.fullmatch('[a-f0-9]{32}',config[name]):raise ValueError('native lease identity')
    expiry=config['expires_at'];pins=config['pins'];preroll=config['preroll']
    if type(expiry) not in (int,float) or not math.isfinite(expiry) or expiry<=0:raise ValueError('native expiry')
    if type(pins) is not dict or set(pins)!=set(PINS) or any(type(pins[k]) is not type(v) or pins[k]!=v for k,v in PINS.items()):
        raise ValueError('native pins')
    if type(preroll) is not dict or set(preroll)!={'sample_rate','reserve_samples','max_reserve_samples','max_wait_seconds'}:
        raise ValueError('native preroll')
    from .tts_preroll import PCMPrerollPolicy
    PCMPrerollPolicy(**preroll)
    return dict(config,pins=dict(pins),preroll=dict(preroll),connection=dict(c))

def address(config,environ=None):
    c=validate_config(config)['connection'];env=os.environ if environ is None else environ
    prefix=env.get('HF_NETWORK_GROUP_PREFIX')
    if (type(prefix) is not str or not prefix or len(prefix)>220
            or not re.fullmatch('[a-z0-9.-]+',prefix)):
        raise ValueError('provider network prefix required; no host fallback')
    host=prefix+c['alias']
    if len(host)>253 or any(not label or len(label)>63 or label.startswith('-') or label.endswith('-') for label in host.split('.')):
        raise ValueError('invalid provider network hostname')
    return host,c['port']

def read_exact(sock,n,deadline):
    out=bytearray()
    while len(out)<n:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('network native deadline')
        sock.settimeout(remaining);body=sock.recv(n-len(out))
        if not body:raise EOFError('network native EOF')
        out.extend(body)
    return bytes(out)

def receive_json(sock,deadline):
    size=struct.unpack('!I',read_exact(sock,4,deadline))[0]
    if not 0<size<=8192:raise ValueError('control byte bound')
    return loads(read_exact(sock,size,deadline))

def send_json(sock,value,deadline):
    body=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
    if not 0<len(body)<=8192:raise ValueError('control byte bound')
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('network native deadline')
    sock.settimeout(remaining);sock.sendall(struct.pack('!I',len(body))+body)

class RemoteNetworkVox:
    def __init__(self,config,*,key,connector=socket.create_connection,environ=None):
        self.config=validate_config(config)
        if type(key) is not str or not 32<=len(key)<=8192 or any(ord(c)<33 or ord(c)>126 for c in key):raise ValueError('native owner secret required')
        self._key=key;self._connector=connector;self._environ=environ
        self._lock=threading.Lock();self._socket_lock=threading.Lock()
        self._active_socket=None;self._closed=False;self._ready=False;self._epoch=0

    def _remaining(self,deadline):
        value=min(deadline-time.monotonic(),self.config['expires_at']-time.time())
        if self._closed or value<=0:raise TimeoutError('network native unavailable')
        return value

    def close(self):
        with self._socket_lock:self._closed=True;sock=self._active_socket
        if sock is not None:
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            sock.close()

    def _connect(self,deadline):
        raw=sock=None
        try:
            self._remaining(deadline)
            host,port=address(self.config,self._environ)
            raw=self._connector((host,port),timeout=min(1.0,self._remaining(deadline)))
            with self._socket_lock:
                self._remaining(deadline);self._active_socket=raw
            # CA/hostname validation is intentionally replaced by an immutable
            # exact DER fingerprint. Verify it BEFORE any owner secret/text.
            context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname=False;context.verify_mode=ssl.CERT_NONE
            context.minimum_version=ssl.TLSVersion.TLSv1_3
            sock=context.wrap_socket(raw,server_hostname=host,do_handshake_on_connect=False,suppress_ragged_eofs=False)
            with self._socket_lock:
                self._remaining(deadline);self._active_socket=sock
            sock.settimeout(self._remaining(deadline));sock.do_handshake()
            fingerprint=hashlib.sha256(sock.getpeercert(binary_form=True)).hexdigest()
            if not hmac.compare_digest(fingerprint,self.config['connection']['certificate_sha256']):raise ValueError('TLS identity mismatch')
            self._remaining(deadline);return sock
        except BaseException:
            if sock is not None:sock.close()
            elif raw is not None:raw.close()
            raise

    def _request(self,sock,kind,epoch,text,deadline):
        self._remaining(deadline)
        send_json(sock,dict(schema='h3-network-voxcpm-rpc-v1',kind=kind,key=self._key,
            session=self.config['session'],boot=self.config['boot'],epoch=epoch,text=text),deadline)

    def read(self,kind,deadline):
        if not self._lock.acquire(False):raise RuntimeError('busy')
        sock=None
        try:
            if kind!='tts_ready' or self._ready:raise ValueError('one native READY required')
            deadline=min(deadline,time.monotonic()+30)
            sock=self._connect(deadline);self._request(sock,'ready',0,None,deadline)
            row=receive_json(sock,deadline)
            expected=dict(PINS,session=self.config['session'],boot=self.config['boot'],expires_at=self.config['expires_at'],epoch=0)
            if (type(row) is not dict or row!=expected or any(type(row[k]) is not type(v) for k,v in expected.items())):
                raise ValueError('network native READY mismatch')
            self._remaining(deadline);self._ready=True;return row
        except Exception:
            self.close();raise RuntimeError('network native READY failed') from None
        finally:
            if sock is not None:sock.close()
            with self._socket_lock:self._active_socket=None
            self._lock.release()

    def evidence(self,deadline):
        if not self._lock.acquire(False):raise RuntimeError('busy')
        sock=None
        try:
            if not self._ready:raise ValueError('READY required')
            deadline=min(deadline,time.monotonic()+15)
            sock=self._connect(deadline);self._request(sock,'evidence',0,None,deadline)
            return receive_json(sock,deadline)
        except Exception:
            self.close();raise RuntimeError('network native evidence failed') from None
        finally:
            if sock is not None:sock.close()
            with self._socket_lock:self._active_socket=None
            self._lock.release()

    def stream(self,text,epoch,deadline):
        if not self._lock.acquire(False):raise RuntimeError('busy')
        sock=None;complete=False
        try:
            if (not self._ready or type(text) is not str or not text.strip() or len(text)>160
                    or type(epoch) is not int or epoch<=self._epoch
                    or type(deadline) not in (int,float) or not math.isfinite(deadline)):
                raise ValueError('network native stream request')
            deadline=min(deadline,time.monotonic()+120);self._remaining(deadline)
            self._epoch=epoch  # Reserve before uncertain connect/request; never retry.
            sock=self._connect(deadline);self._request(sock,'stream',epoch,text,deadline)
            def read(size):
                sock.settimeout(self._remaining(deadline));body=sock.recv(size)
                self._remaining(deadline);return body
            yield from decode(read,dict(session=self.config['session'],boot=self.config['boot'],
                expires_at=self.config['expires_at'],epoch=epoch))
            complete=True
        except Exception:
            raise RuntimeError('network native stream failed; no fallback') from None
        finally:
            if not complete:self.close()
            if sock is not None:sock.close()
            with self._socket_lock:self._active_socket=None
            self._lock.release()
