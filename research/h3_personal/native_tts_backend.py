"""Immutable native backend admission; no readiness I/O or local preparation.

prepare_direct_tts consumes spec['live_direct_session']; readiness and client
closure remain the run_session owner's responsibility. Secrets are consumed
only from the child environment and never added to the immutable config.
"""
from concurrent.futures import Future
import hashlib
import json
import math
import os
import re
import time
from urllib.parse import urlsplit

from .tts_preroll import PCMPrerollPolicy
from .voxcpm_endpoint_client import RemoteVoxCPM
from .voxcpm_endpoint_protocol import PINS, CANCEL_PROTOCOL


_SCHEMA = 'h3-native-voxcpm-backend-v1'
CANCEL_SCHEMA = 'h3-native-voxcpm-cancel-v4'
_KEYS = {'schema', 'url', 'session', 'boot', 'expires_at', 'pins', 'preroll'}
_PREROLL = {'sample_rate', 'reserve_samples', 'max_reserve_samples', 'max_wait_seconds'}
_HOST = r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+endpoints\.huggingface\.cloud'


def validate_native_config(config):
    """Return a detached validated manifest; expiry freshness is runtime-only."""
    try:
        if type(config) is dict and config.get('schema')=='h3-native-voxcpm-network-v2':
            from .network_voxcpm import validate_config
            return validate_config(config)
        cancel = type(config) is dict and config.get('schema') == CANCEL_SCHEMA
        long_session = type(config) is dict and 'session_seconds' in config
        if type(config) is not dict or set(config) != (_KEYS | ({'cancel_protocol'} if cancel else set())
                | ({'session_seconds'} if long_session else set())):
            raise ValueError()
        if long_session and (type(config['session_seconds']) is not int
                or config['session_seconds'] != 10800
                or config.get('schema') not in (_SCHEMA, CANCEL_SCHEMA)):
            raise ValueError()
        if cancel and config['cancel_protocol'] != CANCEL_PROTOCOL:
            raise ValueError()
        if type(config['schema']) is not str or config['schema'] not in (_SCHEMA,'h3-native-voxcpm-rtc-v3',CANCEL_SCHEMA):
            raise ValueError()
        if type(config['url']) is not str:
            raise ValueError()
        raw = config['url']
        url = urlsplit(raw)
        if (url.scheme != 'https' or url.path or url.query or url.fragment
                or url.username is not None or url.password is not None
                or url.port is not None or not re.fullmatch(_HOST, url.hostname or '')
                or raw != 'https://' + url.netloc):
            raise ValueError()
        for field in ('session', 'boot'):
            if type(config[field]) is not str or not re.fullmatch('[0-9a-f]{32}', config[field]):
                raise ValueError()
        expiry = config['expires_at']
        if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry <= 0:
            raise ValueError()
        pins = config['pins']
        if type(pins) is not dict or set(pins) != set(PINS):
            raise ValueError()
        if any(type(pins[k]) is not type(v) or pins[k] != v for k, v in PINS.items()):
            raise ValueError()
        preroll = config['preroll']
        if type(preroll) is not dict or set(preroll) != _PREROLL:
            raise ValueError()
        PCMPrerollPolicy(**preroll)
        return {**config, 'pins': dict(pins), 'preroll': dict(preroll)}
    except Exception:
        raise ValueError('invalid native TTS configuration') from None


def config_sha256(config):
    """Bind the validated immutable selection and preroll, never credentials."""
    return hashlib.sha256(json.dumps(validate_native_config(config),sort_keys=True,
        separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def config_preroll(config):
    """Build the frozen policy from a validated detached manifest."""
    return PCMPrerollPolicy(**validate_native_config(config)['preroll'])


def validate_native_lease(config, *, min_remaining=0):
    """Runtime/pre-submit freshness check; structural validation stays clock-free."""
    value=validate_native_config(config)
    now=time.time()
    if not now+min_remaining<value['expires_at']<=now+value.get('session_seconds', 1200):
        raise ValueError('native TTS finite lease expired or insufficient')
    return value


def native_job_secrets(spec):
    """Return child secret fields without writing or logging them (parent only)."""
    row=spec.get('live_direct_session',{})
    if row.get('tts_mode','complete')!='native_voxcpm':
        if 'native_tts' in row:
            raise ValueError('native config forbidden for legacy mode')
        return {}
    config=validate_native_config(row.get('native_tts'))
    names=(('H3_NATIVE_TTS_KEY',) if config['schema']=='h3-native-voxcpm-network-v2'
           else ('H3_NATIVE_TTS_HF_TOKEN','H3_NATIVE_TTS_KEY'))
    values={name:os.environ.get(name) for name in names}
    if any(type(value) is not str or not 1<=len(value)<=8192 or '\r' in value or '\n' in value
           for value in values.values()):
        raise ValueError('native TTS secrets required')
    return values


def prepare_direct_tts(root, spec, submit_local):
    """Select exactly one explicit backend, returning a preparation Future.

    Native admission errors raise synchronously, with no local fallback. The
    returned native Future is completed without readiness/network operations.
    """
    if type(spec) is not dict or type(spec.get('live_direct_session')) is not dict:
        raise ValueError('explicit direct session required')
    row = spec['live_direct_session']
    mode = row.get('tts_mode', 'complete')
    if type(mode) is not str or mode not in ('complete', 'sentence_incremental', 'native_voxcpm'):
        raise ValueError('explicit TTS mode required')
    if (type(row.get('native_tts_cancel',False)) is not bool
            or (row.get('native_tts_cancel',False) and mode!='native_voxcpm')):
        raise ValueError('native cancel is native backend only')
    if 'native_tts' in spec or 'tts_mode' in spec:
        raise ValueError('backend belongs only to direct session')
    if mode != 'native_voxcpm':
        if 'native_tts' in row:
            raise ValueError('native config forbidden for legacy TTS')
        return submit_local()
    config = validate_native_config(row.get('native_tts'))
    if (type(row.get('native_tts_cancel', False)) is not bool
            or row.get('native_tts_cancel', False) != (config['schema'] == CANCEL_SCHEMA)):
        raise ValueError('explicit native cancel mode/config mismatch')
    if (type(row.get('session_seconds', 1200)) is not int
            or row.get('session_seconds', 1200) != config.get('session_seconds', 1200)):
        raise ValueError('native TTS session duration/config mismatch')
    # Pop both before checking either, including missing-secret failure paths.
    token = os.environ.pop('H3_NATIVE_TTS_HF_TOKEN', None)
    key = os.environ.pop('H3_NATIVE_TTS_KEY', None)
    network=config['schema']=='h3-native-voxcpm-network-v2'
    if not key or (not network and not token):
        raise ValueError('native TTS secrets required')
    validate_native_lease(config)
    if network:
        from .network_voxcpm import RemoteNetworkVox
        client=RemoteNetworkVox(config,key=key)
    elif config['schema']=='h3-native-voxcpm-rtc-v3':
        from pathlib import Path
        from .webrtc_sidecar import bootstrap
        from .voxcpm_rtc_factory import VoxRTCFactory
        from .webrtc_upstream_adapter import RemoteVoxCPMRTC
        workspace=Path(root)/'upstream-rtc';workspace.mkdir(mode=0o700)
        python=bootstrap(Path(root),workspace,time.monotonic()+min(120,config['expires_at']-time.time()))
        factory=VoxRTCFactory(python)
        def open_stream(text,epoch,deadline):
            return factory.open_stream(endpoint=config['url'],token=token,key=key,session=config['session'],
                boot=config['boot'],expires_at=config['expires_at'],text=text,epoch=epoch,deadline=deadline)
        client=RemoteVoxCPMRTC(config['url'],open_stream=open_stream,token=token,key=key,
            session=config['session'],boot=config['boot'],expires_at=config['expires_at'])
    else:
        client = RemoteVoxCPM(config['url'], token=token, key=key,
                             session=config['session'], boot=config['boot'],
                             expires_at=config['expires_at'],
                             **({'session_seconds':10800} if config.get('session_seconds') == 10800 else {}),
                             **({'cancel_protocol':CANCEL_PROTOCOL} if config['schema']==CANCEL_SCHEMA else {}))
    future = Future()
    future.set_result(client)
    return future
