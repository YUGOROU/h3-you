"""Explicit finite HTTPS lease on the existing Vox L4 Endpoint only.

No Endpoint collection creation, H3/ASR startup, device operation or environment
credential injection. Caller owns the overall finite operation alarm. Old media
requests must be drained to native END before reusing this serial TTS service.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import secrets
import sys
import time
import uuid

from research.run_h3_native_coupled import (
    PRIVATE, GUARD, preflight, upload_verified, copy_runtime, deployment,
)
from research.run_h3_irodori_endpoint import control, pause_owned, persist
from research.run_h3_asr_endpoint import assert_deployment
from .native_tts_backend import validate_native_config
from .voxcpm_endpoint_client import RemoteVoxCPM
from .voxcpm_endpoint_protocol import PINS

NAME = 'h3-voxcpm-l4-a474fd64f740'
DEFAULT_PARENT = PRIVATE/'voxcpm-endpoint-rtc-acceptance-20261030-v1/lease'
DEFAULT_REVISION = '045834f1c38778035bbe30030fe8400d08b6038d'
DEFAULT_REPOSITORY = 'YUGOROU/h3-voxcpm-rtc-runtime-e5ae71c9119c'
LOCKS = ('.voxcpm-endpoint-owner.lock', '.irodori-endpoint-owner.lock',
         '.native-streaming-tts-owner.lock', '.network-tts-owner.lock')
FENCES = (GUARD.name, '.irodori-endpoint-unresolved.json',
          '.native-streaming-tts-unresolved.json', '.network-tts-unresolved.json')


def _credentials():
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    token = cached_token()
    return HfApi(token=token), token


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError('bounded nonsymlink parent receipt required')
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise ValueError('parent receipt object required')
    return value


def _parent(parent_campaign):
    """Review exact supplied lineage; never fall back to a different parent."""
    default = parent_campaign is None
    parent = Path(os.path.abspath(DEFAULT_PARENT if default else parent_campaign))
    if (not parent.is_relative_to(PRIVATE)
            or any(p.is_symlink() for p in (parent, *parent.parents))):
        raise ValueError('private reviewed parent required')
    intent = _read(parent/'intent.json')
    config = validate_native_config(_read(parent/'native-config.json'))
    if config['schema'] not in ('h3-native-voxcpm-rtc-v3', 'h3-native-voxcpm-backend-v1',
                                 'h3-native-voxcpm-cancel-v4'):
        raise ValueError('reviewed Endpoint backend required')
    terminal = _read(parent/'terminal.json')
    if (terminal.get('name') != NAME or terminal.get('state') != 'paused'
            or terminal.get('independent_confirmation') is not True):
        raise ValueError('independently paused parent required')
    model = intent.get('model', {})
    revision = model.get('revision')
    repo = model.get('repository')
    if (type(revision) is not str or len(revision) != 40
            or any(c not in '0123456789abcdef' for c in revision)
            or type(repo) is not str or not repo.startswith('YUGOROU/h3-voxcpm-')
            or model.get('task') != 'custom' or intent.get('cacheHttpResponses') is not False):
        raise ValueError('immutable owned parent runtime required')
    if default and (revision != DEFAULT_REVISION or repo != DEFAULT_REPOSITORY
                    or config['schema'] != 'h3-native-voxcpm-rtc-v3'):
        raise ValueError('default RTC lineage changed; explicit reviewed parent required')
    if config['schema'] == 'h3-native-voxcpm-rtc-v3':
        from research.prepare_h3_voxcpm_rtc import deployment as reviewed_deployment
        reviewed = reviewed_deployment(repo, revision, NAME, config['session'],
                                       config['boot'], config['expires_at'])
    else:
        reviewed = deployment(repo, revision, NAME, config['session'],
                              config['boot'], config['expires_at'])
        reviewed['model']['env']['H3_VOX_MEDIA_TRANSPORT'] = 'https-native-v1'
        if config['schema']=='h3-native-voxcpm-cancel-v4':
            reviewed['model']['env']['H3_VOX_CANCEL_PROTOCOL'] = config['cancel_protocol']
    duration_marker = model.get('env', {}).get('H3_VOX_SESSION_SECONDS')
    if duration_marker is not None:
        if (duration_marker != '10800'
                or config.get('session_seconds') != 10800
                or config['schema'] == 'h3-native-voxcpm-rtc-v3'):
            raise ValueError('reviewed HTTPS long-session marker required')
        reviewed['model']['env']['H3_VOX_SESSION_SECONDS'] = duration_marker
    elif 'session_seconds' in config:
        raise ValueError('native duration missing reviewed deployment marker')
    assert_deployment(intent, reviewed)
    return parent, intent


def _confirm_paused(token, path, intent, parent_intent):
    # A failed GET or uncertain PUT must not prevent the pause attempt.
    state = pause_owned(token, path)
    current = control(token, path)
    if state != 'paused' or current.get('status', {}).get('state') != 'paused':
        raise RuntimeError('independent Endpoint pause not confirmed')
    try:
        assert_deployment(current, intent)
        return True
    except ValueError:
        # If the PUT did not apply, the exact originally reviewed paused parent
        # is safe too. Unknown revisions/hardware remain fenced even if paused.
        assert_deployment(current, parent_intent)
        return False


@contextmanager
def lease(directory, *, min_remaining=420, parent_campaign=None, native_tts_cancel=False,
          session_seconds=1200):
    """Live API: one PUT/resume, native READY, yield config/token/key/root.

    parent_campaign=None binds the exact saved accepted RTC revision. Future
    calls must explicitly supply a reviewed, independently paused campaign.
    No retries of PUT/resume and no secret-bearing metadata are persisted.
    session_seconds=10800 explicitly selects the finite three-hour HTTPS owner;
    the default and the startup/readiness wait remain at their legacy bounds.
    """
    if type(session_seconds) is not int or session_seconds not in (1200, 10800):
        raise ValueError('explicit legacy or three-hour Vox lease required')
    if type(native_tts_cancel) is not bool:
        raise ValueError('explicit native cancel boolean required')
    from .voxcpm_endpoint_protocol import CANCEL_PROTOCOL
    if type(min_remaining) is not int or not 420 <= min_remaining <= 900:
        raise ValueError('bounded remaining lease required')
    root = Path(os.path.abspath(directory))
    if (root.exists() or not root.is_relative_to(PRIVATE)
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise ValueError('fresh private lease directory required')
    locks = []; owned = False; token = None
    path = '/v2/endpoint/YUGOROU/'+NAME
    try:
        for name in LOCKS:
            fd = os.open(PRIVATE/name, os.O_CREAT|os.O_WRONLY|os.O_NOFOLLOW, 0o600)
            locks.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in FENCES:
            if (PRIVATE/name).exists() or (PRIVATE/name).is_symlink():
                raise RuntimeError('unresolved prior resource')
        if any(k in os.environ for k in ('H3_NATIVE_TTS_KEY', 'H3_NATIVE_TTS_HF_TOKEN')):
            raise ValueError('preexisting native credentials')
        parent, parent_intent = _parent(parent_campaign)
        api, token = _credentials()
        spec, hardware = preflight(api, token)
        current = control(token, path)
        assert_deployment(current, parent_intent)
        if current.get('status', {}).get('state') != 'paused':
            raise ValueError('existing approved paused L4 Endpoint required')
        root.mkdir(mode=0o700)
        expires = int(time.time())+session_seconds
        startup_deadline = time.monotonic()+1200-min_remaining
        spec = dict(spec, expires_at=expires)
        spec.pop('session_seconds', None)
        if session_seconds == 10800:
            spec['session_seconds'] = session_seconds
        artifact = root/'deployment'
        copy_runtime(artifact, spec)  # owns mkdir; do not create artifact twice
        repo = 'YUGOROU/h3-voxcpm-https-runtime-'+uuid.uuid4().hex[:12]
        revision = upload_verified(api, artifact, repo)
        session = uuid.uuid4().hex; boot = uuid.uuid4().hex
        key = secrets.token_urlsafe(40)
        intent = deployment(repo, revision, NAME, session, boot, expires)
        intent['model']['env']['H3_VOX_MEDIA_TRANSPORT'] = 'https-native-v1'
        if session_seconds == 10800:
            intent['model']['env']['H3_VOX_SESSION_SECONDS'] = str(session_seconds)
        if native_tts_cancel:intent['model']['env']['H3_VOX_CANCEL_PROTOCOL'] = CANCEL_PROTOCOL
        persist(root/'intent.json', intent); persist(root/'hardware.json', hardware)
        persist(root/'parent.json', dict(directory=str(parent),
            repository=parent_intent['model']['repository'], revision=parent_intent['model']['revision']))
        if (time.time() >= expires-min_remaining-30
                or time.monotonic() >= startup_deadline-30):
            raise TimeoutError('insufficient lease before Endpoint update')
        persist(GUARD, dict(name=NAME, namespace='YUGOROU', directory=str(root), expires_at=expires))
        owned = True
        payload = dict(model={**intent['model'], 'secrets': {'HF_TOKEN': token, 'H3_TTS_KEY': key}},
                       compute=intent['compute'])
        control(token, path, payload, 'PUT')
        current = control(token, path); assert_deployment(current, intent)
        if (time.time() >= expires-min_remaining-30
                or time.monotonic() >= startup_deadline-30):
            raise TimeoutError('insufficient lease before Endpoint resume')
        control(token, path+'/resume', {}, 'POST')
        while (time.time() < expires-min_remaining
               and time.monotonic() < startup_deadline):
            current = control(token, path); assert_deployment(current, intent)
            state = current.get('status', {}).get('state')
            if state == 'running':
                break
            if state in ('failed', 'paused'):
                raise RuntimeError('HTTPS Endpoint failed to start')
            time.sleep(5)
        else:
            raise TimeoutError('HTTPS Endpoint readiness deadline')
        config = validate_native_config(dict(schema=(
            'h3-native-voxcpm-cancel-v4' if native_tts_cancel else 'h3-native-voxcpm-backend-v1'),
            url=current['status']['url'].rstrip('/'), session=session, boot=boot,
            expires_at=expires, pins=dict(PINS), preroll=dict(sample_rate=48000,
                reserve_samples=48000, max_reserve_samples=96000, max_wait_seconds=2.0),
            **({'cancel_protocol':CANCEL_PROTOCOL} if native_tts_cancel else {}),
            **({'session_seconds':10800} if session_seconds == 10800 else {})))
        ready_deadline = min(time.monotonic()+30, startup_deadline,
                             time.monotonic()+expires-time.time()-min_remaining)
        if ready_deadline <= time.monotonic():
            raise TimeoutError('insufficient native READY lease')
        client = RemoteVoxCPM(config['url'], token=token, key=key, session=session,
                             boot=boot, expires_at=expires,
                             **({'session_seconds':10800} if session_seconds == 10800 else {}),
                             **({'cancel_protocol':CANCEL_PROTOCOL} if native_tts_cancel else {}))
        try:
            ready = client.read('tts_ready', ready_deadline)
        finally:
            client.close()
        if time.time() >= expires-min_remaining:
            raise TimeoutError('insufficient lease after native READY')
        persist(root/'native-config.json', config); persist(root/'ready.json', ready)
        yield dict(config=config, token=token, key=key, root=root)
        if native_tts_cancel:
            proof_client=RemoteVoxCPM(config['url'],token=token,key=key,session=session,
                boot=boot,expires_at=expires,cancel_protocol=CANCEL_PROTOCOL,
                **({'session_seconds':10800} if session_seconds == 10800 else {}))
            try:
                proof=proof_client.evidence(min(time.monotonic()+5,
                    time.monotonic()+expires-time.time()))
                persist(root/'native-completion-evidence.json',proof)
            except Exception as error:
                persist(root/'native-evidence-unavailable.json',dict(error_type=type(error).__name__,
                    natural_eos_verified=False,cancel_verified=False))
            finally:
                proof_client.close()
    finally:
        primary = sys.exc_info()[1]
        try:
            if owned:
                try:
                    applied = _confirm_paused(token, path, intent, parent_intent)
                    persist(root/'terminal.json', dict(name=NAME, state='paused',
                        independent_confirmation=True, deployment_applied=applied))
                    guard = _read(GUARD)
                    if guard.get('directory') != str(root) or guard.get('name') != NAME:
                        raise RuntimeError('owned Endpoint fence mismatch')
                    GUARD.unlink()
                except BaseException as error:
                    if primary is not None:
                        primary.add_note('Endpoint cleanup unconfirmed; retained resource fence')
                    raise error
        finally:
            for fd in reversed(locks):
                os.close(fd)
