"""Finite direct Jobs AV entry worker; no mailbox, Space or media uploads.

JobAVServer contract (implemented separately): context manager accepting a RAM
config with session, expiresAt, producerKey, consumerKey; .producer is the
already-authenticated socket adapter consumed by live_relay_worker.run_worker.
HF authenticated port exposure is the parent's responsibility.
"""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import time


ANCHOR_POLICY = 'first-listening-tail-every3-v1'
ANCHOR_ENV = 'H3_IDLE_ANCHOR_DIAGNOSTIC'
from research.h3_personal.visual_ancestry_policy import ENV as CONVERSATION_ANCHOR_ENV, VALUE as CONVERSATION_ANCHOR_SCHEMA, CANDIDATE as CONVERSATION_ANCHOR_POLICY


TTS_MODE_ENV = 'H3_TTS_MODE'


def validate_tts_mode(tts_mode):
    if type(tts_mode) is not str or tts_mode not in ('complete', 'sentence_incremental', 'native_voxcpm'):
        raise ValueError('explicit complete or sentence_incremental TTS mode required')
    if TTS_MODE_ENV in os.environ:
        raise ValueError('ambient TTS mode forbidden; use immutable direct session')
    return tts_mode


def endurance_limits(five_minute_test=False, ten_minute_test=False, long_session=False):
    """Only the explicit endurance opt-in changes the accepted limits."""
    if type(five_minute_test) is not bool:
        raise ValueError('five_minute_test must be an explicit boolean')
    if type(ten_minute_test) is not bool:
        raise ValueError('ten_minute_test must be an explicit boolean')
    if type(long_session) is not bool:
        raise ValueError('long_session must be an explicit boolean')
    if sum((five_minute_test, ten_minute_test, long_session)) > 1:
        raise ValueError('endurance modes are mutually exclusive')
    if long_session:
        return (2542, 259306, 10800)
    if ten_minute_test:
        return (142, 14506, 1200)
    return (70, 7200, 1200) if five_minute_test else (16, 1654, 1200)


def materialize_ordered_references(root, identity, expected_sha256, destination):
    """Copy already-authorized staged inputs into private owner-local storage.

    No fetch/upload/transform. The parent owns permission BEFORE staging. Native
    paths live outside the code tree and survive until this session closes.
    """
    import stat
    from research.h3_personal.ordered_references import (validate_content_identity,
        content_identity_sha256, verify_local_files, MAX_FILE_BYTES)
    identity=validate_content_identity(identity)
    if content_identity_sha256(identity)!=expected_sha256:
        raise ValueError('staged ordered identity mismatch')
    destination=Path(destination)
    if (not destination.is_absolute() or destination.is_symlink()
            or not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('fresh empty private input directory required')
    os.chmod(destination,0o700)
    config=dict(schema=identity['schema'],references=[])
    created=[]
    try:
        for record in identity['references']:
            source=Path(root)/'ordered-inputs'/(record['role']+'.png')
            if any(p.is_symlink() for p in (source,*source.parents)):
                raise ValueError('staged input symlink prohibited')
            target=destination/(record['role']+'.png')
            fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as incoming:
                before=os.fstat(incoming.fileno())
                if not stat.S_ISREG(before.st_mode) or not 33<=before.st_size<=MAX_FILE_BYTES:
                    raise ValueError('bounded regular staged image required')
                with open(target,'xb',opener=lambda p,f:os.open(p,f,0o600)) as outgoing:
                    created.append(target);count=0
                    while True:
                        chunk=incoming.read(min(1024*1024,MAX_FILE_BYTES+1-count))
                        if not chunk:break
                        count+=len(chunk)
                        if count>MAX_FILE_BYTES:raise ValueError('staged image exceeds bound')
                        outgoing.write(chunk)
                after=os.fstat(incoming.fileno())
                fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                if fields(before)!=fields(after) or fields(before)!=fields(source.stat(follow_symlinks=False)):
                    raise ValueError('staged image changed during copy')
            config['references'].append(dict(record,path=str(target)))
        return verify_local_files(config,expected_content_sha256=expected_sha256,
                                  short_edge=identity['preprocessing']['short_edge'])
    except BaseException:
        for target in created:target.unlink(missing_ok=True)
        raise


def validate_spec(spec):
    """Pure pre-GPU check; lease is a duration, never submission wall time."""
    from research.h3_personal.fast_ref2va import config_for_layout
    from research.launch_h3_fast_ref2va import IDENTITY_PREFIX, REFERENCE_SHA
    if type(spec) is not dict:
        raise ValueError('direct spec required')
    row = spec.get('live_direct_session')
    keys = {'schema', 'session', 'port', 'max_windows', 'max_frames', 'lease_seconds'}
    if (type(row) is not dict or not keys <= set(row) <= keys | {'expected_asr', 'anchor_policy', 'five_minute_test', 'ten_minute_test', 'tts_mode', 'overlap_send', 'http_keepalive', 'continuous_av', 'native_tts', 'media_transport', 'ice_profile', 'asr_media_transport', 'turn_interruptions', 'visual_intents', 'attended_visual', 'stable_selfie', 'paired_cuts', 'native_tts_cancel', 'native_cancel_trigger', 'ordered_reference_identity', 'ordered_reference_sha256', 'long_session', 'session_seconds'}
            or row.get('schema') != 'h3-live-direct-session-v1'
            or type(row.get('session')) is not str
            or not re.fullmatch('[a-f0-9]{32}', row['session'])):
        raise ValueError('explicit immutable direct session required')
    ordered_keys={'ordered_reference_identity','ordered_reference_sha256'}
    if ordered_keys & set(spec):
        raise ValueError('ordered references belong only to immutable direct session')
    if ordered_keys & set(row):
        from research.h3_personal.ordered_references import validate_content_identity,content_identity_sha256
        if not ordered_keys <= set(row) or row.get('visual_intents') is not True:
            raise ValueError('ordered reference identity/hash require visual mode')
        identity=validate_content_identity(row['ordered_reference_identity'])
        if identity['preprocessing']['short_edge']!=1024 or row['ordered_reference_sha256']!=content_identity_sha256(identity):
            raise ValueError('ordered reference immutable identity mismatch')
    if 'asr_media_transport' in row and (row['asr_media_transport']!='webrtc-exact-v1' or 'expected_asr' not in row):
        raise ValueError('explicit RTC ASR source binding required')
    transport=row.get('media_transport','http-paired-v1')
    if type(transport) is not str or transport not in ('http-paired-v1','webrtc-rtp-v1'):
        raise ValueError('explicit H3 media transport required')
    if any(k in spec for k in ('media_transport','ice_profile')):
        raise ValueError('media selection belongs only to direct session')
    if transport=='webrtc-rtp-v1':
        if row.get('ice_profile')!='hf-cloudflare-turn-tls-v1' or row.get('continuous_av',False):
            raise ValueError('RTC requires fixed approved TURN profile and no HTTP media stream')
    elif 'ice_profile' in row:raise ValueError('ICE profile is RTC only')
    mode = validate_tts_mode(row.get('tts_mode', 'complete'))
    if type(row.get('turn_interruptions', False)) is not bool:
        raise ValueError('explicit turn interruption selection required')
    attended = row.get('attended_visual', False)
    stable_selfie = row.get('stable_selfie', False)
    if type(stable_selfie) is not bool or 'stable_selfie' in spec:
        raise ValueError('explicit immutable stable selfie selection required')
    if stable_selfie and (row.get('visual_intents') is not True
            or row.get('turn_interruptions') is not False or row.get('paired_cuts', False)
            or row.get('native_tts_cancel', False) or row.get('native_cancel_trigger') is not None
            or transport != 'http-paired-v1' or mode != 'native_voxcpm'
            or (attended is False and 'expected_asr' in row)
            or row.get('anchor_policy') != CONVERSATION_ANCHOR_POLICY):
        raise ValueError('stable selfie requires visual native HTTPS with exact conversation anchor and no cuts/cancel')
    long_session = row.get('long_session', False)
    if type(long_session) is not bool or 'long_session' in spec:
        raise ValueError('explicit immutable long session selection required')
    if ('session_seconds' in row and (not long_session
            or type(row['session_seconds']) is not int or row['session_seconds'] != 10800)):
        raise ValueError('session duration marker requires explicit three-hour session')
    if long_session and (attended is not True or mode != 'native_voxcpm'
            or transport != 'http-paired-v1' or row.get('turn_interruptions', False)
            or row.get('native_tts', {}).get('schema') != 'h3-native-voxcpm-backend-v1'):
        raise ValueError('long session requires attended native HTTPS without interruption')
    if type(attended) is not bool or 'attended_visual' in spec:
        raise ValueError('explicit attended visual selection required')
    if (type(row.get('visual_intents', False)) is not bool or 'visual_intents' in spec
            or (row.get('visual_intents', False) and
                (not (row.get('turn_interruptions') is True or attended or stable_selfie)
                 or ('anchor_policy' in row and not stable_selfie)))):
        raise ValueError('visual intents require bound interruption or attended latest-tail policy')
    # Attended half-duplex keeps latest-tail visual ancestry but no interrupt_turn
    # controls, so it must never silently acquire the headless interruption route.
    if attended and (row.get('visual_intents') is not True or row.get('turn_interruptions') is not False
            or transport != 'http-paired-v1' or mode != 'native_voxcpm'
            or row.get('paired_cuts', False) or row.get('native_tts_cancel', False)
            or row.get('native_cancel_trigger') is not None):
        raise ValueError('attended visual requires native HTTPS latest-tail without interruption controls')
    if (type(row.get('paired_cuts',False)) is not bool or 'paired_cuts' in spec
            or (row.get('paired_cuts',False) and row.get('visual_intents') is not True)):
        raise ValueError('paired cuts require immutable visual mode')
    if ('native_cancel_trigger' in spec or row.get('native_cancel_trigger') not in (None,'native-preroll-v1')
            or (row.get('native_cancel_trigger') is not None and row.get('native_tts_cancel') is not True)):
        raise ValueError('native trigger requires immutable native cancel mode')
    if (type(row.get('native_tts_cancel',False)) is not bool or 'native_tts_cancel' in spec
            or (row.get('native_tts_cancel',False) and
                (row.get('turn_interruptions') is not True or mode!='native_voxcpm'))):
        raise ValueError('native cancel requires immutable native interruption mode')
    if (row.get('native_tts',{}).get('schema')=='h3-native-voxcpm-cancel-v4') != row.get('native_tts_cancel',False):
        raise ValueError('native cancel config/mode mismatch')
    if 'turn_interruptions' in spec:
        raise ValueError('turn interruption belongs to immutable direct session')
    if row.get('turn_interruptions', False) and (transport != 'http-paired-v1'
            or mode != 'native_voxcpm' or row.get('continuous_av') is not True
            or row.get('native_tts', {}).get('schema') != (
                'h3-native-voxcpm-cancel-v4' if row.get('native_tts_cancel',False)
                else 'h3-native-voxcpm-backend-v1')):
        raise ValueError('interruptible route requires native HTTPS Vox and continuous paired AV')
    if 'native_tts' in spec:
        raise ValueError('native TTS belongs only to the direct session')
    if mode == 'native_voxcpm':
        from research.h3_personal.native_tts_backend import validate_native_config
        validate_native_config(row.get('native_tts'))
    elif 'native_tts' in row:
        raise ValueError('native TTS config forbidden for legacy mode')
    if type(row.get('overlap_send', False)) is not bool:
        raise ValueError('explicit overlap_send bool required')
    if 'overlap_send' in spec:
        raise ValueError('overlap_send belongs only to the direct session')
    if type(row.get('http_keepalive', False)) is not bool:
        raise ValueError('explicit http_keepalive bool required')
    if 'http_keepalive' in spec:
        raise ValueError('http_keepalive belongs only to the direct session')
    if type(row.get('continuous_av', False)) is not bool:
        raise ValueError('explicit continuous_av bool required')
    if 'continuous_av' in spec:
        raise ValueError('continuous_av belongs only to the direct session')
    if 'tts_mode' in spec:
        raise ValueError('TTS mode belongs only to the direct session')
    windows, frames, timeout = endurance_limits(row.get('five_minute_test', False), row.get('ten_minute_test', False), long_session)
    if row.get('five_minute_test', False) or row.get('ten_minute_test', False) or long_session:
        if (row['lease_seconds'] != timeout or type(row['lease_seconds']) is not int
                or any(type(spec.get(k)) is not dict
                       or type(spec[k].get('timeout_seconds')) is not int
                       or spec[k]['timeout_seconds'] != timeout for k in ('config', 'execution'))):
            raise ValueError('provider timeout and lease must match immutable session limits')
    for key, expected in (('port', 7860), ('max_windows', windows), ('max_frames', frames)):
        if type(row[key]) is not int or row[key] != expected:
            raise ValueError('fixed direct port/window/frame limits required')
    seconds = row['lease_seconds']
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= timeout:
        raise ValueError('finite boot-relative lease within selected limits required')
    if 'expected_asr' in row:
        from research.h3_personal.asr_source_contract import validate_expected_asr
        validate_expected_asr(row['expected_asr'])
    if 'anchor_policy' in row and (type(row['anchor_policy']) is not str
            or row['anchor_policy'] not in (ANCHOR_POLICY, CONVERSATION_ANCHOR_POLICY)):
        raise ValueError('exact experimental idle anchor policy required')
    if 'anchor_policy' in spec:
        raise ValueError('anchor policy belongs only to the direct session')
    expected = asdict(config_for_layout('portrait'))
    # JSON numeric equality alone accepts bool and float in place of integer flags.
    actual = spec.get('fast_config')
    if (type(actual) is not dict or set(actual) != set(expected)
            or any(type(actual[k]) is not type(v) or actual[k] != v for k, v in expected.items())):
        raise ValueError('original BF16 SMD4 portrait geometry required')
    if (spec.get('layout') != 'portrait' or spec.get('attention_mode') != 'fa4-packed'
            or spec.get('runtime_policy') != 'resident-visual-cached'
            or spec.get('identity_prefix') != IDENTITY_PREFIX
            or spec.get('reference_sha256') != REFERENCE_SHA):
        raise ValueError('accepted native reference1024 runtime required')
    for flag in ('regional_compile', 'warm_stage_diagnostic', 'incremental_transfer'):
        if flag in spec and spec[flag] is not False:
            raise ValueError('experimental runtime flags forbidden')
    for key in ('locked_speech', 'streaming_trial', 'conversation_scenario',
                'live_relay_session', 'face_crop_box'):
        if spec.get(key) is not None:
            raise ValueError('conflicting conversation mode')
    live = spec.get('live_voice_session')
    if (type(live) is not dict or set(live) != {'id', 'max_turns', 'fresh_input_required'}
            or live['id'] != row['session'] or type(live['max_turns']) is not int
            or live['max_turns'] != (1000 if long_session else 2) or live['fresh_input_required'] is not True):
        raise ValueError('fresh two-turn session binding required')
    return row


def run_session(root, paths, spec, deployment, tts_future, deadline):
    from research.h3_personal.visual_ancestry_policy import ENV as CONVERSATION_ANCHOR_ENV
    row = validate_spec(spec)
    if ANCHOR_ENV in os.environ or CONVERSATION_ANCHOR_ENV in os.environ:
        raise ValueError('preexisting idle anchor environment forbidden')
    if 'H3_ORDERED_REFERENCE_CONTEXT_FILE' in os.environ:
        raise ValueError('preexisting ordered reference context forbidden; select through immutable session')
    # Start the lease at entry/boot, not submission or READY (queue time is free).
    boot = time.monotonic()
    deadline = min(deadline, boot + row['lease_seconds'])
    expires = time.time() + max(0, deadline - boot)
    from research.h3_personal.job_av_server import JobAVServer
    from research.h3_personal.accepted_profiles import choose_profile
    from research.h3_personal.conversation_conditioning import prompt_sha, role_profile_digest
    from research.h3_personal.ref2va_runtime import sha
    from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
    from research.h3_personal.streaming_windows import ResamplingContext, NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256
    from research.h3_personal.live_relay_worker import run_worker
    from research.h3_personal.paired_av import LiveAVBinding
    native = row.get('tts_mode') == 'native_voxcpm'
    if native:
        from research.h3_personal.native_tts_backend import config_preroll, config_sha256
    else:
        from research.hf_h3_live_voice import close_tts_future
    runtime = None
    ordered_temporary = None
    tts = None
    rtc = None
    try:
        root = Path(root)
        reference = paths['reference'] / 'production-inputs/references/secondary.jpeg'
        ordered_options={}
        if 'ordered_reference_identity' in row:
            import tempfile
            ordered_temporary=tempfile.TemporaryDirectory(prefix='h3-ordered-inputs-')
            config=materialize_ordered_references(root,row['ordered_reference_identity'],
                row['ordered_reference_sha256'],Path(ordered_temporary.name).resolve())
            ordered_options['ordered_references']=config
        profile = choose_profile(ordered_source_sha256=(sha(reference),), prompt_sha256=prompt_sha('speech'))
        if profile.name != 'reference1024':
            raise ValueError('reference1024 required; no fallback')
        identity = paths['personal'] / spec['identity_prefix']
        output = root.with_name(root.name + '-live-direct')
        runtime = LiveVoiceRuntime(model=paths['base'], model_manifest=root/'model-manifest.json',
            identity=identity/'adapter_model.safetensors', identity_config=identity/'adapter_config.json',
            smd=paths['smd']/'lora/adapter_model.safetensors', reference=reference,
            archive=root/'runtime/FastVideo.tar.gz', output=output, continuation=True,
            quality_profile=profile, conversation_roles=True,
            **({'visual_intents':True} if row.get('visual_intents', False) else {}), **ordered_options)
    except BaseException:
        if ordered_temporary is not None:ordered_temporary.cleanup()
        if native:
            tts_future.add_done_callback(lambda future: future.result().close() if not future.cancelled() and future.exception() is None else None)
        else:
            close_tts_future(tts_future, None)
        raise
    from research.hf_h3_direct_conversation import (ANCHOR_POLICY, CONVERSATION_ANCHOR_ENV,
        CONVERSATION_ANCHOR_SCHEMA, CONVERSATION_ANCHOR_POLICY)
    # Explicit visual candidate experiment; live speech handoff is not proven.
    if row.get('anchor_policy') == ANCHOR_POLICY:
        os.environ[ANCHOR_ENV] = ANCHOR_POLICY
    elif row.get('anchor_policy') == CONVERSATION_ANCHOR_POLICY:
        os.environ[CONVERSATION_ANCHOR_ENV] = CONVERSATION_ANCHOR_SCHEMA
    try:
        runtime.start()
        tts = tts_future.result(timeout=max(.001, deadline-time.monotonic()))
        if native:
            remaining = row['native_tts']['expires_at'] - time.time()
            deadline = min(deadline, time.monotonic() + remaining)
            expires = min(expires, row['native_tts']['expires_at'])
            if deadline - time.monotonic() < 120:
                raise TimeoutError('insufficient native TTS lease after preparation')
        tts.read('tts_ready', deadline)
        # Preserve the ordinary READY startup safety margin.
        if deadline-time.monotonic() < 120:
            raise TimeoutError('insufficient direct conversation lease after preparation')
        binding = LiveAVBinding(row['session'], role_profile_digest(runtime.quality_profile, 'speech'), 1,
            max_segments=row['max_windows'], max_total_frames=row['max_frames'],
            **({'long_session': True} if row.get('long_session') else {}))
        if row.get('media_transport')=='webrtc-rtp-v1':
            from research.h3_personal.webrtc_sidecar import LiveRTCSidecar
            rtc=LiveRTCSidecar(root,binding,expires,os.environ['H3_DIRECT_CONSUMER_KEY'],
                              os.environ['HF_TOKEN'],deadline)
            rtc.__enter__()
            deadline=min(deadline,rtc.deadline)
            expires=min(expires,rtc.expires_at)
        from research.h3_personal.live_relay import RelayConfig
        config = RelayConfig(row['session'],expires,
            os.environ['H3_DIRECT_PRODUCER_KEY'],os.environ['H3_DIRECT_CONSUMER_KEY'],
            long_session=row.get('long_session', False))
        rtc_options={}
        if rtc is not None:
            from research.h3_personal.webrtc_job_gateway import RTCGateway
            rtc_options['rtc_gateway']=RTCGateway(rtc.port,config.consumer_key,binding.session,expires)
        with JobAVServer(config, port=row['port'],**rtc_options,
                         **({'keepalive': True} if row.get('http_keepalive', False) else {}),
                         **({'continuous_av': True} if row.get('continuous_av', False) else {})) as server:
            def on_ready():
                ready_options = {'tts_mode': 'sentence_incremental'} if row.get('tts_mode') == 'sentence_incremental' else {}
                if rtc is not None:
                    ready_options.update(media_transport='webrtc-rtp-v1',rtc_boot=rtc.boot,
                                         ice_profile='hf-cloudflare-turn-tls-v1')
                if native:
                    ready_options.update(tts_mode='native_voxcpm',
                        native_tts_config_sha256=config_sha256(row['native_tts']))
                if row.get('overlap_send', False):
                    ready_options['overlap_send'] = True
                if row.get('http_keepalive', False):
                    ready_options['http_keepalive'] = True
                if row.get('continuous_av', False):
                    ready_options['continuous_av'] = True
                if row.get('turn_interruptions', False):
                    ready_options['turn_interruptions'] = True
                if row.get('visual_intents', False):ready_options['visual_intents'] = True
                if row.get('attended_visual', False):ready_options['attended_visual'] = True
                if row.get('stable_selfie', False):
                    ready_options.update(stable_selfie=True, anchor_policy=CONVERSATION_ANCHOR_POLICY)
                if row.get('long_session', False):ready_options['long_session'] = True
                if row.get('paired_cuts', False):ready_options['paired_cuts'] = True
                if row.get('native_tts_cancel',False):ready_options['native_tts_cancel'] = True
                if row.get('native_cancel_trigger') is not None:
                    ready_options['native_cancel_trigger'] = row['native_cancel_trigger']
                if 'ordered_reference_sha256' in row:
                    ready_options['ordered_reference_sha256']=row['ordered_reference_sha256']
                print(json.dumps(dict(type='h3_direct_ready', status='ready', session=binding.session,
                    profile_sha256=binding.profile_sha256, epoch=binding.epoch,
                    max_segments=binding.max_segments, max_total_frames=binding.max_total_frames,
                    port=row['port'], expires_at=expires, **ready_options)), flush=True)
            context = ResamplingContext('native-48k32k-prefix128', NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256)
            worker_options = {'expected_asr': row['expected_asr']} if 'expected_asr' in row else {}
            if rtc is not None:
                worker_options.update(media_transport='webrtc-rtp-v1',media_adapter=rtc.media_adapter)
            if row.get('overlap_send', False):
                worker_options['overlap_send'] = True
            if row.get('turn_interruptions', False):
                worker_options['turn_interruptions'] = True
            if row.get('visual_intents', False):worker_options['visual_intents'] = True
            if row.get('attended_visual', False):worker_options['attended_visual'] = True
            if row.get('stable_selfie', False):worker_options['stable_selfie'] = True
            if row.get('long_session', False):
                worker_options.update(long_session=True, hold_listening=True)
            if row.get('paired_cuts', False):worker_options['paired_cuts'] = True
            if row.get('native_tts_cancel',False):worker_options['native_tts_cancel'] = True
            if native:
                worker_options.update(tts_mode='native_voxcpm',
                    tts_preroll=config_preroll(row['native_tts']))
            if row.get('tts_mode') == 'sentence_incremental':
                worker_options['tts_mode'] = 'sentence_incremental'
            if row.get('five_minute_test', False) or row.get('ten_minute_test', False):
                worker_options['hold_listening'] = True
            if row.get('ten_minute_test', False):
                worker_options['ten_minute_test'] = True
            if row.get('anchor_policy') == CONVERSATION_ANCHOR_POLICY:
                worker_options['visual_ancestry_policy'] = CONVERSATION_ANCHOR_POLICY
            result = run_worker(runtime, tts, server.producer, binding=binding, context=context,
                                root=output/'conversation', deadline=deadline, on_ready=on_ready,
                                **worker_options)
            if rtc is not None:
                fields=('source_complete','media_complete','complete','closing_tail_unverified',
                        'local_ipc_source_verified','remote_original_pcm_verified')
                print(json.dumps(dict(type='h3_rtc_delivery',session=binding.session,
                    **{k:result[k] for k in fields})),flush=True)
            return result
    finally:
        try:
            try:
                if rtc is not None:rtc.close()
            finally:
                runtime.close()
        finally:
            try:
                if native:
                    if tts is not None:
                        tts.close()
                    else:
                        tts_future.add_done_callback(lambda future: future.result().close() if not future.cancelled() and future.exception() is None else None)
                else:
                    close_tts_future(tts_future, tts)
            finally:
                os.environ.pop(ANCHOR_ENV, None)
                os.environ.pop(CONVERSATION_ANCHOR_ENV, None)
                if ordered_temporary is not None:ordered_temporary.cleanup()
