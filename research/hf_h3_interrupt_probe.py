"""Device-free native interruption probe. Saved PCM, never a fresh conversation.

Seven original 124/22 windows in one warm H3 owner. An unfinished first source is
interrupted during its backend call; a second complete copy of the approved PCM
finishes naturally, then one final listening window flushes the same decoder.
No TTS/ASR/Pi/device/RTC deployment or presentation acceptance is implied.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import wave

SCHEMA = 'h3-native-interruption-probe-v1'
SECTION = dict(schema=SCHEMA, profile='reference1024', windows=7,
               source='approved-locked-pcm', first_prefix_samples=204128,
               second_source='full-original-pcm', devices=False)
REPORT = 'interrupt-probe/report.json'


def validate_spec(spec):
    from research.hf_h3_streaming_trial import validate_spec as previous
    previous(spec)
    selected = spec.get('interrupt_probe')
    if (type(selected) is not dict or selected != SECTION
            or any(type(selected[k]) is not type(v) for k, v in SECTION.items())):
        raise ValueError('exact bounded native interruption probe required')
    if spec.get('late_listening_diagnostic') is not None:
        raise ValueError('no combined diagnostic')
    if spec['config']['timeout_seconds'] != 1200:
        raise ValueError('original finite 1200s Job required')
    return spec['interrupt_probe']


def _source(driver, pcm, epoch, *, sealed):
    from research.h3_personal.streaming_pcm import PCMIngress, PCMStreamBinding, PCMChunk
    binding = PCMStreamBinding(driver.binding.session, f'probe-speech-{epoch}', epoch,
                               driver.binding.profile_sha256)
    stream = PCMIngress(binding, max_bytes=2*577920, max_samples=577920, timeout=300)
    for offset in range(0, len(pcm), 64000):
        stream.append(PCMChunk(binding, offset//64000, offset//2, pcm[offset:offset+64000]))
    if sealed:
        stream.seal(total_samples=len(pcm)//2, sha256=hashlib.sha256(pcm).hexdigest())
    driver.attach(stream)
    return stream


def run_probe(root, *, pcm, binding, context, backend, timeout=180):
    """CPU-testable native scheduling seam; backend is one already-warm owner.

    Dispositions below are headless quarantine accounting, NOT sink ACKs. The
    old-tail/listening packet is deliberately quarantined BEFORE admitting new
    speech. This proves no old PCM is selected by this probe after invalidation,
    not uninterrupted playout or audible mute latency.
    """
    from research.h3_personal.live_conversation import LiveConversation
    if type(pcm) is not bytes or len(pcm) != 577920*2:
        raise ValueError('full approved 12.04s PCM required')
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 0 < timeout <= 300):
        raise ValueError('finite probe duration required')
    deadline = time.monotonic()+timeout
    timings = {}; records = []; interrupts = []; invalidated = threading.Event()
    selected_new_digest = hashlib.sha256(); selected_new_samples = [0]
    with LiveConversation(root, binding=binding, context=context, max_idle_windows=2,
                          max_turns=2, timeout=timeout) as driver:
        def step(*, interrupt=False):
            prepared = driver.prepare()
            if prepared is None:
                raise RuntimeError('unexpected unavailable saved PCM')
            index = prepared.manifest['streaming_audio']['sequence']
            trigger = None; errors = []; done = threading.Event(); entered = threading.Event()
            if interrupt:
                def request():
                    try:
                        entered.wait(timeout=max(.001, deadline-time.monotonic()))
                        time.sleep(.02)  # finite injection after parent backend entry
                        timings['interrupt_request'] = time.monotonic()
                        # Invalidate selection before requesting native retirement.
                        invalidated.set()
                        timings['local_invalidation'] = time.monotonic()
                        driver.request_speech_interrupt(1)
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        done.set()
                trigger = threading.Thread(target=request, name='h3-probe-turn-interrupt')
                trigger.start()
            begin = time.monotonic()
            entered.set()
            try:
                result = backend(prepared.manifest)
                backend_returned = time.monotonic()
            finally:
                if trigger is not None:
                    trigger.join(timeout=max(.001, min(2., deadline-time.monotonic())))
                    if trigger.is_alive():
                        raise RuntimeError('interrupt injector did not stop')
            if errors:
                raise errors[0]
            if interrupt and (not done.is_set() or timings['interrupt_request'] >= backend_returned):
                raise RuntimeError('interrupt was not injected during backend call')
            packet = driver.complete_native(prepared, result)
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError('finite probe deadline')
            regions = packet.envelope.regions
            old = any(region.source_epoch == 1 for region in regions)
            discard = invalidated.is_set() and (old or interrupt)
            for region in regions:
                if region.source_epoch == 2:
                    if discard:
                        raise ValueError('new speech cannot lose its first mixed-tail packet')
                    lo = (region.sample_start-packet.envelope.sample_start)*2
                    hi = (region.sample_stop-packet.envelope.sample_start)*2
                    selected_new_digest.update(packet.pcm[lo:hi])
                    selected_new_samples[0] += region.sample_stop-region.sample_start
            records.append(dict(index=index, role=prepared.manifest['streaming_audio']['role'],
                manifest=prepared.manifest['streaming_audio'],
                committed_frame=result['committed_frame'], emitted_frame=result['emitted_frame'],
                native_seconds=now-begin,
                video_sha256=hashlib.sha256(packet.video).hexdigest(),
                pcm_sha256=hashlib.sha256(packet.pcm).hexdigest(),
                source_epochs=sorted({r.source_epoch for r in regions if r.kind == 'speech'}),
                headless_disposition='discarded-interrupted' if discard else 'selected',
                presentation_verified=False))
            if interrupt:
                timings['old_forward_completed'] = now
                marker = driver.retire_interrupted_speech()
                timings['native_reentry_ready'] = time.monotonic()
                if marker is None:
                    raise ValueError('first source falsely completed instead of interrupted')
                interrupts.append(marker)
            return packet
        step()  # initial listening; one conversation-level persistent decoder
        first = _source(driver, pcm[:SECTION['first_prefix_samples']*2], 1, sealed=False)
        step(interrupt=True)
        if first.snapshot()['state'] == 'sealed':
            raise ValueError('interrupted source must remain genuinely unsealed')
        step()  # real listening window quarantines old decoder tail, not new speech
        _source(driver, pcm, 2, sealed=True)
        while driver.inspect().status == 'ready':
            step()
        if driver.inspect().status != 'end':
            raise RuntimeError('second source did not reach genuine sealed end')
        driver.retire_speech(); driver.request_stop(); step()
        state = driver.control.snapshot()
        if (len(records) != SECTION['windows'] or state['phase'] != 'closed'
                or state['committed_frame'] != 736 or state['emitted_frame'] != 736
                or state['interrupted_turns'] != 1 or state['completed_turns'] != 1
                or sum('previous_speech_interrupted' in row['manifest'] for row in records) != 1
                or any(1 in row['source_epochs'] and row['headless_disposition'] == 'selected'
                       for row in records[1:])
                or selected_new_samples[0] != len(pcm)//2
                or selected_new_digest.hexdigest() != hashlib.sha256(pcm).hexdigest()):
            raise ValueError('native interruption/reentry accounting differs')
        return dict(schema=SCHEMA, complete=True, windows=records, admission=state,
                    interruption_markers=interrupts, timings=timings, clock='local_monotonic',
                    local_invalidation_seconds=timings['local_invalidation']-timings['interrupt_request'],
                    safe_boundary_seconds=timings['native_reentry_ready']-timings['interrupt_request'],
                    new_live_input=False, devices_used=False, discord_used=False,
                    saved_pcm_model='legacy-Irodori-approved-fixture',
                    source_media_uploaded=False, current_vox_route_verified=False,
                    tts_streaming=False, transport_verified=False, playback_verified=False,
                    headless_selection_only=True, new_source_selected_samples=selected_new_samples[0],
                    new_source_selected_sha256=selected_new_digest.hexdigest(),
                    new_source_pcm_exact=True)


def run_session(root, paths, spec, deployment, deadline):
    from huggingface_hub import HfApi, CommitOperationAdd
    from research.h3_personal.accepted_profiles import choose_profile
    from research.h3_personal.conversation_conditioning import prompt_sha, role_profile_digest
    from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
    from research.h3_personal.live_native_backend import LiveNativeBackend
    from research.h3_personal.paired_av import LiveAVBinding
    from research.h3_personal.ref2va_runtime import sha
    from research.h3_personal.streaming_windows import (
        ResamplingContext, NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256)
    validate_spec(spec)
    root = Path(root); paths = {k: Path(v) for k, v in paths.items()}
    api = HfApi(token=os.environ['HF_TOKEN'])
    if api.repo_info(deployment, repo_type='dataset').private is not True:
        raise ValueError('private output required')
    speech = paths['speech']/spec['locked_speech']['path']
    if sha(speech) != spec['locked_speech']['sha256']:
        raise ValueError('approved saved speech changed')
    with wave.open(str(speech), 'rb') as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()) != (1, 2, 48000, 577920):
            raise ValueError('original PCM geometry required')
        pcm = handle.readframes(577920)
    reference = paths['reference']/'production-inputs/references/secondary.jpeg'
    profile = choose_profile(ordered_source_sha256=(sha(reference),), prompt_sha256=prompt_sha('speech'))
    if profile.name != 'reference1024':
        raise ValueError('reference1024; no fallback')
    identity = paths['personal']/spec['identity_prefix']
    output = root.with_name(root.name+'-interrupt-probe')
    runtime = LiveVoiceRuntime(model=paths['base'], model_manifest=root/'model-manifest.json',
        identity=identity/'adapter_model.safetensors', identity_config=identity/'adapter_config.json',
        smd=paths['smd']/'lora/adapter_model.safetensors', reference=reference,
        archive=root/'runtime/FastVideo.tar.gz', output=output, continuation=True,
        quality_profile=profile, conversation_roles=True)
    backend = None
    started = time.monotonic()
    try:
        runtime.start()
        if deadline-time.monotonic() < 120:
            raise TimeoutError('insufficient remaining native probe lifetime')
        backend = LiveNativeBackend(runtime, output/'native')
        binding = LiveAVBinding('native-interruption-probe', role_profile_digest(profile, 'speech'),
                                1, max_segments=8, max_total_frames=838)
        report = run_probe(output/'conversation', pcm=pcm, binding=binding,
            context=ResamplingContext('native-48k32k-prefix128', NATIVE_TORCH,
                                      NATIVE_TORCHAUDIO, REFERENCE_SHA256),
            backend=backend, timeout=min(300, deadline-time.monotonic()-45))
        report.update(native_backend_connected=True, both_rank_evidence=True,
                      native_receipts=list(backend.receipts), startup_and_probe_seconds=time.monotonic()-started,
                      source_wav_sha256=sha(speech), profile=profile.name)
        backend.retire()  # acknowledged final decoder release; no model reload
        payload = (json.dumps(report, sort_keys=True)+'\n').encode()
        if len(payload) > 1<<20 or time.monotonic() >= deadline:
            raise ValueError('bounded report/deadline required')
        api.create_commit(deployment, repo_type='dataset', operations=[
            CommitOperationAdd(path_in_repo=REPORT, path_or_fileobj=payload)],
            commit_message='Private saved-PCM native interruption boundary; no devices or conversation claim')
        return report
    finally:
        runtime.close()
