"""Opt-in worker hooks for the pinned FastVideo Ref2VA/SMD4 path.

No patching, model lifecycle, decode, or GPU acceptance is performed here.
Overlay order: append_conditions AFTER native fixed encodes and BEFORE native
condition shapes/.999 RF noise; make_layout instead of native _build_layout;
bind_audio after packed latents/layout assignment instead of external_audio's
locker; before_forward BEFORE the native interrupt break on EVERY rank;
hold_audio INSTEAD OF audio_scheduler.step; commit before target decode.
The existing external_audio clean target timestep hook must remain installed
(audio_timestep=1.0); native condition_audio_timestep is already 1.0.
Decoder/parent must consume context['suffix'], never emit overlap again.

Manifest (atomic JSON at ENV): mode=independent, or mode=latent-tail-v1,
session_id, index, wav, sha256, total_frames, window_frames=124,
overlap_frames=22, seed=314159, expected_windows (1..4), final (bool).
Optional source_samples/source_sample_rate describe the ORIGINAL decoded WAV;
optional cancelled (bool) is a cancellation flag, never a fallback selector.
Receipts contain scalars only, no source paths/media/tensors; schema is SCHEMA.
Torch/FastVideo imports are lazy; importing this module is stdlib-only.
"""
from contextlib import nullcontext
from functools import wraps
import hashlib
import json
import os
from pathlib import Path

from .fast_continuation import (Session, TorchOps, SamplerContract, WindowCancelled,
                                build_layout)
from .continuation_plan import audio_boundary, video_t

ENV = 'H3_SMD_CONTINUATION_WINDOW_FILE'
KEY = '_h3_continuation'
LAYOUT_KEY = 'minimax_h3_layout'
SCHEMA = 'h3-smd-continuation-stage-v1'


def _profile_range(batch, label):
    """Selected-window CPU annotation; no Torch import/call on default path."""
    context = batch.extra.get(KEY)
    manifest = context.get('manifest', {}) if context is not None else {}
    policy = manifest.get('trace_policy')
    if policy is None or policy.get('selected_window') != manifest.get('index'):
        return nullcontext()
    import torch
    return torch.profiler.record_function(label)


def forward_profile(batch, index):
    """Wrap only the native transformer call (zero-based SMD4 index)."""
    if type(index) is not int or not 0 <= index < 4:
        raise ValueError('SMD4 forward index required')
    return _profile_range(batch, ('h3.forward0', 'h3.forward1',
                                  'h3.forward2', 'h3.forward3')[index])


def _profile_stage(label):
    def decorate(function):
        @wraps(function)
        def wrapped(stage, batch, *args, **kwargs):
            with _profile_range(batch, label):
                return function(stage, batch, *args, **kwargs)
        return wrapped
    return decorate


def _manifest():
    path = os.environ.get(ENV)
    if not path:
        return None
    m = json.loads(Path(path).read_text())
    if not isinstance(m, dict) or m.get('mode') not in ('independent', 'latent-tail-v1'):
        raise ValueError('explicit independent or latent-tail-v1 mode required')
    trace, denoise = None, None
    if m.get('trace_policy') is not None or m.get('denoise_policy') is not None:
        from .utilization_policy import policies
        trace, denoise = policies(m.get('trace_policy'), m.get('denoise_policy'))
    if (trace is not None or denoise is not None) and (
            m['mode'] != 'latent-tail-v1' or m.get('parallel_acceptance')):
        raise ValueError('utilization requires isolated continuation manifest')
    if m['mode'] == 'independent':
        return m
    for key in ('index', 'total_frames', 'window_frames', 'overlap_frames', 'seed', 'expected_windows'):
        if type(m.get(key)) is not int:
            raise ValueError('integer manifest field required: ' + key)
    n = m['expected_windows']
    if not (1 <= n <= 4 and 0 <= m['index'] < n and
            m['total_frames'] == 124 + (n - 1) * 102 and
            m['window_frames'] == 124 and m['overlap_frames'] == 22 and m['seed'] == 314159):
        raise ValueError('only bounded 124/22 global continuation geometry is supported')
    if type(m.get('final')) is not bool or m['final'] != (m['index'] == n - 1):
        raise ValueError('final must match the last planned window')
    if not isinstance(m.get('session_id'), str) or not m['session_id']:
        raise ValueError('nonempty session_id required')
    if not isinstance(m.get('wav'), str) or not m['wav']:
        raise ValueError('source WAV required')
    sha = m.get('sha256')
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('lowercase source SHA256 required')
    for key in ('source_samples', 'source_sample_rate'):
        if key in m and (type(m[key]) is not int or m[key] <= 0):
            raise ValueError('positive source metadata required')
    if 'cancelled' in m and type(m['cancelled']) is not bool:
        raise ValueError('cancelled must be bool')
    for diagnostic in ('parallel_acceptance', 'measure_memory'):
        if diagnostic in m and type(m[diagnostic]) is not bool:
            raise ValueError('explicit boolean diagnostic field required')
    if m.get('parallel_acceptance') and m.get('candidate', {}).get('audio_context') != 'window-context-canonical-v1':
        raise ValueError('parallel acceptance requires v6 quality baseline')
    if 'candidate' in m:
        from .continuation_candidate import normalize
        if normalize(m['candidate']) is None:
            raise ValueError('explicit candidate contract required')
    if 'efficiency' in m:
        from .continuation_efficiency import normalize as efficiency_policy
        if efficiency_policy(m['efficiency']) is None or m.get('candidate', {}).get('audio_context') != 'window-context-canonical-v1':
            raise ValueError('efficiency requires explicit v6 quality baseline')
    if 'conversation' in m:
        from .conversation_stages import validate_manifest
        validate_manifest(m)
    if 'streaming_audio' in m:
        from .streaming_native_stages import validate_manifest as validate_stream
        validate_stream(m)
    return m


def active():
    m = _manifest()
    return m is not None and m['mode'] == 'latent-tail-v1'


def _source_hash(m):
    if hashlib.sha256(Path(m['wav']).read_bytes()).hexdigest() != m['sha256']:
        raise ValueError('source WAV bytes changed')


def _identity(m):
    return {k: v for k, v in m.items() if k not in ('index', 'final', 'cancelled')}


def _geometry(batch):
    shape = batch.raw_latent_shape
    if (not shape or len(shape) != 5 or shape[0] != 1 or shape[1] != 24 or
            shape[2] != video_t(124) or batch.num_frames != 124 or
            any(type(n) is not int or n <= 0 or n % 2 for n in shape[3:])):
        raise ValueError('native 124-frame, 24-channel, even latent canvas required')
    generators = batch.generator if isinstance(batch.generator, list) else [batch.generator]
    if not generators or any(g is None or g.initial_seed() != 314159 for g in generators):
        raise ValueError('continuation requires actual target-noise seed 314159')
    return shape[3], shape[4]


def _encode_source_local(stage, batch, args, m):
    """Build immutable canonical audio; default is one full-source encode."""
    import torch
    from fastvideo.distributed import get_local_torch_device
    from fastvideo.models import pinned_offload
    from fastvideo.pipelines.basic.minimax_h3.reference import decode_reference_audio, prepare_reference_waveform
    vae = stage.audio_vae
    if (int(vae.hop_length), int(vae.sampling_rate), int(vae.latent_channels)) != (800, 32000, 32):
        raise ValueError('pinned audio VAE geometry required')
    _source_hash(m)
    waveform, rate = decode_reference_audio(m['wav'])
    if type(rate) is not int or rate <= 0:
        raise ValueError('invalid source sample rate')
    samples = int(waveform.shape[-1])
    if samples <= 0 or ('source_samples' in m and m['source_samples'] != samples) or (
            'source_sample_rate' in m and m['source_sample_rate'] != rate):
        raise ValueError('source sample metadata differs')
    speech = prepare_reference_waveform(waveform, rate, 32000, max_duration=samples / rate + 1.0)
    total = audio_boundary(m['total_frames'])
    capacity = total * 800
    if samples * 32000 > capacity * rate or int(speech.shape[-1]) > capacity:
        raise ValueError('whole speech exceeds global capacity; never truncate')
    if speech.ndim != 2 or speech.shape[0] != 2 or not bool(torch.isfinite(speech).all()):
        raise ValueError('finite stereo waveform required')
    origin, next_history = None, None
    if 'conversation' in m:
        from .conversation_stages import validate_manifest
        from .conversation_audio import audio_origin, prepare_waveform, TorchWaveOps
        conversation = validate_manifest(m)
        origin = audio_origin(conversation['source_onset_frame'], m['total_frames'])
        speech, next_history, padded = prepare_waveform(
            speech, origin, getattr(stage, '_h3_conversation_audio_history', None), TorchWaveOps())
    else:
        padded = capacity - int(speech.shape[-1])
        speech = torch.cat((speech, torch.zeros((2, padded), dtype=speech.dtype, device=speech.device)), dim=-1)
    device = get_local_torch_device()
    loaded = any(p.device != device for p in vae.parameters())
    if loaded:
        pinned_offload.load(vae, device, pin=bool(getattr(args, 'pin_cpu_memory', False)))
    from .continuation_candidate import normalize, encode_window_context
    options = normalize(m.get('candidate'))
    window_context = options is not None and options['audio_context'] == 'window-context-canonical-v1'
    extra = {}
    def encode(waveform):
        posterior = vae.encode(waveform.to(device)[:, None]).latent_dist
        value = vae.normalize_latents(posterior.mode().float()).transpose(1, 2)
        return value.detach().cpu().clone().contiguous()
    try:
        with torch.no_grad():
            if window_context:
                from .continuation_plan import plan
                canonical, extra = encode_window_context(
                    speech, plan(m['total_frames'], 124, 22), encode)
            else:
                canonical = encode(speech)
    finally:
        if loaded and bool(getattr(args, 'vae_cpu_offload', False)):
            pinned_offload.unload(vae)
    _source_hash(m)
    if tuple(canonical.shape) != (2, total, 32) or not TorchOps().finite_cpu(canonical):
        raise ValueError('full source must encode to canonical normalized CPU [2,globalT,32]')
    if options is not None:
        extra.update(candidate_audio_context=options['audio_context'],
                     candidate_progress_receipts=options['progress_receipts'])
        extra.setdefault('source_window_encodes', 0)
    from .continuation_efficiency import normalize as efficiency_policy
    efficiency = efficiency_policy(m.get('efficiency'))
    if efficiency is not None:
        extra.update(efficiency_audio_validation=efficiency['audio_validation'],
                     efficiency_decode_schedule=efficiency['decode_schedule'])
    if origin is not None:
        # Publish history only after successful encoding/validation. Never send
        # waveform tensors through batch.extra or scalar receipts.
        stage._h3_conversation_audio_history = next_history
        extra['audio_origin'] = dict(origin)
    return canonical, dict(source_samples=samples, source_sample_rate=rate,
                           padded_samples=padded, global_audio_t=total,
                           full_source_encodes=0 if window_context else 1, **extra)


def _encode_source(stage, batch, args, m):
    from .continuation_candidate import normalize
    options = normalize(m.get('candidate'))
    if options is None or options['audio_context'] == 'full-source-v1':
        return _encode_source_local(stage, batch, args, m)
    # Candidate adds multiple local VAE calls. Agree on failure before either
    # rank can proceed into subsequent native/model collectives.
    import torch
    from fastvideo.distributed import get_local_torch_device
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() != 2:
        raise ValueError('candidate preparation requires initialized two-rank group')
    result, error = None, None
    try:
        result = _encode_source_local(stage, batch, args, m)
    except Exception as exc:
        error = exc
    ok = torch.tensor(0 if error else 1, dtype=torch.int32, device=get_local_torch_device())
    torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
    if not bool(ok.item()):
        raise WindowCancelled('collective candidate audio preparation rejected') from error
    return result


def append_conditions(stage, batch, args, references, video_rows, audio_rows):
    m = _manifest()
    if m is None or m['mode'] == 'independent':
        return video_rows, audio_rows
    if 'streaming_audio' in m:
        from .streaming_native_stages import append_conditions as streaming_append
        return streaming_append(stage, batch, args, references, video_rows, audio_rows, m)
    if 'conversation' in m:
        from .conversation_stages import append_conditions as conversation_append
        return conversation_append(stage, batch, args, references, video_rows, audio_rows, m)
    if m.get('cancelled') or getattr(stage, 'interrupt', False):
        raise WindowCancelled('continuation cancelled before preparation')
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
    height, width = _geometry(batch)
    if (not getattr(stage, 'ref2va', False) or not references or audio_rows or
            any(r.media_type != 'image' or r.has_audio or r.num_audio_latents or
                r.num_latent_frames != 1 for r in references) or len(video_rows) != len(references)):
        raise ValueError('unchanged ordered original image-only Ref2VA encodes required')
    if KEY in batch.extra:
        raise ValueError('duplicate continuation preparation')
    _source_hash(m)
    fixed_signature = tuple((r.num_latent_frames, r.latent_height, r.latent_width,
                             TorchOps().digest(rows)) for r, rows in zip(references, video_rows))
    state = getattr(stage, '_h3_continuation_owner', None)
    if state is None or state['identity']['session_id'] != m['session_id']:
        if m['index'] != 0 or (state is not None and not state['completed']):
            raise ValueError('new turn requires index0 and completed previous owner')
        source, info = _encode_source(stage, batch, args, m)
        owner = Session(m['session_id'], source_audio=source, total_frames=m['total_frames'],
                        height=height, width=width, window_frames=124, overlap_frames=22)
        state = dict(identity=_identity(m), owner=owner, geometry=(height, width),
                     fixed_signature=fixed_signature, info=info, completed=False)
        stage._h3_continuation_owner = state
    if (state['identity'] != _identity(m) or state['geometry'] != (height, width) or
            state['fixed_signature'] != fixed_signature):
        raise ValueError('immutable turn source/plan/canvas/fixed image rows changed')
    token = state['owner'].begin_window(m['index'])
    context = dict(manifest=dict(m), owner=state['owner'], token=token, state=state,
                   forwards=0, steps_held=0, pending_forward=False)
    batch.extra[KEY] = context
    # Allocate new lists: cached fixed rows/descriptors are never extended/mutated.
    vr, ar = list(video_rows), list(audio_rows)
    if token.guide_video is not None:
        descriptor = MiniMaxH3PreparedReference(
            media_type='video', has_audio=True, num_latent_frames=token.window.guide_video_t,
            latent_height=height, latent_width=width, num_audio_latents=token.window.guide_audio_t)
        vr.append(token.guide_video.clone())
        ar.append(token.guide_audio.clone().reshape(-1, 32))
        # The local list must change too: native shapes/noise are computed next.
        references.append(descriptor)
        if batch.references is not references:
            batch.references = list(batch.references) + [descriptor]
    return vr, ar


def make_layout(stage, batch, args):
    del stage, args
    from fastvideo.pipelines.basic.minimax_h3 import packing, reference
    c = batch.extra[KEY]
    from .late_listening_diagnostic import position_policy
    policy = position_policy(batch)
    height, width = _geometry(batch)
    # Only original images: build_layout itself creates the synthetic descriptor.
    images = [r for r in batch.references if r.media_type == 'image']
    return build_layout(packing, reference, text_tags=batch.extra['minimax_h3_text_token_tags'],
                        images=images, window=c['token'].window, height=height, width=width,
                        ranks=_world_size(), temporal_policy=policy)


def _world_size():
    import torch
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() != 2:
        raise ValueError('active worker continuation requires the initialized two-rank group')
    return 2


def _assert_audio(batch, c, *, canonical=True):
    import torch
    layout = batch.extra[LAYOUT_KEY]
    n = int(layout.num_condition_audio_rows)
    if (not canonical and c['manifest'].get('efficiency', {}).get('audio_validation') ==
            'device-steps-canonical-boundaries-v1'):
        if (batch.audio_latents.device != c['target_locked'].device or
                batch.audio_latents.dtype != c['target_locked'].dtype or
                tuple(batch.audio_latents.shape) != (n+c['target_locked'].shape[0], 32)):
            raise ValueError('locked audio shape/device/dtype drift')
        # Device boolean joins the existing MIN without any local .item(),
        # torch.equal host wait, or D2H canonical copy. Bind/commit below still
        # verify exact CPU canonical correspondence before publication.
        return torch.logical_and(torch.eq(batch.audio_latents[:n], c['guide_locked']).all(),
                                 torch.eq(batch.audio_latents[n:], c['target_locked']).all())
    if (not torch.equal(batch.audio_latents[:n], c['guide_locked']) or
            not torch.equal(batch.audio_latents[n:], c['target_locked'])):
        raise ValueError('guide or target locked audio changed')
    # Receipt clones must themselves still match the Session-owned canonical source.
    token = c['token']
    ops = c['owner'].ops
    if not ops.equal(c['target_locked'].detach().cpu().reshape(2, -1, 32), token.locked_audio):
        raise ValueError('target source clone changed or dtype lost precision')
    if token.guide_audio is not None and not ops.equal(
            c['guide_locked'].detach().cpu().reshape(2, -1, 32), token.guide_audio):
        raise ValueError('guide source clone changed')


def bind_audio(stage, batch, args):
    del args
    import torch
    c = batch.extra[KEY]
    def check():
        token, layout = c['token'], batch.extra[LAYOUT_KEY]
        n = 2 * token.window.guide_audio_t
        if (int(layout.num_condition_audio_rows) != n or
                int(layout.num_audio_latents) != token.window.audio_end - token.window.audio_start):
            raise ValueError('layout must use exact GLOBAL audio boundaries')
        target = token.locked_audio.reshape(-1, 32).to(batch.audio_latents).clone()
        guide = (token.guide_audio.reshape(-1, 32).to(batch.audio_latents).clone()
                 if n else target[:0].clone())
        if tuple(batch.audio_latents.shape) != (n + target.shape[0], 32):
            raise ValueError('packed audio row count mismatch')
        c['guide_locked'], c['target_locked'] = guide.clone(), target.clone()
        batch.audio_latents = torch.cat((guide, target))
        _assert_audio(batch, c)
        _receipt(batch, committed=False)
    _collective_check(stage, batch, check)
    return batch


def _collective_check(stage, batch, check):
    """All ranks reach MIN even on a local manifest/validation error."""
    import torch
    c = batch.extra[KEY]
    error, device_ok = None, None
    try:
        if c['owner'].pending_window is not c['token'] or c['owner'].faulted:
            raise ValueError('stale or faulted continuation attempt')
        if _manifest() != c['manifest'] or c['manifest'].get('cancelled') or getattr(stage, 'interrupt', False):
            raise WindowCancelled('manifest replaced/cancelled or worker interrupted')
        _source_hash(c['manifest'])
        device_ok = check()
        if device_ok is not None and (device_ok.shape != () or
                device_ok.dtype != torch.bool or device_ok.device != batch.latents.device):
            raise ValueError('scalar device boolean validation required')
    except Exception as exc:
        error = exc
    ok = (device_ok.to(dtype=torch.int32) if error is None and device_ok is not None else
          torch.tensor(0 if error else 1, dtype=torch.int32, device=batch.latents.device))
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() != 2:
        error = error or ValueError('continuation collective requires the initialized two-rank group')
        ok.fill_(0)
    else:
        try:
            torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
        except Exception:
            # Timeout/transport errors are terminal too; never leave an owner
            # apparently reusable after failed agreement. No retry/reset here.
            if c['owner'].pending_window is c['token']:
                c['owner'].fault_window(c['token'])
            raise
    if not bool(ok.item()):
        if c['owner'].pending_window is c['token']:
            c['owner'].fault_window(c['token'])
        raise WindowCancelled('collective continuation boundary rejected') from error


@_profile_stage('h3.before_forward')
def before_forward(stage, batch, index):
    c = batch.extra[KEY]
    def check():
        if (type(index) is not int or index != c['steps_held'] or index >= 4 or
                c['pending_forward'] or len(stage.scheduler.timesteps) != 4 or
                len(stage.audio_scheduler.timesteps) != 4):
            raise ValueError('ordered existing SMD4/four-forward schedule required')
        controller = batch.extra.get('_h3_denoise_execution')
        if controller is not None:
            controller.on_before_forward(stage, batch, index)
        return _assert_audio(batch, c, canonical=False)
    _collective_check(stage, batch, check)
    c['forwards'] += 1
    c['pending_forward'] = True


@_profile_stage('h3.hold')
def hold_audio(stage, batch, audio_velocity):
    """Discard velocity. Overlay MUST NOT also call audio_scheduler.step."""
    c = batch.extra[KEY]
    def check():
        if audio_velocity is None or not c['pending_forward'] or c['steps_held'] >= 4:
            raise ValueError('one audio hold per actual forward required')
        device_ok = _assert_audio(batch, c, canonical=False)
        controller = batch.extra.get('_h3_denoise_execution')
        if controller is not None:
            controller.on_hold_audio(stage, batch)
        c['steps_held'] += 1
        c['pending_forward'] = False
        # Separate performance toggle: retain bind/final receipts and every
        # lock/collective/commit check; only omit overwritten progress I/O.
        if c['manifest'].get('candidate', {}).get('progress_receipts') != 'bind-and-commit-v1':
            _receipt(batch, committed=False)
        return device_ok
    _collective_check(stage, batch, check)


@_profile_stage('h3.commit')
def commit(stage, batch, args):
    if 'conversation_owner' in batch.extra[KEY]:
        from .conversation_stages import commit as conversation_commit
        return conversation_commit(stage, batch, args)
    del args
    c = batch.extra[KEY]
    clean = {}
    def check():
        if (c['forwards'] != 4 or c['steps_held'] != 4 or c['pending_forward'] or 'suffix' in c):
            raise ValueError('commit requires exactly four completed SMD4 forwards')
        _assert_audio(batch, c)
        controller = batch.extra.get('_h3_denoise_execution')
        if controller is not None:
            controller.on_commit(stage, batch)
        layout, token, owner = batch.extra[LAYOUT_KEY], c['token'], c['owner']
        owner._source_ok()
        for value, digest in zip((token.guide_video, token.guide_audio, token.locked_audio), owner._digests):
            if value is not None and owner.ops.digest(value) != digest:
                raise ValueError('clean borrowed guide/source mutated')
        clean['video'] = owner.ops.copy(batch.latents[int(layout.num_condition_video_rows):])
        clean['audio'] = owner.ops.copy(batch.audio_latents[int(layout.num_condition_audio_rows):].reshape(2, -1, 32))
        owner._check(clean['video'], (video_t(124) * owner._per_frame, 96))
        owner._check(clean['audio'], tuple(token.locked_audio.shape))
        if not owner.ops.equal(clean['audio'], token.locked_audio):
            raise ValueError('clean target not exact source audio')
    _collective_check(stage, batch, check)
    c['suffix'] = c['owner'].commit_window(c['token'], clean_video=clean['video'],
                                         clean_audio=clean['audio'], observed_sampler=SamplerContract('SMD4', 4))
    c['state']['completed'] = c['manifest']['final']
    _receipt(batch, committed=True)
    return batch


def _receipt(batch, *, committed):
    import torch
    folder = os.environ.get('H3_REF2VA_COVERAGE_DIR')
    if not folder:
        raise RuntimeError('private H3_REF2VA_COVERAGE_DIR required')
    c = batch.extra[KEY]
    m, t = c['manifest'], c['token']
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    r = dict(schema=SCHEMA, session_id=m['session_id'], window_index=m['index'], worker_rank=rank,
             mode=m['mode'], wav_sha256=m['sha256'], total_frames=m['total_frames'],
             window_frames=124, overlap_frames=22, seed=314159, expected_windows=m['expected_windows'],
             final=m['final'], audio_start=t.window.audio_start, audio_end=t.window.audio_end,
             guide_video_t=t.window.guide_video_t, guide_audio_t=t.window.guide_audio_t,
             target_audio_rows=int(c['target_locked'].shape[0]),
             guide_audio_rows=int(c['guide_locked'].shape[0]), forwards=c['forwards'],
             steps_held=c['steps_held'], sampler='SMD4', audio_timestep_policy='fixed_clean_1.0',
             audio_velocity_discarded=True, audio_scheduler_step_skipped=True,
             rows_unchanged=True, committed=committed, truncation=False, gpu_validated=False,
             **c['state']['info'])
    if committed:
        s = c['suffix']
        r.update(suffix_video_rows=int(s.video.shape[0]), suffix_audio_t=int(s.audio.shape[1]),
                 video_row_offset=s.video_row_offset, audio_time_offset=s.audio_time_offset,
                 cumulative_video_rows=s.cumulative_video_rows, cumulative_audio_t=s.cumulative_audio_t)
    path = Path(folder) / f'h3-continuation-rank-{rank}-window-{m["index"]}.json'
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix('.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(r, stream, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)
