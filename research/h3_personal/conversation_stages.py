"""Explicit finite native conversation hooks; legacy Session remains unchanged."""
from .conversation_continuation import ConversationOwner
from .fast_continuation import SamplerContract, WindowCancelled

SCHEMA = 'h3-model-conversation-v1'


def source_identity(m):
    from copy import deepcopy
    identity = deepcopy({k: v for k, v in m.items()
                         if k not in ('index', 'source_index', 'final', 'cancelled')})
    identity['conversation'].pop('conversation_final')
    return identity


def validate_manifest(m):
    c = m.get('conversation')
    required = {'schema', 'id', 'profile_sha256', 'source_epoch', 'source_id',
                'role', 'source_onset_frame', 'conversation_final', 'total_frames'}
    if not isinstance(c, dict) or set(c) != required:
        raise ValueError('exact conversation fields required')
    sha = c['profile_sha256']
    if (c['schema'] != SCHEMA or not isinstance(c['id'], str) or not c['id'] or
            not isinstance(sha, str) or len(sha) != 64 or
            any(x not in '0123456789abcdef' for x in sha) or
            type(c['source_epoch']) is not int or not 0 <= c['source_epoch'] <= 2 or
            c['source_id'] != m['session_id'] or
            c['role'] != ('speech' if c['source_epoch'] == 1 else 'listening') or
            type(c['source_onset_frame']) is not int or
            c['source_onset_frame'] != (0, 124, 430)[c['source_epoch']] or
            type(c['conversation_final']) is not bool or
            c['conversation_final'] != (c['source_epoch'] == 2 and m['final']) or
            type(c['total_frames']) is not int or c['total_frames'] != 532 or
            m['total_frames'] != (328 if c['source_epoch'] == 1 else 124) or
            m['mode'] != 'latent-tail-v1' or
            m.get('candidate', {}).get('audio_context') != 'window-context-canonical-v1'):
        raise ValueError('invalid bounded conversation contract')
    if type(m.get('source_index')) is not int or m['source_index'] != m['index']:
        raise ValueError('source index drift')
    return c


def common(device, operation):
    """Vote after all local allocations, before any model collective/publication."""
    import torch
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = exc
    status = torch.tensor(0 if error else 1, dtype=torch.int32, device=device)
    d = torch.distributed
    if not d.is_initialized() or d.get_world_size() != 2:
        raise WindowCancelled('conversation requires two ranks') from error
    d.all_reduce(status, op=d.ReduceOp.MIN)
    if not bool(status.item()):
        raise WindowCancelled('common conversation boundary rejected') from error
    return result


class SourceFacade:
    """Legacy hook view: model token identity, source-local fault mapping."""
    def __init__(self, owner, token):
        self.owner, self.token = owner, token
        self.session_id = token.source_id
        self.ops = owner.ops
        self._per_frame = owner.per_frame
        self._digests = owner._digests

    @property
    def pending_window(self):
        return self.owner.pending_window

    @property
    def faulted(self):
        return self.owner.faulted or self.owner._terminal

    def _source_ok(self):
        self.owner._source_ok()

    def _check(self, value, shape):
        self.owner._check(value, shape)

    def fault_window(self, token):
        if token is not self.token.model:
            raise ValueError('foreign model token')
        self.owner.source_session.fault_window(self.token.local)


def append_conditions(stage, batch, args, references, video_rows, audio_rows, m):
    from . import fast_continuation_stages as legacy
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
    from fastvideo.distributed import get_local_torch_device
    device = get_local_torch_device()
    state = getattr(stage, '_h3_conversation_state', None)

    def prepare():
        nonlocal state
        c = validate_manifest(m)
        if m.get('cancelled') or getattr(stage, 'interrupt', False) or (state and state.get('failed')):
            raise WindowCancelled('cancelled or failed conversation')
        height, width = legacy._geometry(batch)
        if (not getattr(stage, 'ref2va', False) or not references or audio_rows or
                len(references) != len(video_rows) or legacy.KEY in batch.extra or
                any(r.media_type != 'image' or r.has_audio or r.num_audio_latents or
                    r.num_latent_frames != 1 for r in references)):
            raise ValueError('original image-only references required')
        legacy._source_hash(m)
        signature = tuple((r.num_latent_frames, r.latent_height, r.latent_width,
                           legacy.TorchOps().digest(v)) for r, v in zip(references, video_rows))
        binding = (c['id'], c['profile_sha256'], height, width, signature)
        if state is None:
            if c['source_epoch'] != 0 or m['index'] != 0:
                raise ValueError('first source required')
            owner = ConversationOwner(c['id'], profile=c['profile_sha256'], height=height,
                                      width=width, model_healthy=getattr(stage, '_h3_conversation_model_healthy', None),
                                      enabled=True, max_sessions=3)
            state = dict(owner=owner, binding=binding, epoch=-1, completed=True)
            stage._h3_conversation_state = state
        if state['binding'] != binding:
            raise ValueError('fixed conversation binding changed')
        owner = state['owner']
        if state['epoch'] != c['source_epoch']:
            if (m['index'] != 0 or c['source_epoch'] != state['epoch']+1 or
                    not state['completed'] or owner.active_native is not None or
                    owner.emittedF != max(0, owner.committedF-5)):
                raise ValueError('source transition requires previous decode ACK')
        return owner, height, width

    owner, height, width = common(device, prepare)
    try:
        if state['epoch'] != m['conversation']['source_epoch']:
            # This helper already votes on both ranks after all VAE calls.
            source, info = legacy._encode_source(stage, batch, args, m)
            def start():
                owner.start_source(m['session_id'], canonical_audio=source,
                                   total_frames=m['total_frames'], profile=m['conversation']['profile_sha256'],
                                   height=height, width=width, audio_origin=info.get('audio_origin'))
                state.update(epoch=m['conversation']['source_epoch'], identity=source_identity(m),
                             info=info, completed=False)
            common(device, start)
        def allocate():
            if state['identity'] != source_identity(m):
                raise ValueError('immutable source identity changed')
            if m['index'] != owner._index:
                raise ValueError('source window index drift')
            token = owner.begin_window()
            facade = SourceFacade(owner, token)
            context = dict(manifest=dict(m), owner=facade, conversation_owner=owner,
                           global_token=token, token=token.model, state=state,
                           forwards=0, steps_held=0, pending_forward=False)
            vr, ar = list(video_rows), list(audio_rows)
            descriptor = None
            if token.model.guide_video is not None:
                t = token.model
                descriptor = MiniMaxH3PreparedReference(media_type='video', has_audio=True,
                    num_latent_frames=t.window.guide_video_t, latent_height=height,
                    latent_width=width, num_audio_latents=t.window.guide_audio_t)
                vr.append(t.guide_video.clone())
                ar.append(t.guide_audio.clone().reshape(-1, 32))
            refs = list(references) + ([descriptor] if descriptor else [])
            return context, vr, ar, refs
        context, vr, ar, refs = common(device, allocate)
        batch.extra[legacy.KEY] = context
        references[:] = refs
        batch.references = refs
        return vr, ar
    except Exception:
        state['failed'] = True
        raise


def commit(stage, batch, args):
    from . import fast_continuation_stages as legacy
    c = batch.extra[legacy.KEY]
    owner = c['conversation_owner']
    def prepare():
        if (c['forwards'] != 4 or c['steps_held'] != 4 or c['pending_forward'] or 'suffix' in c):
            raise ValueError('four completed forwards required')
        if (legacy._manifest() != c['manifest'] or c['manifest'].get('cancelled') or
                getattr(stage, 'interrupt', False) or c['state'].get('failed')):
            raise WindowCancelled('manifest drift/interrupt')
        legacy._source_hash(c['manifest'])
        legacy._assert_audio(batch, c)
        controller = batch.extra.get('_h3_denoise_execution')
        if controller is not None:
            controller.on_commit(stage, batch)
        layout = batch.extra[legacy.LAYOUT_KEY]
        video = owner.ops.copy(batch.latents[int(layout.num_condition_video_rows):])
        audio = owner.ops.copy(batch.audio_latents[int(layout.num_condition_audio_rows):].reshape(2, -1, 32))
        return owner.prepare_commit(c['global_token'], clean_video=video, clean_audio=audio,
            observed_sampler=SamplerContract('SMD4', 4),
            conversation_final=(c['manifest'].get('streaming_audio') or c['manifest'].get('conversation'))['conversation_final'])
    try:
        ticket = common(batch.latents.device, prepare)
        result = owner.apply_prepared(ticket)
        c['global_suffix'], c['suffix'] = result, result.suffix
        c['state']['completed'] = c['manifest']['final']
        # No legacy pending-window check after publication.
        if 'streaming_audio' in c['manifest']:
            from .streaming_native_stages import receipt as stream_receipt
            common(batch.latents.device, lambda: stream_receipt(batch))
        else:
            common(batch.latents.device, lambda: receipt(batch))
    except Exception:
        c['state']['failed'] = True
        raise
    return batch


def receipt(batch):
    from . import fast_continuation_stages as legacy
    # Existing receipt writer includes only scalar source/guide geometry, no media.
    legacy._receipt(batch, committed=True)
    import json, os
    from pathlib import Path
    import torch
    c = batch.extra[legacy.KEY]
    owner, token = c['conversation_owner'], c['global_token']
    rank = torch.distributed.get_rank()
    path = Path(os.environ['H3_REF2VA_COVERAGE_DIR']) / f'h3-conversation-rank-{rank}-epoch-{c["state"]["epoch"]}-window-{token.local.index}.json'
    data = dict(schema=SCHEMA, conversation=c['manifest']['conversation'],
                committed_frames=owner.committedF, source_id=token.source_id,
                canonical_audio_sha256=owner.canonical_digest,
                audio_origin=c['state']['info']['audio_origin'],
                guide_digests=owner._digests)
    temporary = path.with_suffix('.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, sort_keys=True)
    os.replace(temporary, path)


def abandon(stage, *, model_healthy, video_stage=None):
    """Explicit terminal disposal, never a model-health inference or reset."""
    if model_healthy is not True:
        raise ValueError('external healthy-model admission required')
    state = getattr(stage, '_h3_conversation_state', None)
    if state is None:
        raise ValueError('no conversation')
    owner = state['owner']
    if owner.active_native is not None and not (owner.faulted or state.get('failed')):
        raise ValueError('cannot abandon in-flight healthy window')
    if video_stage is not None:
        from .conversation_decode import abandon as abandon_decode
        abandon_decode(video_stage)
    owner.abandon()
    state['failed'] = True
