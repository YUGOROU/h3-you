"""Host-RPC conversation abandonment. Never resets or unloads a native model."""
import json
import os
from pathlib import Path

SCHEMA = 'h3-conversation-control-v1'


def read_manifest():
    path = os.environ.get('H3_SMD_CONTINUATION_WINDOW_FILE')
    value = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(value, dict):
        raise ValueError('manifest must be an object')
    return value


def roles_active():
    from .conversation_conditioning import ROLES_ENV
    return bool(os.environ.get(ROLES_ENV))


def is_control():
    if not roles_active():
        return False
    # Presence, not successful validation: malformed controls must never cold-release.
    path = os.environ.get('H3_SMD_CONTINUATION_WINDOW_FILE')
    if not path:
        return False
    try:
        return 'conversation_control' in read_manifest()
    except Exception:
        return True


def validate_control(m):
    if set(m) != {'conversation_control'}:
        raise ValueError('control must be independent of media manifest')
    c = m['conversation_control']
    if (not isinstance(c, dict) or
            set(c) != {'schema', 'action', 'conversation_id', 'reason'} or
            c['schema'] != SCHEMA or c['action'] not in ('abandon','retire') or
            type(c['conversation_id']) is not str or not c['conversation_id'] or
            len(c['conversation_id']) > 256 or
            (c['action'],c['reason']) not in (('abandon','local-sink-failure'),
                ('abandon','local-preparation-failure'),('retire','completed'))):
        raise ValueError('invalid conversation control')
    return c


def healthy(pipeline, stage):
    if (getattr(pipeline, '_smd_continuation_warm_faulted', False) or
            getattr(pipeline, '_h3_utilization_faulted', False) or
            getattr(stage, '_h3_conversation_model_healthy', None) is not True):
        raise RuntimeError('known healthy conversation model required')


def boundary(pipeline, batch, device):
    """All ranks enter, including local parse failures; before any native step."""
    if not roles_active() or not os.environ.get('H3_SMD_CONTINUATION_WINDOW_FILE'):
        return False
    from .conversation_stages import common, validate_manifest
    from .fast_ref2va import static_merge_receipt
    mapping = pipeline._stage_name_mapping
    stage = mapping['latent_preparation_stage']
    m = read_manifest()
    if 'conversation_control' not in m and 'conversation' not in m and 'streaming_audio' not in m:
        if m:
            from .fast_continuation_stages import _manifest
            _manifest()  # Legacy validation; independent startup has no WAV.
        return False

    def prepare():
        if 'conversation_control' in m:
            c = validate_control(m)
            healthy(pipeline, stage)
            state = getattr(stage, '_h3_conversation_state', None)
            retired = getattr(stage, '_h3_retired_conversations', ())
            if (c['conversation_id'] in retired or len(retired) >= 4 or
                    state is None or state['owner'].conversation_id != c['conversation_id']):
                raise ValueError('unknown or retired conversation')
            if c['action']=='retire' and (not state['owner']._final_acked
                    or state['owner'].emittedF!=state['owner'].committedF):
                raise ValueError('completed final decoder ACK required for retirement')
            # This function is called only at the serial generate RPC entry.
            if (getattr(pipeline, '_h3_conversation_control_boundary', None) is not True or
                    getattr(pipeline, '_h3_conversation_forward_inflight', False) or
                    state['owner']._active is not None):
                raise RuntimeError('forward still in flight')
            return m, c
        if m:
            from .fast_continuation_stages import _manifest, _source_hash
            validated = _manifest()
            _source_hash(validated)
        if 'conversation' in m or 'streaming_audio' in m:
            if 'streaming_audio' in m:
                from .streaming_native_stages import validate_manifest as validate_stream
                c = validate_stream(m)
            else:
                c = validate_manifest(m)
            if c['id'] in getattr(stage, '_h3_retired_conversations', ()):
                raise ValueError('retired conversation')
            if not hasattr(stage, '_h3_conversation_model_healthy'):
                if (getattr(pipeline, '_denoise_stages_ready', None) is not True or
                        getattr(pipeline, '_smd_continuation_warm_faulted', False) or
                        getattr(pipeline, '_h3_utilization_faulted', False)):
                    raise RuntimeError('native pipeline not healthy')
                static_merge_receipt(pipeline)
            else:
                healthy(pipeline, stage)
        return m, None

    m, c = common(device, prepare)
    if c is None:
        if ('conversation' in m or 'streaming_audio' in m) and not hasattr(stage, '_h3_conversation_model_healthy'):
            stage._h3_conversation_model_healthy = True
        return False

    def cleanup():
        from .conversation_decode import abandon
        stage._h3_conversation_state['owner'].abandon()
        abandon(mapping['video_decoding_stage'])
    # Failed cleanup retains poisoned ownership; never declares it reusable.
    common(device, cleanup)
    stage._h3_retired_conversations = (*getattr(stage, '_h3_retired_conversations', ()), c['conversation_id'])
    stage._h3_conversation_state = None
    if hasattr(stage, '_h3_conversation_audio_history'):
        del stage._h3_conversation_audio_history
    mapping['video_decoding_stage']._h3_conversation_decoder = None
    import torch
    batch.output = torch.empty((0, 3, 0, 0, 0), device='cpu', dtype=torch.float32)
    batch.audio_output = None
    return True
