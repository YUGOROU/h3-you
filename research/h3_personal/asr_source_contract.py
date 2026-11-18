"""Strict ASR provenance; extraction is never speaker identity verification."""
import copy
import re

_BASE = dict(policy='endpoint-native-v1', endpoint_name='h3-asr-t4-7c926aae18',
             endpoint_source='native_completed', completion_policy='input-finalized',
             capture_stopped=True, cleanup_complete=True)
_CAPTURE = set('schema profile_sha256 input_epoch captured_samples extracted_samples submitted_samples capture_frames capture_stopped capture_cleanup capture_app mute_behavior eof pending_samples discarded_pending_samples raw_audio_fallback operator_confirmed_remote_idle'.split())


def validate_expected_asr(meta):
    """Return a detached validated immutable-session expectation (Comet or Zoom)."""
    if type(meta) is not dict or set(meta) not in (set(_BASE) | {'revision'}, set(_BASE) | {'revision', 'input_source'}):
        raise ValueError('exact expected native ASR metadata required')
    if (any(type(meta[k]) is not type(v) or meta[k] != v for k, v in _BASE.items())
            or type(meta['revision']) is not str or re.fullmatch('[0-9a-f]{40}', meta['revision']) is None):
        raise ValueError('exact completed native ASR metadata required')
    if 'input_source' in meta:
        from research.h3_personal.zoom_input import source_contract
        s = meta['input_source']
        if type(s) is not dict:
            raise ValueError('exact Zoom source required')
        try:
            rebuilt = source_contract(reference_sha256=s['reference_sha256'], capture_source_sha256=s['capture_source_sha256'])
        except (KeyError, TypeError) as exc:
            raise ValueError('exact Zoom source required') from exc
        # JSON equality alone would accept bools in numeric profile fields.
        import json
        if json.dumps(s, sort_keys=True) != json.dumps(rebuilt, sort_keys=True):
            raise ValueError('Zoom source profile drift')
    return copy.deepcopy(meta)


def validate_actual_asr(actual, expected=None):
    """Validate finalized capture, optionally bound to the session expectation."""
    if type(actual) is not dict:
        raise ValueError('exact actual ASR metadata required')
    base = {k: v for k, v in actual.items() if k != 'zoom_capture'}
    validated = validate_expected_asr(base)
    if expected is not None and validated != validate_expected_asr(expected):
        raise ValueError('ASR immutable source mismatch')
    zoom = 'input_source' in validated
    if set(actual) != set(validated) | ({'zoom_capture'} if zoom else set()):
        raise ValueError('exact actual ASR variant required')
    if zoom:
        c = actual['zoom_capture']
        if type(c) is not dict or set(c) != _CAPTURE:
            raise ValueError('exact Zoom capture receipt required')
        fixed = dict(schema='h3-zoom-asr-capture-v1', profile_sha256=validated['input_source']['profile_sha256'],
                     capture_stopped=True, capture_cleanup=True, capture_app='zoom', mute_behavior='unmuted',
                     eof=True, pending_samples=0, discarded_pending_samples=0, raw_audio_fallback=False,
                     operator_confirmed_remote_idle=True)
        if any(type(c[k]) is not type(v) or c[k] != v for k, v in fixed.items()):
            raise ValueError('Zoom capture not finalized/eligible')
        for k in ('input_epoch', 'captured_samples', 'extracted_samples', 'submitted_samples', 'capture_frames'):
            if type(c[k]) is not int or c[k] <= 0:
                raise ValueError('positive integer capture counts required')
        if not (c['captured_samples'] == c['extracted_samples'] == c['submitted_samples'] <= c['capture_frames'] <= 720000):
            raise ValueError('Zoom extracted sample clock mismatch/bound')
    return copy.deepcopy(actual)


def asr_origin(meta):
    validate_actual_asr(meta)
    return 'zoom_process_voxlock_endpoint_asr' if 'input_source' in meta else 'comet_process_audio_endpoint_asr'
