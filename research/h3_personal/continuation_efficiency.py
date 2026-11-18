"""Explicit precision-preserving efficiency policy; no device/model operations."""
SCHEMA = 'h3-continuation-efficiency-v1'
AUDIO = ('per-step-canonical-v1', 'device-steps-canonical-boundaries-v1')
DECODE = ('alternating-v1', 'two-clip-waves-v1')


def comparison_contract():
    return dict(schema='h3-efficiency-comparison-v1',
                order=['acceptance', 'baseline', 'optimized'],
                profile_stages=False, atol=0.0,
                retain_latents_in_memory_only=True, total_frames=328,
                arrivals=[37, 30, 30])


def normalize(options):
    if options is None:
        return None
    if type(options) is not dict or set(options) != {'schema', 'audio_validation', 'decode_schedule'}:
        raise ValueError('invalid efficiency policy keys/type')
    for key, allowed in (('schema', (SCHEMA,)), ('audio_validation', AUDIO), ('decode_schedule', DECODE)):
        if type(options[key]) is not str or options[key] not in allowed:
            raise ValueError('invalid efficiency policy ' + key)
    return dict(options)


def make_options(*, device_audio_validation=False, parallel_decode_waves=False):
    if any(type(v) is not bool for v in (device_audio_validation, parallel_decode_waves)):
        raise ValueError('efficiency switches must be explicit bool')
    if not device_audio_validation and not parallel_decode_waves:
        return None
    return dict(schema=SCHEMA, audio_validation=AUDIO[int(device_audio_validation)],
                decode_schedule=DECODE[int(parallel_decode_waves)])
