"""Official-price single-GPU comparison; no runtime/quality or speedup claims.

Sources: modal.com/pricing, modal.com/docs/guide/gpu,
modal.com/docs/guide/region-selection, pytorch.org/blog/pytorch-2-7/.
"""
GPU_HOURLY = {'H200': 4.5396, 'B200': 6.2496, 'B300': 7.0992}
HOST_HOURLY = 4 * .04716 + 256 * .007992
BROAD = {'us', 'eu', 'ap'}
NARROW = {'us-east', 'us-central', 'us-south', 'us-west', 'eu-west', 'eu-north', 'eu-south',
          'ap-northeast', 'ap-southeast', 'ap-south', 'ap-melbourne', 'jp', 'au', 'uk', 'ca', 'me', 'sa', 'af', 'mx'}
COUNTRIES = {'jp': 'JP', 'au': 'AU', 'uk': 'GB', 'ca': 'CA', 'mx': 'MX'}


def multiplier(region=None):
    if region is None:
        return 1.0
    if region in BROAD:
        return 1.15
    if region in NARROW:
        return 1.75
    raise ValueError('actual Modal region identifiers required, not AWS region names')


def hourly(gpu='H200', region=None):
    if gpu not in GPU_HOURLY:
        raise ValueError('explicit single GPU only; no fallback/multi-GPU syntax')
    return (GPU_HOURLY[gpu] + HOST_HOURLY) * multiplier(region)


def comparison(cuda='12.8', region=None):
    version = tuple(int(part) for part in cuda.split('.')[:2])
    h200 = hourly('H200', region)
    return {gpu: {'hourly_usd': hourly(gpu, region), 'cost_break_even_speedup': hourly(gpu, region) / h200,
                  'cuda_minimum_met': version >= ((13, 1) if gpu == 'B300' else (12, 8)),
                  'h3_gpu_verified': gpu == 'H200', 'quality_verified_on_this_gpu': False,
                  'production_guard_supported': gpu == 'H200'} for gpu in GPU_HOURLY}


def authorized_rate_limit(request):
    """Default10/hr; only pinned explicit JP/H200 Modal user approval exceeds it."""
    if request.get('rate_usd_hour', 0) <= 10:
        return 10
    if 'hf_price_precision_approval' in request:
        from research.h3_personal.compute_price_approval import request_limit
        return request_limit(request)
    from research import hf_identity_train as trainer
    pin = request.get('compute_rate_approval', {})
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('pinned rate approval required above10/hour')
    proof = trainer.read_json(pin['path'])
    expected = hourly('H200', 'jp')
    if (trainer.sha256(pin['path']) != pin.get('sha256')
            or proof.get('schema') != 'h3-modal-rate-approval-v1'
            or proof.get('approved') is not True or not proof.get('approved_by')
            or not proof.get('approval_text') or proof.get('provider') != 'modal'
            or proof.get('gpu') != 'H200' or proof.get('region') != 'jp'
            or proof.get('rate_usd_hour') != expected or proof.get('max_total_usd') != 50
            or request.get('rate_usd_hour') != expected
            or request.get('territory_approval', {}).get('provider') != 'modal'
            or request.get('territory_approval', {}).get('compute_country') != 'JP'
            or request['overall_seconds'] * expected / 3600 > 50):
        raise ValueError('pinned explicit JP H20050USD user rate approval required')
    return expected


def validate_placement(placement, evidence):
    region = placement['region']
    # A broad pool is not proof of a particular non-excluded country.
    if COUNTRIES.get(region) != placement['country']:
        raise ValueError('single-country Modal region/country binding required')
    if (evidence.get('approved') is not True or evidence.get('provider') != 'modal'
            or evidence.get('region_multiplier') != multiplier(region)
            or any(evidence.get(k) != placement[k] for k in ('cloud', 'region', 'country'))):
        raise ValueError('placement/pricing evidence differs from published regional multiplier')
    return hourly('H200', region)
