"""Narrow USER-approved HF price-precision exception, not a general rate waiver."""
RATE = 10.00002
MAX_SECONDS = 4800
MAX_USD = 13.34


def validate(proof):
    scope=proof.get('scope') if isinstance(proof,dict) else None
    if scope not in ('h3-four-condition-matrix-v1','h3-corrected-native-continuation-v1',
                     'h3-fp8-streaming-experiment-v1'):
        raise ValueError('unapproved experiment scope')
    expected = dict(schema='h3-hf-price-precision-approval-v1', approved=True,
                    provider='huggingface_jobs', flavor='h200x2',
                    scope=scope, max_rate_usd_hour=RATE,
                    max_timeout_seconds=MAX_SECONDS, max_total_usd=MAX_USD)
    if (not isinstance(proof, dict) or any(proof.get(k)!=v for k,v in expected.items())
            or not proof.get('approved_by') or not proof.get('approval_text')):
        raise ValueError('explicit fixed HF two-H200 price-precision approval required')
    return RATE


def request_limit(request):
    proof=request.get('hf_price_precision_approval')
    validate(proof)
    native=request.get('inference',{}).get('native_continuation')
    experiment=request.get('inference',{}).get('acceleration_experiment')
    if experiment is not None or proof['scope']=='h3-fp8-streaming-experiment-v1':
        from .acceleration_experiment import validate as check_experiment
        from .continuation_contract import validate as check
        if proof['scope']!='h3-fp8-streaming-experiment-v1':
            raise ValueError('experimental precision requires separate explicit scope')
        check_experiment(experiment);check(native)
    elif native is not None or proof['scope']=='h3-corrected-native-continuation-v1':
        from .continuation_contract import validate as check
        if proof['scope']!='h3-corrected-native-continuation-v1':
            raise ValueError('matrix price approval cannot authorize native continuation')
        check(native)
    if (request.get('rate_usd_hour')!=RATE or request.get('preview_only') is not True or
        request.get('territory_approval',{}).get('provider')!='huggingface_jobs' or
        not 0 < request.get('overall_seconds',0) <= MAX_SECONDS):
        raise ValueError('HF precision exception is finite inference-only at exact approved rate')
    return RATE
