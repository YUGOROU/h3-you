"""Narrow USER-approved two-Job exception. Default authority remains unchanged."""
PAIR=dict(schema='h3-explicit-two-job-approval-v1',id='h3-native-fp8-pair-20261015-v1',
    approved_by='USER',approval_text='USER reported HF107.80+Modal35.59 and said go to170USD cumulative and two parallel Jobs at20.00004USD/hour',
    cumulative_cap_usd=170,max_concurrent_usd_hour=20.00004,max_concurrent_jobs=2,max_timeout_seconds=4500,
    provider='huggingface_jobs',flavor='h200x2')
SCOPES={'h3-corrected-native-continuation-v1','h3-fp8-streaming-experiment-v1'}


def approval(proof):
    value=(proof or {}).get('parallel_pair')
    if value is None:return None
    if value!=PAIR or proof.get('scope') not in SCOPES:
        raise ValueError('explicit narrow two-Job authority required')
    return dict(value)


def cap(proof):
    return 170 if approval(proof) else 150
