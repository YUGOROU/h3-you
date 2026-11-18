"""Build a separate pinned experimental Job spec; NEVER submits or edits baseline."""
import argparse
import copy
from pathlib import Path
from research import build_hf_identity_candidate as candidate,hf_bootstrap_identity_pipeline as wire
from research.h3_personal.acceleration_experiment import SCHEMA,validate

PRESETS={
    'bf16-control':('none',277,False),
    'fp8-ffn':('fp8-ffn',277,False),
    'fp8-blocks':('fp8-blocks',277,False),
    'short-native':('none',158,False),
    'incremental-native':('none',277,True),
    'fp8-short-incremental':('fp8-blocks',158,True),
}


def experiment(preset):
    quantization,window,incremental=PRESETS[preset]
    return validate(dict(schema=SCHEMA,quantization=quantization,window_frames=window,
                         incremental_decode=incremental,fast_accum=True))


def build(previous_spec,*,runtime_mount,runtime_repo,runtime_revision,preset,spent_and_reserved_usd,total_seconds=3600):
    old=wire.load(previous_spec)
    if not old['pipeline'].get('preview_only') or not old['pipeline'].get('inference',{}).get('native_continuation'):
        raise ValueError('reviewed native recorded-speech baseline required')
    from research.h3_personal.execution_budget import cap
    if type(spent_and_reserved_usd) not in (int,float) or not 0<=spent_and_reserved_usd<=cap(old['pipeline'].get('hf_price_precision_approval')):
        raise ValueError('explicit reviewed ledger bound required')
    if type(total_seconds) is not int or not 2400<=total_seconds<=4800:
        raise ValueError('explicit finite40..80minute experimental allocation required')
    bootstrap_seconds=min(1200,total_seconds-1201)
    driver_seconds=total_seconds-bootstrap_seconds
    mount=Path(runtime_mount)
    manifest=wire.load(mount/'runtime-manifest.json')
    required={'research/prepare_h3_fp8_runtime.py','research/h3_personal/acceleration_experiment.py',
              'research/h3_personal/incremental_decode.py'}
    if not required<=set(manifest['bootstrap_files']):raise ValueError('new complete experimental runtime required')
    spec=candidate.build(previous_spec,runtime_mount=mount,runtime_repo=runtime_repo,
        runtime_revision=runtime_revision,rate_usd_hour=10,total_seconds=6000)
    r=spec['pipeline'];r.update(overall_seconds=driver_seconds,rate_usd_hour=10.00002,
        operator_managed_stages=True,spent_and_reserved_usd=spent_and_reserved_usd)
    r['inference']=copy.deepcopy(old['pipeline']['inference']);r['inference']['seconds']=min(3600,driver_seconds)
    r['inference']['acceleration_experiment']=experiment(preset)
    r['hf_price_precision_approval']=copy.deepcopy(old['pipeline']['hf_price_precision_approval'])
    r['hf_price_precision_approval'].update(scope=SCHEMA,
        approved_by='USER explicitly requests separate aggressive acceleration experiment Job',
        approval_text='Separate H200 FP8 base Linear experiment with immutable identity/Turbo originals; native short windows and incremental decode, exact same twoH20010.00002USD/hour, finite4800s/13.34USD per attempt; USER manages corrective retry costs.')
    spec['runtime_bootstrap']['timeout_seconds']=bootstrap_seconds
    # Mandatory price proof remains explicit; do not invent provider/rate approvals.
    from research.h3_personal.compute_price_approval import request_limit
    request_limit(r)
    from research.h3_personal.execution_budget import cap
    if spent_and_reserved_usd+10.00002*total_seconds/3600>cap(r['hf_price_precision_approval']):raise ValueError('cumulative allocation cap exceeded')
    return spec


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=PRESETS,default='fp8-short-incremental')
    p.add_argument('--total-seconds',type=int,default=3600)
    p.add_argument('--previous-spec',type=Path);p.add_argument('--runtime-mount',type=Path)
    p.add_argument('--runtime-repo');p.add_argument('--runtime-revision')
    p.add_argument('--spent-and-reserved-usd',type=float);p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    if not 2400<=a.total_seconds<=4800:p.error('total-seconds must be2400..4800')
    if a.output is None:
        import json
        print(json.dumps(dict(plan_only=True,gpu_submitted=False,timeout_seconds=a.total_seconds,configuration=experiment(a.preset),
            fresh_runtime_required=True,baseline_unchanged=True),indent=2));return 0
    if any(v is None for v in (a.previous_spec,a.runtime_mount,a.runtime_repo,a.runtime_revision,a.spent_and_reserved_usd)):
        p.error('new runtime pins, previous spec and reviewed ledger bound required')
    value=build(a.previous_spec,runtime_mount=a.runtime_mount,runtime_repo=a.runtime_repo,
        runtime_revision=a.runtime_revision,preset=a.preset,spent_and_reserved_usd=a.spent_and_reserved_usd,
        total_seconds=a.total_seconds)
    with a.output.open('x') as f:
        import json
        json.dump(value,f,indent=2);f.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
