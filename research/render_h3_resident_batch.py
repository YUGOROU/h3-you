"""Three same-condition BF16/50point renders in a preprovisioned offline GPU runtime.

Measures cache hit / measured-envelope residency; no quality-reducing switches.
Not a Job launcher. Default plan-only, no artifact uploads or playback.
"""
import argparse
import json
import os
from pathlib import Path
from research.h3_personal.component_cache import sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['request','checkpoint','case','adapter-sha256','output']:p.add_argument('--'+flag,required=True)
    p.add_argument('--execute-local-models',action='store_true')
    args=p.parse_args(argv)
    if not args.execute_local_models:
        print(json.dumps({'plan_only':True,'gpu_job_launch':False,'renders':3,'baseline_points':50,
                          'precision':'bf16','quantization':'none','requires':'preprovisioned runtime'}));return 0
    os.umask(0o077);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from research.hf_identity_preview import create_resident_session,generate,decode
    request=json.loads(Path(args.request).read_text());base=json.loads(Path(args.case).read_text())
    if sha256(Path(args.checkpoint)/'adapter_model.safetensors')!=args.adapter_sha256:raise ValueError('adapter differs')
    if base.get('schedule_points')!=50:raise ValueError('preserve baseline50points')
    out=Path(args.output);out.mkdir(parents=True,mode=0o700,exist_ok=False);results=[]
    with create_resident_session(allow_measured_auxiliary=True) as session:
        for i in range(3):
            directory=out/str(i);directory.mkdir(mode=0o700);case=dict(base,output=str(directory))
            generate(request,args.checkpoint,case,session=session)
            decode(request,case,session=session)
            results.append({'case':i,'latent_file_sha256':sha256(directory/'video-latents.safetensors'),
                'generation':json.loads((directory/'inference-profile.json').read_text()),
                'decode':json.loads((directory/'decode-profile.json').read_text())})
        report={'schema':'h3-same-condition-resident-batch-v1','quality_verified':False,
            'exact_latent_file_bytes_equal':len({r['latent_file_sha256'] for r in results})==1,
            'model_loads':session.loads,'resident_hits':session.hits,'results':results}
    (out/'batch-profile.json').write_text(json.dumps(report,sort_keys=True))
    print(json.dumps({'completed':True,'exact_latent_bytes_equal':report['exact_latent_file_bytes_equal'],
                      'resident_hits':report['resident_hits'],'report':str(out/'batch-profile.json')}));return 0


if __name__=='__main__':raise SystemExit(main())
