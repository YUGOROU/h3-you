"""Private four-condition H3 comparison orchestration inside one finite GPU Job."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

SOURCE_DIGESTS={
 'models/transformers/transformer_minimax_h3.py':'45f1d4a4e9e72128f0d6c5b5d66379b61131a354b58d2776280a83f6c89f8d51',
 'models/modeling_utils.py':'87cc36742ac805343093a589206b07e5d833201f8d8f2942daddf3d6a27f2ed0',
 'models/attention_dispatch.py':'5fa72380e153fc6e9cddd23848800958e1431fbfe1402e60595c250e75c77eb1',
 'models/_modeling_parallel.py':'be790a2cd4dd7be50705de2c3d1ab0d6cb7d44bea540cf2cefe10d32c2dc30ca',
 'hooks/context_parallel.py':'d58fac70f989d5aa82ea38011f62b59e3220898bbc22bd59a0acff9b64ff4481'}


def run_matrix(request_path,checkpoint,casepath,pipeline,env,api):
    import diffusers
    from huggingface_hub import CommitOperationAdd
    from research import hf_identity_train as train
    from research.hf_identity_encode import dump
    from research.h3_personal.context_parallel_plan import DIFFUSERS_REVISION,verify_source_files
    from research.h3_personal.turbo_adapter import TURBO_SHA256
    request=train.read_json(request_path);case=train.read_json(casepath);root=Path(case['output'])
    config=pipeline['inference']['context_parallel_matrix']
    turbo=Path(config['path'])
    if config['sha256']!=TURBO_SHA256 or turbo.stat().st_size!=config['bytes'] or train.sha256(turbo)!=TURBO_SHA256:
        raise ValueError('complete pinned Turbo payload required')
    source_root=Path(diffusers.__file__).resolve().parent
    receipts=[dict(file=k,sha256=v,revision=DIFFUSERS_REVISION) for k,v in SOURCE_DIGESTS.items()]
    verify_source_files(source_root,receipts)
    receipts_path=root/'source-receipts.json';dump(receipts_path,receipts)
    def run(args):train.bounded_process([sys.executable,*args],deadline=None,env=env)
    run(['-m','research.hf_identity_preview','pack-comparison','--request',str(request_path),
         '--checkpoint',str(checkpoint),'--case',str(casepath)])
    packed=root/'private-packed.safetensors';meta=root/'private-packed.json'
    reports=[];revision=None
    repo=request['output_repo'];prefix='identity/'+request['run_id']+'/acceleration-matrix/'
    for label,extra in [('baseline',[]),('turbo8',['--turbo',str(turbo)])]:
        out=root/label
        run(['-m','torch.distributed.run','--standalone','--nproc-per-node=2',
             '-m','research.benchmark_h3_context_parallel','--request',str(request_path),
             '--checkpoint',str(checkpoint),'--packed',str(packed),'--metadata',str(meta),
             '--metadata-sha256',train.sha256(meta),'--source-root',str(source_root),
             '--source-receipts',str(receipts_path),'--source-receipts-sha256',train.sha256(receipts_path),
             '--output',str(out),'--full-trajectory','--execute',*extra])
        report=train.read_json(out/'comparison-report.json');reports.append(report)
        operations=[CommitOperationAdd(path_in_repo=prefix+label+'/comparison-report.json',
                                       path_or_fileobj=str(out/'comparison-report.json'))]
        for variant in ('single-gpu','two-gpu-cp'):
            clip=out/variant;clip.mkdir(mode=0o700)
            shutil.copyfile(out/(variant+'-latents.safetensors'),clip/'video-latents.safetensors')
            decode_case=dict(case,output=str(clip));decode_casepath=clip/'case.json'
            dump(decode_casepath,decode_case)
            run(['-m','research.hf_identity_preview','decode','--request',str(request_path),
                 '--checkpoint',str(checkpoint),'--case',str(decode_casepath)])
            for name in ('review.mp4','decode-profile.json'):
                operations.append(CommitOperationAdd(path_in_repo=prefix+label+'/'+variant+'/'+name,
                                                     path_or_fileobj=str(clip/name)))
        if not api.repo_info(repo,repo_type='model').private:raise ValueError('private output required')
        revision=api.create_commit(repo,repo_type='model',operations=operations,
            commit_message='Private generated '+label+' comparison; no raw inputs/packed tensors').oid
    summary=dict(schema='h3-four-condition-acceleration-matrix-v1',reports=reports,
        combined_denoise_speedup=reports[0]['single_gpu_denoise_seconds']/reports[1]['cp_denoise_seconds'],
        identity_quality_verified=False,realtime_verified=False,
        scope='paired fixed inputs; user must compare all four generated videos; DiT-only speedup')
    summary_path=root/'matrix-report.json';dump(summary_path,summary)
    revision=api.create_commit(repo,repo_type='model',operations=[CommitOperationAdd(
        path_in_repo=prefix+'matrix-report.json',path_or_fileobj=str(summary_path))],
        commit_message='Four-condition completed matrix report; subjective identity not accepted').oid
    return revision
