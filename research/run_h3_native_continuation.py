"""Finite native continuation orchestration. All conditioning stays local/private."""
import json
from pathlib import Path
import sys
from research import hf_identity_train as train
from research.hf_identity_encode import dump
from research.h3_personal.continuation_contract import validate


def run_native(request_path,checkpoint,casepath,pipeline,env,api):
    from huggingface_hub import CommitOperationAdd
    request=train.read_json(request_path);case=train.read_json(casepath);root=Path(case['output'])
    config=validate(pipeline['inference']['native_continuation'])
    case.update(native=config,audio_path=config['audio']['path'],audio_sha256=config['audio']['sha256'],schedule_points=9)
    experiment=pipeline['inference'].get('acceleration_experiment')
    if experiment is not None:
        from research.h3_personal.acceleration_experiment import validate as validate_experiment
        case['acceleration_experiment']=validate_experiment(experiment)
    elif 'acceleration_experiment' in case:
        raise ValueError('experimental case cannot silently enter baseline branch')
    from research.h3_personal.local_asset import stage
    case['local_turbo']=str(stage(config['turbo']['path'],root/'turbo-local.safetensors',config['turbo']['sha256']))
    dump(casepath,case)
    def run(args):train.bounded_process([sys.executable,*args],deadline=None,env=env)
    for phase in ('reference','text','audio'):
        print('NATIVE_PREPARE',phase,flush=True)
        run(['-m','research.hf_native_continuation',phase,'--request',str(request_path),'--case',str(casepath)])
    files=['reference.safetensors','reference-binding.json','text.safetensors','text-binding.json',
           'audio.safetensors','audio-report.json']
    manifest=dict(schema='h3-private-native-conditioning-v1',request_sha256=train.sha256(request_path),
        checkpoint_export_sha256=train.sha256(Path(checkpoint)/'EXPORTED.json'),
        case_sha256=train.sha256(casepath),files={n:train.sha256(root/n) for n in files},
        private_conditioning_do_not_upload=True)
    dump(root/'private-conditioning.json',manifest)
    run(['-m','torch.distributed.run','--standalone','--nproc-per-node=2',
         '-m','research.hf_native_continuation','generate','--request',str(request_path),
         '--case',str(casepath),'--checkpoint',str(checkpoint),
         '--conditioning-sha256',train.sha256(root/'private-conditioning.json')])
    run(['-m','research.hf_native_continuation','decode','--request',str(request_path),'--case',str(casepath)])
    # Strict generated-only allowlist. No input WAV/image, latent, prompt, local paths or bindings.
    names=['review.mp4','continuation-profile.json','decode-profile.json']
    branch='acceleration-experiment' if experiment is not None else 'native-continuation'
    repo=request['output_repo'];prefix='identity/'+request['run_id']+'/'+branch+'/'
    if api.repo_info(repo,repo_type='model').private is not True:raise ValueError('private output required')
    revision=api.create_commit(repo,repo_type='model',operations=[CommitOperationAdd(
        path_in_repo=prefix+n,path_or_fileobj=str(root/n)) for n in names],
        commit_message='Private generated native continuation; original recorded audio lossless mux; no raw inputs or latents').oid
    dump(root/'native-upload.json',dict(revision=revision,files=names,prefix=prefix))
    return revision
