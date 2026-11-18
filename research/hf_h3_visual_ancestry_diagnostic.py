"""Finite ancestry worker; no cloud launch or device operations."""
import copy
import json
import math
import os
from pathlib import Path
import time
from research.h3_personal.visual_ancestry_diagnostic import (
    SECTION, SCHEMA, ENV, LIMIT, speech, inventory, run_diagnostic)


def validate_spec(spec):
    from research.hf_h3_continuation_diagnostic import strict_equal
    from research.hf_h3_idle_diagnostic import validate_spec as idle_validate, SECTION as IDLE
    if type(spec) is not dict or not strict_equal(spec.get('visual_ancestry_comparison'), SECTION):
        raise ValueError('exact visual ancestry comparison required')
    if 'idle_diagnostic' in spec:
        raise ValueError('conflicting idle mode')
    clone=copy.deepcopy(spec)
    clone.pop('visual_ancestry_comparison'); contract=clone.pop('diagnostic_speech',None)
    if type(contract) is not dict:
        raise ValueError('pinned diagnostic speech required')
    clone['idle_diagnostic']=dict(IDLE)
    idle_validate(clone)
    return contract


def publish(api, deployment, directory, report, *, partial=False):
    from huggingface_hub import CommitOperationAdd
    if not api.repo_info(deployment,repo_type='dataset').private:
        raise ValueError('private generated-only publication required')
    if partial:
        # Known generated basenames only; never recurse into driver/input trees.
        import re
        names=[]
        for arm in ('arm-0','arm-1'):
            folder=Path(directory)/arm
            if not folder.exists():
                continue
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError('regular partial output directory required')
            names.extend(arm+'/'+p.name for p in sorted(folder.iterdir())
                if (p.name in ('review.mov','evidence.json') or
                    re.fullmatch(r'(?:contact-[0-9]{3}\.png|native-[0-9]{3}\.mkv)',p.name)))
        entries=inventory(directory,names)
    else:
        entries=inventory(directory,[e['path'] for e in report['files']])
    if not partial and tuple(report['files']) != entries:
        raise ValueError('immutable inventory changed')
    summary=dict(schema=SCHEMA,completed=not partial,diagnostic_only=True)
    data=json.dumps(summary,sort_keys=True).encode()
    if len(data)+sum(e['bytes'] for e in entries)>LIMIT:
        raise ValueError('publication limit')
    ops=[CommitOperationAdd(path_in_repo='visual-ancestry/'+('failure.json' if partial else 'report.json'),path_or_fileobj=data)]
    for e in entries:
        # Snapshot bytes after validating immutable explicit inventory.
        body=(Path(directory)/e['path']).read_bytes()
        from research.h3_personal.visual_ancestry_diagnostic import digest
        if digest(body)!=e['sha256']: raise ValueError('artifact changed')
        ops.append(CommitOperationAdd(path_in_repo='visual-ancestry/'+e['path'],path_or_fileobj=body))
    api.create_commit(deployment,repo_type='dataset',operations=ops,commit_message='Private finite generated ancestry diagnostic')


def run_session(root, paths, spec, deployment, deadline):
    from huggingface_hub import HfApi
    from research.h3_personal.live_voice_runtime import LiveVoiceRuntime
    from research.h3_personal.accepted_profiles import choose_profile
    from research.h3_personal.conversation_conditioning import prompt_sha
    from research.h3_personal.ref2va_runtime import sha
    contract=validate_spec(spec);root=Path(root);pcm=speech(root,contract)
    if 'H3_IDLE_ANCHOR_DIAGNOSTIC' in os.environ:
        raise ValueError('old idle environment must be absent')
    if not math.isfinite(deadline) or deadline-time.monotonic()<360:
        raise ValueError('finite worker deadline required')
    paths={k:Path(v) for k,v in paths.items()}
    api=HfApi(token=os.environ['HF_TOKEN'])
    if not api.repo_info(deployment,repo_type='dataset').private: raise ValueError('private deployment required')
    reference=paths['reference']/'production-inputs/references/secondary.jpeg'
    profile=choose_profile(ordered_source_sha256=(sha(reference),),prompt_sha256=prompt_sha('speech'))
    if profile.name!='reference1024': raise ValueError('no quality fallback')
    output=root.with_name(root.name+'-ancestry-generated')
    identity=paths['personal']/spec['identity_prefix']
    runtime=LiveVoiceRuntime(model=paths['base'],model_manifest=root/'model-manifest.json',
        identity=identity/'adapter_model.safetensors',identity_config=identity/'adapter_config.json',
        smd=paths['smd']/'lora/adapter_model.safetensors',reference=reference,
        archive=root/'runtime/FastVideo.tar.gz',output=output,continuation=True,
        quality_profile=profile,conversation_roles=True)
    previous=os.environ.get(ENV)
    try:
        try:
            os.environ[ENV]=SCHEMA
            runtime.start()
            if deadline-time.monotonic()<360:
                raise TimeoutError('insufficient comparison/publication time after warmup')
            report=run_diagnostic(runtime,output/'diagnostic',pcm=pcm,deadline=min(deadline-60,time.monotonic()+600))
        finally:
            try: runtime.close()
            finally:
                if previous is None: os.environ.pop(ENV,None)
                else: os.environ[ENV]=previous
    except Exception:
        if time.monotonic()<deadline:
            try: publish(api,deployment,output/'diagnostic',{},partial=True)
            except Exception: pass
        raise
    publish(api,deployment,output/'diagnostic',report)
    return report
