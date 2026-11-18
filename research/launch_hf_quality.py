"""Stage minimal private previews; explicitly submit ONE CPU-only bounded Job.
Default is local preparation only. No automatic retry. Never calls GPU/train APIs.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def stage(root, run_id, feature_version=1):
    if feature_version not in (1,2):raise ValueError('unsupported feature version')
    root=Path(root).expanduser().resolve()
    candidates=json.loads((root/'candidate_manifest.json').read_text())
    labels=json.loads((root/'image_labels.json').read_text())['labels']
    if len({c['id'] for c in candidates})!=len(candidates):raise ValueError('duplicate candidate IDs')
    snapshots=json.loads((root/'calibration_queue.json').read_text())
    if any(labels.get(i,{}).get('label') not in ('identity','motion','reject') for i in snapshots['ids']):
        raise ValueError('calibration judgments must be complete')
    target=root/'quality-jobs'/run_id;target.mkdir(parents=True,exist_ok=False)
    samples=[]
    bundle=target/'previews.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for c in candidates:
            if feature_version==1 and c.get('pool')!='review':continue
            ident=c['id']
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',ident) or ident in ('.','..'):raise ValueError('unsafe sample ID')
            image=root/'previews'/(ident+'.jpg')
            label=labels.get(ident)
            if label and label['source_sha256']!=c['source_sha256']:raise ValueError('source label hash mismatch')
            archive.add(image,arcname=ident+'.jpg',recursive=False)
            samples.append({k:c[k] for k in ('id','source_sha256','session_id','start_seconds','end_seconds')}|
                {'label':label['label'] if label and label['label']!='hold' else None,'preview_sha256':sha256(image)})
    project=Path(__file__).resolve().parent
    shutil.copyfile(project/'hf_quality_worker.py',target/'hf_quality_worker.py')
    shutil.copyfile(project/'h3_personal/quality_policy.py',target/'quality_policy.py')
    if feature_version==2:
        shutil.copyfile(project/'hf_quality_v2_worker.py',target/'hf_quality_v2_worker.py')
        shutil.copyfile(project/'h3_personal/quality_temporal.py',target/'quality_temporal.py')
    snapshot_digest=hashlib.sha256(json.dumps(labels,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    request={'run_id':run_id,'feature_version':feature_version,'samples':samples,'bundle_sha256':sha256(bundle),
        'label_snapshot_sha256':snapshot_digest,'repo':None,
        'worker_sha256':sha256(target/'hf_quality_worker.py'),'policy_sha256':sha256(target/'quality_policy.py'),
        'extension_sha256':sha256(target/'hf_quality_v2_worker.py') if feature_version==2 else None,
        'temporal_summary_sha256':sha256(target/'quality_temporal.py') if feature_version==2 else None,
        'preview_kind':'uniform grayscale 320px temporal filmstrip; no raw video/audio/reference photo uploads'}
    (target/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
    return target,request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir',type=Path,required=True)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--feature-version',type=int,choices=(1,2),default=1)
    p.add_argument('--previous-total-usd',type=float,default=29.8207)
    a=p.parse_args()
    if not 0<=a.previous_total_usd<=149.95:p.error('cumulative $150 budget would be exceeded')
    run='quality-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target,request=stage(a.selection_dir,run,a.feature_version)
    plan={'run_id':run,'state':'prepared_not_submitted','cpu_only':True,'flavor':'cpu-upgrade','timeout':'1h',
        'spend_cap_usd':.05,'previous_total_usd':a.previous_total_usd,
        'samples':len(request['samples']),'labelled':sum(s['label'] is not None for s in request['samples']),
        'upload_bytes':sum(x.stat().st_size for x in target.iterdir()),'automatic_retry':False,
        'input_scope':request['preview_kind'],'h3_model_or_training_used':False}
    (target/'plan.json').write_text(json.dumps(plan,indent=2))
    if a.submit:
        os.environ['HF_TOKEN']=(Path.home()/'.cache/huggingface/token').read_text().strip()
        from huggingface_hub import HfApi,Volume
        api=HfApi();owner=api.whoami()['name']
        if list(api.list_jobs(namespace=owner,labels={'project':'h3-quality'},status=['SCHEDULING','RUNNING'])):
            raise RuntimeError('active quality job exists; no duplicate submission')
        hardware=next(h for h in api.list_jobs_hardware() if h.name=='cpu-upgrade')
        rate=hardware.unit_cost_usd*(60 if hardware.unit_label=='minute' else 1 if hardware.unit_label=='hour' else 0)
        if hardware.accelerator or not 0<rate<=.05:raise RuntimeError('CPU-only price cap could not be verified')
        repo=f'{owner}/h3-quality-{run}'
        api.create_repo(repo,repo_type='dataset',private=True,exist_ok=False)
        if not api.repo_info(repo,repo_type='dataset').private:raise RuntimeError('private inputs required')
        request['repo']=repo;(target/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
        commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=target,
            allow_patterns=['previews.tar.gz','request.json','hf_quality_worker.py','quality_policy.py',
                            'hf_quality_v2_worker.py','quality_temporal.py'],
            commit_message='Private minimal previews and human labels for CPU quality proposals')
        if not api.repo_info(repo,repo_type='dataset',revision=commit.oid).private:raise RuntimeError('private pinned mount required')
        command='''set -euo pipefail
apt-get update -qq
apt-get install -y -qq libgl1 libglib2.0-0
python -m pip install -q numpy==1.26.4 mediapipe==0.10.21 opencv-contrib-python==4.11.0.86 scikit-learn==1.6.1 huggingface_hub==1.33.0
mkdir -p /work
python /inputs/hf_quality_worker.py
'''
        if a.feature_version==2:
            command=command.replace('python /inputs/hf_quality_worker.py','python /inputs/hf_quality_v2_worker.py')
        plan.update(feature_version=a.feature_version,repo=repo,input_revision=commit.oid,usd_per_hour=rate,state='inputs_uploaded_not_submitted')
        (target/'plan.json').write_text(json.dumps(plan,indent=2))
        # One bounded submission, never automatically retried even after an API/network error.
        job=api.run_job(image='python:3.11-slim-bookworm',flavor='cpu-upgrade',timeout='1h',
            command=['bash','-lc',command],secrets={'HF_TOKEN':os.environ['HF_TOKEN']},
            env={'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'',
                 'HF_HUB_DISABLE_TELEMETRY':'1'},
            volumes=[Volume(type='dataset',source=repo,revision=commit.oid,mount_path='/inputs',read_only=True)],
            labels={'project':'h3-quality','name':run})
        plan.update(state='submitted',job_id=job.id)
        (target/'plan.json').write_text(json.dumps(plan,indent=2))
    if a.submit:
        (a.selection_dir.expanduser()/'quality-job-latest.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
