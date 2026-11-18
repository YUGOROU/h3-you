"""Explicit, bounded HF CPU refit. Uploads scalar features/labels, NOT media.
Default: local stage only. New fit never changes human judgments or precision gates.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(root, run_id):
    root=Path(root).expanduser().resolve()
    old=json.loads((root/'quality-job-latest.json').read_text())
    prior=json.loads((root/'quality_proposals.json').read_text())
    if prior['run_id']!=old['run_id'] or not prior.get('cpu_only') or not prior.get('output_revision'):
        raise ValueError('completed, revision-pinned prior result required')
    features=root/'quality-jobs'/old['run_id']/'outputs/features.json'
    payload=json.loads(features.read_text())
    if payload['feature_names']!=prior['feature_names']:raise ValueError('feature names differ')
    labels=json.loads((root/'image_labels.json').read_text())['labels']
    queue=json.loads((root/'calibration_queue.json').read_text())
    if any(labels.get(i,{}).get('label') not in ('identity','motion','reject') for i in queue['ids']):
        raise ValueError('additional teacher judgments not complete')
    candidates={c['id']:c for c in json.loads((root/'candidate_manifest.json').read_text())}
    samples={}
    for row in payload['rows']:
        ident=row['id'];c=candidates[ident];label=labels.get(ident)
        if ident in samples:raise ValueError('duplicate feature row')
        for field in ('source_sha256','session_id','start_seconds','end_seconds'):
            if row[field]!=c[field]:raise ValueError('feature range/provenance mismatch')
        if label and label['source_sha256']!=c['source_sha256']:raise ValueError('label source mismatch')
        samples[ident]={'source_sha256':c['source_sha256'],
                       'label':label['label'] if label and label['label']!='hold' else None}
    target=root/'quality-jobs'/run_id;target.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(features,target/'features.json')
    project=Path(__file__).resolve().parent
    shutil.copyfile(project/'hf_quality_fit_worker.py',target/'hf_quality_fit_worker.py')
    shutil.copyfile(project/'h3_personal/quality_policy.py',target/'quality_policy.py')
    request={'run_id':run_id,'repo':None,'samples':samples,'feature_names':payload['feature_names'],
        'features_sha256':sha256(target/'features.json'),'model_sha256':prior['model_sha256'],
        'feature_origin':prior.get('feature_origin',{'run_id':old['run_id'],'repo':old['repo'],
            'output_revision':prior['output_revision'],'model_sha256':prior['model_sha256']}),
        'label_snapshot_sha256':hashlib.sha256(json.dumps(labels,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'policy_sha256':sha256(target/'quality_policy.py'),'worker_sha256':sha256(target/'hf_quality_fit_worker.py'),
        'teacher_ids':queue['ids'],'teacher_labels':{i:labels[i]['label'] for i in queue['ids']},
        'labels_without_cached_features':sum(i not in samples for i in labels),
        'note':'Initial327 approvals lacking cached features are not model-training rows; they remain authoritative human labels.'}
    (target/'request.json').write_text(json.dumps(request,indent=2))
    return target,request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir',type=Path,required=True)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--previous-total-usd',type=float,default=29.8216)
    a=p.parse_args()
    if not 0<=a.previous_total_usd<=149.95:p.error('cumulative budget exceeded')
    run='quality-refit-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target,request=stage(a.selection_dir,run)
    plan={'run_id':run,'state':'prepared_not_submitted','cpu_only':True,'flavor':'cpu-upgrade',
        'timeout':'20m','spend_cap_usd':.05,'previous_total_usd':a.previous_total_usd,
        'features_reused':True,'media_uploaded':False,'media_reencoded':False,'automatic_retry':False,
        'samples':len(request['samples']),'labelled':sum(r['label'] is not None for r in request['samples'].values()),
        'labels_without_cached_features':request['labels_without_cached_features']}
    def save():(target/'plan.json').write_text(json.dumps(plan,indent=2))
    save()
    if a.submit:
        os.environ['HF_TOKEN']=(Path.home()/'.cache/huggingface/token').read_text().strip()
        from huggingface_hub import HfApi,Volume
        api=HfApi();owner=api.whoami()['name']
        if list(api.list_jobs(namespace=owner,labels={'project':'h3-quality'},status=['SCHEDULING','RUNNING'])):
            raise RuntimeError('active quality job exists; no duplicate')
        hardware=next(h for h in api.list_jobs_hardware() if h.name=='cpu-upgrade')
        rate=hardware.unit_cost_usd*(60 if hardware.unit_label=='minute' else 1 if hardware.unit_label=='hour' else 0)
        if hardware.accelerator or not 0<rate<=.05:raise ValueError('CPU price cap unverified')
        repo=f'{owner}/h3-{run}'
        api.create_repo(repo,repo_type='dataset',private=True,exist_ok=False)
        if not api.repo_info(repo,repo_type='dataset').private:raise ValueError('private inputs required')
        request['repo']=repo;(target/'request.json').write_text(json.dumps(request,indent=2))
        commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=target,
            allow_patterns=['request.json','features.json','hf_quality_fit_worker.py','quality_policy.py'],
            commit_message='Private CPU refit on existing scalar quality features with new teachers')
        plan.update(repo=repo,input_revision=commit.oid,state='inputs_uploaded_not_submitted',usd_per_hour=rate);save()
        command='''set -euo pipefail
python -m pip install -q numpy==1.26.4 scikit-learn==1.6.1 huggingface_hub==1.33.0
python /inputs/hf_quality_fit_worker.py
'''
        job=api.run_job(image='python:3.11-slim-bookworm',flavor='cpu-upgrade',timeout='20m',
            command=['bash','-lc',command],secrets={'HF_TOKEN':os.environ['HF_TOKEN']},
            env={'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'','HF_HUB_DISABLE_TELEMETRY':'1'},
            volumes=[Volume(type='dataset',source=repo,revision=commit.oid,mount_path='/inputs',read_only=True)],
            labels={'project':'h3-quality','name':run})
        plan.update(state='submitted',job_id=job.id);save()
    (a.selection_dir.expanduser()/'quality-job-latest.json').write_text(json.dumps(plan,indent=2)) if a.submit else None
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
