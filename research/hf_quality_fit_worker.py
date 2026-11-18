"""CPU-only refit using pinned prior scalar features; no image/model inference."""
import hashlib
import json
from pathlib import Path
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from huggingface_hub import HfApi
    from quality_policy import propose
    inputs=Path('/inputs');request=json.loads((inputs/'request.json').read_text())
    data=inputs/'features.json'
    if sha256(data)!=request['features_sha256']:raise ValueError('feature artifact hash mismatch')
    payload=json.loads(data.read_text());rows=payload['rows']
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate rows')
    if payload['feature_names']!=request['feature_names']:raise ValueError('feature recipe mismatch')
    for r in rows:
        evidence=request['samples'][r['id']]
        if r['source_sha256']!=evidence['source_sha256']:raise ValueError('source mismatch')
        r['label']=evidence['label']
    api=HfApi()
    if not api.repo_info(request['repo'],repo_type='dataset').private:raise ValueError('private outputs required')
    out=Path('/work/output');out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();status={'state':'running','cpu_only':True,'feature_rows':len(rows),'media_reencoded':False}
    def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    def sync():api.upload_folder(repo_id=request['repo'],repo_type='dataset',folder_path=out,
        path_in_repo='outputs/'+request['run_id'],commit_message='Private CPU refit on reused quality features')
    write('status.json',status);sync()
    try:
        result=propose(rows)
        if result['validation']['status']=='worker_model_error':raise RuntimeError('policy model fit failed')
        result.update(schema_version=1,run_id=request['run_id'],cpu_only=True,
            model_sha256=request['model_sha256'],input_snapshot_sha256=request['label_snapshot_sha256'],
            feature_names=payload['feature_names'],preview_kind='uniform grayscale 320px filmstrip',
            features_reused=True,feature_origin=request['feature_origin'],
            human_labels_overwritten=False,full_clip_review_complete=False)
        write('quality_proposals.json',result);write('features.json',payload)
        status.update(state='complete',completed=len(rows),errors=sum(bool(r.get('error')) for r in rows),
            elapsed_seconds=time.monotonic()-start,
            proposal_counts={k:sum(p['decision']==k for p in result['proposals'].values()) for k in ('auto_proposal','review')})
    except Exception as exc:
        status.update(state='error',error=type(exc).__name__+': '+str(exc)[:300]);raise
    finally:
        write('status.json',status);sync();print(json.dumps(status),flush=True)

if __name__=='__main__':main()
