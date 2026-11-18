"""Freeze the USER-approved715 ranges. No selection, inference, uploads or jobs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def build(selected,verify_sources=True):
    if len(selected)!=715:raise ValueError('fixed production selection requires exactly715')
    samples=[];ids=set();sources={}
    for c in selected:
        sid=c['id'];digest=c['source_sha256'];path=str(Path(c['source']).expanduser().resolve())
        if (not isinstance(sid,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',sid)
                or sid in ('.','..') or sid in ids):raise ValueError('unsafe/duplicate ID')
        if not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest):raise ValueError('source hash required')
        if c['pool'] not in ('identity','motion') or c.get('human_review')!='approved_sampled_images':
            raise ValueError('only explicitly human-selected ranges allowed')
        start,end=c['start_seconds'],c['end_seconds']
        if (type(start) not in (int,float) or type(end) not in (int,float) or
                not all(math.isfinite(v) for v in (start,end)) or start<0 or abs(end-start-124/24)>1e-6):
            raise ValueError('invalid124-frame range')
        ids.add(sid)
        if path in sources and sources[path]!=digest:raise ValueError('contradictory source hash')
        sources[path]=digest
        samples.append({'id':sid,'source':path,'source_sha256':digest,'session_id':c['session_id'],
                        'start_seconds':start,'end_seconds':end,'quality_pool':c['pool']})
    counts=Counter(s['quality_pool'] for s in samples)
    if counts!={'identity':590,'motion':125}:raise ValueError('fixed pool composition changed')
    if verify_sources:
        for path,digest in sources.items():
            if file_sha256(path)!=digest:raise ValueError('source bytes changed')
    semantic=[{k:v for k,v in row.items() if k!='source'} for row in samples]
    result={'schema':'h3-production-selection-v1','dataset_sha256':hashlib.sha256(canonical(semantic)).hexdigest(),
        'sample_count':715,'pool_counts':dict(counts),'clip_seconds_with_tail_overlap':sum(s['end_seconds']-s['start_seconds'] for s in samples),
        'geometry':{'width':768,'height':1344,'num_frames':124,'fps':24},'samples':samples,
        'selection_frozen':True,'source_bytes_verified':verify_sources,
        'approval':'USER fixed715 reviewed clips for production; CPU/GPU Jobs authorized with post-launch reporting',
        'technical_training_ready':False,'full_clip_review_complete':False,
        'budget':{'cumulative_cap_usd':150,'max_concurrent_gpu_usd_per_hour':10,
                  'finite_timeouts_required':True,'private_artifacts_required':True,'automatic_retry':False},
        'h3_territory_clearance_required':True}
    return result


def freeze(selected_path,output):
    selected_path=Path(selected_path);output=Path(output)
    result=build(json.loads(selected_path.read_text()))
    result['selected_manifest_sha256']=file_sha256(selected_path)
    data=canonical(result)+b'\n';output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if output.read_bytes()!=data:raise ValueError('frozen dataset cannot be overwritten')
    else:
        with output.open('xb') as f:f.write(data)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selected-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=freeze(a.selected_manifest.expanduser(),a.output.expanduser())
    print(json.dumps({k:r[k] for k in ('dataset_sha256','sample_count','pool_counts','clip_seconds_with_tail_overlap','source_bytes_verified')},indent=2))

if __name__=='__main__':main()
