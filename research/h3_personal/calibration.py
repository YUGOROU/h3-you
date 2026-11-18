"""Lightweight representative queue preparation; no decoding, network or jobs."""
import argparse
import hashlib
from collections import defaultdict
import json
import math
from pathlib import Path
from .preprocess import dump


def scaled_features(candidates, rows):
    """Small stdlib scalar transform, NOT face inference/model fitting.

    Clamp at cohort 5/95 percentiles so one extreme feature does not dominate.
    Detection failures remain eligible; this is annotation sampling, not rejection.
    """
    vectors={}
    for c in candidates:
        row=rows.get(c['id'])
        if row is None or row.get('error'):
            raise ValueError('usable cached feature missing')
        for field in ('source_sha256','session_id','start_seconds','end_seconds'):
            if row[field]!=c[field]:raise ValueError('cached feature provenance mismatch')
        values=row['features']
        if not values or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise ValueError('invalid scalar features')
        vectors[c['id']]=values
    if len({len(v) for v in vectors.values()})!=1:raise ValueError('inconsistent feature dimensions')
    bounds=[]
    for col in zip(*vectors.values()):
        values=sorted(col)
        bounds.append((values[int((len(values)-1)*.05)],values[int((len(values)-1)*.95)]))
    return {ident:[min(1.,max(0.,(v-lo)/(hi-lo))) if hi>lo else 0.
                   for v,(lo,hi) in zip(values,bounds)] for ident,values in vectors.items()}


def choose(candidates, labels, count=30, feature_rows=None):
    if type(count) is not int or count <= 0:
        raise ValueError('count must be positive')
    groups=defaultdict(list)
    for c in candidates:
        if c.get('pool')=='review' and c['id'] not in labels:
            groups[c['session_id']].append(c)
    if not groups:
        return []
    count=min(count,sum(map(len,groups.values())))
    vectors=scaled_features([c for items in groups.values() for c in items],feature_rows) if feature_rows is not None else None
    quotas={s:0 for s in groups}
    # First cover each session, then reserve up to six examples per session.
    for _ in range(6):
        for s in sorted(groups):
            if sum(quotas.values())>=count:break
            if quotas[s]<len(groups[s]):quotas[s]+=1
    while sum(quotas.values())<count:
        s=max((s for s in groups if quotas[s]<len(groups[s])),
              key=lambda s:(len(groups[s])-quotas[s],s))
        quotas[s]+=1
    chosen=[]
    for s in sorted(groups):
        items=sorted(groups[s],key=lambda c:(c['start_seconds'],c['id']))
        item_ids={c['id'] for c in items}
        n=quotas[s]
        for i in range(n):
            # Temporal strata; cycle low/median/high proxy scores, not only top quality.
            bucket=items[i*len(items)//n:(i+1)*len(items)//n]
            ranked=sorted(bucket,key=lambda c:(c.get('score',0),c['id']))
            index=(0,len(ranked)//2,len(ranked)-1)[i%3]
            if vectors is not None:
                previous=[vectors[ident] for ident in chosen if ident in item_ids]
                if previous:
                    # Farthest cached feature within each temporal stratum.
                    selected=max(bucket,key=lambda c:(min(sum((a-b)**2 for a,b in zip(vectors[c['id']],v))
                        for v in previous),c['id']))
                else:
                    selected=ranked[len(ranked)//2]
                chosen.append(selected['id'])
            else:
                chosen.append(ranked[index]['id'])
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir',type=Path,required=True)
    p.add_argument('--count',type=int,default=30)
    p.add_argument('--features',type=Path,help='Already fetched scalar features JSON; no inference/network')
    a=p.parse_args(); root=a.selection_dir.expanduser()
    candidates=json.loads((root/'candidate_manifest.json').read_text())
    labels=json.loads((root/'image_labels.json').read_text())['labels'] if (root/'image_labels.json').exists() else {}
    payload=json.loads(a.features.expanduser().read_text()) if a.features else None
    rows={r['id']:r for r in payload['rows']} if payload else None
    if payload and len(rows)!=len(payload['rows']):raise ValueError('duplicate cached feature rows')
    ids=choose(candidates,labels,a.count,rows)
    by_id={c['id']:c for c in candidates}
    dump(root/'calibration_queue.json',{'schema_version':1,'ids':ids,
        'source_hashes':{i:by_id[i]['source_sha256'] for i in ids},
        'method':('unlabelled review; session coverage, temporal strata, robust-scaled feature diversity'
                  if rows is not None else 'unlabelled review candidates; session coverage and temporal strata with cycling proxy score quantiles'),
        'feature_names':payload['feature_names'] if payload else None,
        'features_sha256':hashlib.sha256(a.features.expanduser().read_bytes()).hexdigest() if payload else None,
        'label_snapshot_sha256':hashlib.sha256(json.dumps(labels,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'limits':'representative calibration aid, not a random evaluation or guaranteed coverage of all failure types'})
    print(json.dumps({'count':len(ids),'sessions':{s:sum(by_id[i]['session_id']==s for i in ids) for s in sorted({by_id[i]['session_id'] for i in ids})}},indent=2))

if __name__=='__main__':main()
