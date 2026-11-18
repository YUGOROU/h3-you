"""Apply explicit local human review decisions; never encode or upload media.
Decisions JSON: [{id, source_sha256, action: accept|reject, pool: identity|motion,
face_reviewed: true, audio_reviewed: true, reason: ...}]. Human audio review must
cover third-party voices and sync. Accepted clips are not claimed independent.
"""
import argparse
import json
from pathlib import Path
from .preprocess import dump


def apply(candidates, decisions):
    by_id = {c['id']:c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError('duplicate candidate IDs')
    seen, selected = set(), []
    for d in decisions:
        ident = d['id']
        if ident in seen or ident not in by_id:
            raise ValueError('unknown/duplicate review ID')
        seen.add(ident)
        c = by_id[ident]
        if d['source_sha256'] != c['source_sha256']:
            raise ValueError('review belongs to different source')
        if d['action'] not in ('accept','reject') or not d.get('reason'):
            raise ValueError('explicit decision and reason required')
        if d['action'] == 'accept':
            if d.get('face_reviewed') is not True or d.get('audio_reviewed') is not True:
                raise ValueError('face/audio review required')
            if d.get('pool') not in ('identity','motion'):
                raise ValueError('explicit pool required')
            selected.append({**c, 'pool':d['pool'], 'human_review':'approved',
                'audio_review':'approved', 'training_approved':True, 'review_reason':d['reason']})
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    a=p.parse_args()
    candidates=json.loads((a.selection_dir/'candidate_manifest.json').read_text())
    selected=apply(candidates,json.loads(a.decisions.read_text()))
    dump(a.selection_dir/'selected_manifest.json',selected)
    summary={'selected':len(selected),'clip_seconds_with_overlap':sum(c['end_seconds']-c['start_seconds'] for c in selected),
        'pools':{k:sum(c['pool']==k for c in selected) for k in ('identity','motion')},
        'independent_evaluation_claimed':False,'gpu_job_submitted':False}
    dump(a.selection_dir/'human_review_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
