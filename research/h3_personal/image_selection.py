"""Export authoritative human image labels as provisional selected clip ranges.
No encoding/uploads. This does not assert full-clip or Trainer/GPU readiness.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from .preprocess import dump


def build(candidates, labels):
    by_id={c['id']:c for c in candidates}
    if len(by_id)!=len(candidates):raise ValueError('duplicate candidates')
    for ident,label in labels.items():
        if ident not in by_id or label['source_sha256']!=by_id[ident]['source_sha256']:
            raise ValueError('human label provenance mismatch')
        if label['label'] not in ('identity','motion','reject','hold'):raise ValueError('unknown human label')
    selected=[]
    for c in candidates:
        label=labels.get(c['id'])
        if not label or label['label'] not in ('identity','motion'):continue
        selected.append({**c,'pool':label['label'],'human_review':'approved_sampled_images',
            'review_scope':label.get('review_scope','sampled_color_frames'),
            'audio_stability':'user_reported_verified','audio_speaker_and_sync_independently_verified':False,
            'full_clip_review_complete':False,'training_approved':False,
            'selection_basis':'explicit_human_image_label_not_automatic_quality_prediction'})
    counts=Counter(c['pool'] for c in selected)
    summary={'selected':len(selected),'pools':dict(counts),
        'clip_seconds_with_tail_overlap':sum(c['end_seconds']-c['start_seconds'] for c in selected),
        'human_rejected':sum(l['label']=='reject' for l in labels.values()),
        'unlabelled_candidates':sum(c['id'] not in labels for c in candidates),
        'held_candidates':sum(l['label']=='hold' for l in labels.values()),
        'unlabelled_are_not_rejected':True,'training_ready':False,
        'sampling_target':{'identity':.65,'motion':.35},
        'note':'Sampling target is not the observed pool ratio; no automatic resampling performed.'}
    return selected,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir',type=Path,required=True)
    a=p.parse_args();root=a.selection_dir.expanduser()
    candidates=json.loads((root/'candidate_manifest.json').read_text())
    labels=json.loads((root/'image_labels.json').read_text())['labels']
    selected,summary=build(candidates,labels)
    dump(root/'selected_manifest.json',selected)
    dump(root/'selected_image_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
