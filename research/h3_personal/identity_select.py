"""Local-only identity candidate triage. No uploads, latent encoding, or job API.

Requires OpenCV only for --scan. Haar detections are visibility proxies, NOT
identity/occlusion/pose judgments. All candidates require human review.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
from .preprocess import DURATION, dump, probe, starts


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def assess(frames):
    """Conservative triage; thresholds provisional at 320px analysis width."""
    if not frames or any(f.get('decode_error') for f in frames):
        return {'pool': 'review', 'reasons': ['decode_error'], 'score': 0}
    visible = [f for f in frames if f['face_count'] == 1]
    reasons = []
    if len(visible) < len(frames):
        reasons.append('face_detection_incomplete_or_multiple')
    if not visible:
        return {'pool': 'review', 'reasons': reasons, 'score': 0}
    area = min(f['face_area_ratio'] for f in visible)
    sharp = min(f['face_laplacian_variance'] for f in visible)
    clipping = max(f['face_clipped_fraction'] for f in visible)
    if area < .025:
        reasons.append('small_face_proxy')
    if sharp < 35:
        reasons.append('soft_face_proxy')
    if clipping > .25:
        reasons.append('face_exposure_proxy')
    score = round(area * min(sharp, 300) * (1 - clipping), 4)
    pool = 'review' if reasons else ('identity' if area >= .08 else 'motion')
    return {'pool': pool, 'reasons': reasons, 'score': score}


def inventory(directories):
    records, seen = [], set()
    for directory in directories:
        for path in sorted(directory.expanduser().glob('*.mov')):
            p = probe(path)
            video = next(s for s in p['streams'] if s['codec_type'] == 'video')
            audios = [s for s in p['streams'] if s['codec_type'] == 'audio']
            digest = sha256(path)
            reasons = []
            if digest in seen:
                reasons.append('duplicate_sha256')
            seen.add(digest)
            if path.name.startswith('rejected_'):
                reasons.append('explicitly_rejected_source')
            if not audios:
                reasons.append('no_audio_track')
            metadata_path = path.with_name(path.name.removeprefix('rejected_')).with_suffix('.json')
            meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            if meta.get('status') != 'completed' or meta.get('stopReason') or meta.get('error'):
                reasons.append('capture_metadata_incomplete_or_interrupted')
            duration = float(video.get('duration', p['format']['duration']))
            rotation = next((d['rotation'] for d in video.get('side_data_list', []) if 'rotation' in d), 0)
            records.append({'session_id': path.stem, 'original_path': str(path.resolve()),
                'sha256': digest, 'size_bytes': path.stat().st_size, 'duration_seconds': duration,
                'fps': video['avg_frame_rate'], 'width': video['width'], 'height': video['height'],
                'orientation_rotation': rotation, 'audio_present': bool(audios),
                'audio_streams': [{k: a.get(k) for k in ('codec_name','sample_rate','channels','start_time','duration')} for a in audios],
                'video_start_time': video.get('start_time'), 'outfit': 'unverified',
                'environment': 'unverified', 'framing': 'unverified',
                'speaker_and_sync_review': 'pending', 'exclusion_reasons': reasons})
    return records


def scan(record, output, limit=None):
    import cv2
    import numpy as np
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        raise RuntimeError('face detector unavailable')
    w, h = record['width'], record['height']
    if abs(record['orientation_rotation']) % 180 == 90:
        w, h = h, w
    height = round(h / w * 320 / 2) * 2
    result = []
    for i, start in enumerate(starts(record['duration_seconds'])):
        if limit is not None and i >= limit:
            break
        ident = f"{record['session_id']}_{i:04d}"
        cached = output / 'metrics' / (ident + '.json')
        if cached.exists():
            item = json.loads(cached.read_text())
            if item['source_sha256'] != record['sha256']:
                raise ValueError('source changed; use new output directory')
            result.append(item)
            continue
        # Five temporal observations; one seek/decode per clip, autorotation on.
        raw = subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(start + .15),
            '-i',record['original_path'],'-t',str(DURATION - .3),'-an','-vf',
            f'fps=1,scale=320:{height}', '-frames:v','5','-pix_fmt','gray',
            '-f','rawvideo','pipe:1'], check=True, capture_output=True).stdout
        images = np.frombuffer(raw, dtype=np.uint8)
        if images.size % (320 * height):
            raise ValueError('unexpected frame buffer')
        images = images.reshape(-1, height, 320)
        metrics = []
        for img in images:
            faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(24,24))
            f = {'face_count': len(faces)}
            if len(faces) == 1:
                x,y,fw,fh = map(int, faces[0]); face = img[y:y+fh,x:x+fw]
                f.update(face_box=[x,y,fw,fh], face_area_ratio=fw*fh/(320*height),
                    face_laplacian_variance=float(cv2.Laplacian(face,cv2.CV_64F).var()),
                    face_clipped_fraction=float(((face < 12) | (face > 243)).mean()))
            metrics.append(f)
        if len(metrics) < 4:
            metrics.append({'decode_error': True})
        item = {'id': ident, 'session_id': record['session_id'], 'source': record['original_path'],
            'source_sha256': record['sha256'], 'start_seconds': start, 'end_seconds': start + DURATION,
            'frames': metrics, **assess(metrics), 'human_review': 'pending',
            'audio_review': 'pending', 'training_approved': False}
        # Small filmstrip only; do not duplicate full recordings.
        if len(images):
            strip = np.concatenate(list(images), axis=1)
            cv2.imwrite(str(output / 'previews' / (ident + '.jpg')), strip, [cv2.IMWRITE_JPEG_QUALITY,70])
        dump(cached, item)
        result.append(item)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scan', action='store_true')
    p.add_argument('--limit-per-session', type=int)
    a = p.parse_args()
    if a.limit_per_session is not None and a.limit_per_session <= 0:
        p.error('limit must be positive')
    out = a.output.expanduser().resolve()
    if any(out == s.expanduser().resolve() or s.expanduser().resolve() in out.parents for s in a.source):
        p.error('outputs must be separate from originals')
    out.mkdir(parents=True, exist_ok=True)
    for d in ('metrics','previews'):
        (out/d).mkdir(exist_ok=True)
    records = inventory(a.source)
    dump(out/'source_manifest.json', records)
    candidates = []
    for r in records:
        print(r['session_id'], r['duration_seconds'], r['exclusion_reasons'], flush=True)
        if not r['exclusion_reasons']:
            if a.scan:
                candidates.extend(scan(r,out,a.limit_per_session))
            else:
                candidates.extend({'id': f"{r['session_id']}_{i:04d}", 'source_sha256':r['sha256'],
                    'session_id':r['session_id'], 'source':r['original_path'], 'start_seconds':s,
                    'end_seconds':s+DURATION, 'training_approved':False}
                    for i,s in enumerate(starts(r['duration_seconds'])))
        dump(out/'candidate_manifest.json',candidates)
    if not (out/'selected_manifest.json').exists():
        dump(out/'selected_manifest.json',[])  # Never overwrite prior human approval.
    summary = {'source_seconds':sum(r['duration_seconds'] for r in records if not r['exclusion_reasons']),
        'candidates':len(candidates),'triage_pools':{k:sum(c.get('pool')==k for c in candidates) for k in ('identity','motion','review')},
        'selected':len(json.loads((out/'selected_manifest.json').read_text())),
        'gpu_job_submitted':False, 'quality_selection_complete':False,
        'sampling_target':{'identity':.65,'motion':.35},'audio_speaker_sync_verified':False}
    dump(out/'selection_summary.json',summary)
    rows = []
    for c in candidates:
        if 'pool' not in c: continue
        rows.append(f"<section><h3>{html.escape(c['id'])} — {c['pool']} / {c['score']}</h3>"
            f"<p>{html.escape(', '.join(c['reasons']))}</p><img loading='lazy' width='960' src='previews/{c['id']}.jpg'></section>")
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Local identity triage</title><h1>未承認候補 — 顔の鮮明さ・遮蔽・音声は人手確認が必要</h1>'+''.join(rows))
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
