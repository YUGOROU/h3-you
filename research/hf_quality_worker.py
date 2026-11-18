"""HF CPU-only landmark/quality worker. Never trains H3 or identifies a person."""
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import time
import urllib.request

MODEL_URL='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
MODEL_SHA256='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
FEATURE_NAMES=['detection_fraction','multiple_face_fraction','face_area_min','face_area_median',
    'face_sharpness_min_log','face_sharpness_median_log','eyes_sharpness_min_log',
    'mouth_sharpness_min_log','face_clip_max','landmarks_in_frame_min',
    'pitch_abs_max','pitch_abs_median','yaw_abs_max','yaw_abs_median','roll_abs_max',
    'eye_openness_median','center_motion_max','area_change_max']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def frame_metrics(rgb, result):
    import cv2
    import numpy as np
    if len(result.face_landmarks)!=1:return None
    points=np.array([(p.x,p.y) for p in result.face_landmarks[0]])
    h,w=rgb.shape[:2]
    low=points.min(axis=0);high=points.max(axis=0)
    x0,y0=np.floor(low*[w,h]).astype(int);x1,y1=np.ceil(high*[w,h]).astype(int)
    x0=max(0,x0);y0=max(0,y0);x1=min(w,x1);y1=min(h,y1)
    if x1-x0<12 or y1-y0<12:return None
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    roi=gray[y0:y1,x0:x1]
    def sharp(indices):
        sub=points[indices]*[w,h]
        x,y=np.floor(sub.min(axis=0)-4).astype(int);xx,yy=np.ceil(sub.max(axis=0)+4).astype(int)
        patch=gray[max(0,y):min(h,yy),max(0,x):min(w,xx)]
        return float(cv2.Laplacian(patch,cv2.CV_64F).var()) if patch.size else 0.
    # Canonical-face camera-relative Euler proxies, not calibrated absolute pose.
    matrix=np.asarray(result.facial_transformation_matrixes[0])[:3,:3]
    u,_,v=np.linalg.svd(matrix); rotation=u@v
    if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@v
    pitch,yaw,roll=cv2.RQDecomp3x3(rotation)[0]
    distance=lambda a,b:float(np.linalg.norm((points[a]-points[b])*[w,h]))
    eyes_open=(distance(159,145)/(distance(33,133)+1e-6)+distance(386,374)/(distance(362,263)+1e-6))/2
    return {'area':(x1-x0)*(y1-y0)/(w*h),'sharp':float(cv2.Laplacian(roi,cv2.CV_64F).var()),
        'eyes_sharp':min(sharp([33,133,159,145]),sharp([362,263,386,374])),
        'mouth_sharp':sharp([61,291,13,14]),'clipped':float(((roi<12)|(roi>243)).mean()),
        'inside':float(((points>=0)&(points<=1)).all(axis=1).mean()),
        'pitch':abs(float(pitch)),'yaw':abs(float(yaw)),'roll':abs(float(roll)),
        'eye_open':eyes_open,'center':((low+high)/2).tolist()}


def extract(strip_path, landmarker):
    import cv2
    import mediapipe as mp
    import numpy as np
    strip=cv2.imread(str(strip_path))
    if strip is None or strip.shape[1]%320 or not 4<=strip.shape[1]//320<=5:
        raise ValueError('invalid 320px temporal filmstrip')
    n=strip.shape[1]//320;frames=[];multiple=0
    for i in range(n):
        rgb=cv2.cvtColor(strip[:,i*320:(i+1)*320],cv2.COLOR_BGR2RGB)
        result=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
        multiple+=int(len(result.face_landmarks)>1)
        metric=frame_metrics(rgb,result)
        if metric:frames.append(metric)
    return aggregate_metrics(frames,n,multiple)


def aggregate_metrics(frames,n,multiple):
    import numpy as np
    if not frames:return {'features':[0.]*len(FEATURE_NAMES),'detection_fraction':0.,'frame_count':n}
    values=lambda k:[f[k] for f in frames]
    log=lambda x:math.log1p(max(0.,x))
    minval=lambda k:min(values(k));med=lambda k:float(np.median(values(k)));maxval=lambda k:max(values(k))
    moves=[math.dist(a['center'],b['center']) for a,b in zip(frames,frames[1:])]
    changes=[abs(a['area']-b['area']) for a,b in zip(frames,frames[1:])]
    features=[len(frames)/n,multiple/n,minval('area'),med('area'),log(minval('sharp')),log(med('sharp')),
        log(minval('eyes_sharp')),log(minval('mouth_sharp')),maxval('clipped'),minval('inside'),
        maxval('pitch'),med('pitch'),maxval('yaw'),med('yaw'),maxval('roll'),med('eye_open'),
        max(moves,default=0.),max(changes,default=0.)]
    if not all(math.isfinite(v) for v in features):raise ValueError('nonfinite features')
    return {'features':features,'detection_fraction':len(frames)/n,'frame_count':n}


def main(extractor=None,feature_names=None):
    extractor=extract if extractor is None else extractor
    feature_names=FEATURE_NAMES if feature_names is None else feature_names
    from huggingface_hub import HfApi
    import cv2
    import mediapipe as mp
    from quality_policy import propose
    cv2.setNumThreads(1)
    root=Path('/work');root.mkdir(exist_ok=True)
    inputs=Path('/inputs');out=root/'output';out.mkdir(exist_ok=True)
    request=json.loads((inputs/'request.json').read_text()); api=HfApi()
    if not api.repo_info(request['repo'],repo_type='dataset').private:raise RuntimeError('private outputs required')
    start=time.monotonic()
    def write(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    def sync():
        api.upload_folder(repo_id=request['repo'],repo_type='dataset',folder_path=out,
            path_in_repo='outputs/'+request['run_id'],commit_message='Private CPU quality analysis outputs')
    status={'state':'running','cpu_only':True,'completed':0,'total':len(request['samples'])}
    write('status.json',status);sync()
    try:
        if digest(inputs/'previews.tar.gz')!=request['bundle_sha256']:raise ValueError('bundle hash mismatch')
        # Only explicit, flat, regular JPEG files accepted. No archive path traversal.
        with tarfile.open(inputs/'previews.tar.gz','r:gz') as archive:
            for member in archive.getmembers():
                if not member.isfile() or '/' in member.name or '\\' in member.name or not member.name.endswith('.jpg'):
                    raise ValueError('unsafe preview bundle')
                with archive.extractfile(member) as source, (root/member.name).open('wb') as target:target.write(source.read())
        model=root/'face_landmarker.task'
        with urllib.request.urlopen(MODEL_URL,timeout=60) as response:model.write_bytes(response.read(8_000_001))
        if digest(model)!=MODEL_SHA256:raise ValueError('landmark model hash mismatch')
        options=mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model),delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=2,
            min_face_detection_confidence=.5,min_face_presence_confidence=.5,
            output_facial_transformation_matrixes=True)
        rows=[]
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            for i,sample in enumerate(request['samples']):
                if time.monotonic()-start>45*60:raise TimeoutError('worker 45-minute processing cap')
                row={k:sample[k] for k in ('id','source_sha256','session_id','start_seconds','end_seconds','label')}
                try:
                    image=root/(sample['id']+'.jpg')
                    if digest(image)!=sample['preview_sha256']:raise ValueError('preview hash mismatch')
                    row.update(extractor(image,landmarker))
                except Exception as exc:
                    row.update(error=type(exc).__name__+': '+str(exc)[:180],features=[],detection_fraction=0.)
                rows.append(row)
                if (i+1)%50==0:
                    status['completed']=i+1;write('features.json',{'feature_names':feature_names,'rows':rows});write('status.json',status);sync()
                    print('quality samples',i+1,'/',len(request['samples']),flush=True)
        write('features.json',{'feature_names':feature_names,'rows':rows})
        result=propose(rows)
        result.update(schema_version=1,run_id=request['run_id'],model_sha256=MODEL_SHA256,
            input_snapshot_sha256=request['label_snapshot_sha256'],human_labels_overwritten=False,
            full_clip_review_complete=False,preview_kind='uniform grayscale 320px filmstrip',
            cpu_only=True,feature_names=feature_names,
            feature_recipe='temporal-v2' if extractor is not extract else 'legacy-v1')
        write('quality_proposals.json',result)
        status.update(state='complete',completed=len(rows),errors=sum(bool(r.get('error')) for r in rows),
            elapsed_seconds=time.monotonic()-start,
            proposal_counts={k:sum(p['decision']==k for p in result['proposals'].values()) for k in ('auto_proposal','review')})
    except Exception as exc:
        status.update(state='error',error=type(exc).__name__+': '+str(exc)[:500]);raise
    finally:
        write('status.json',status);sync()
        print(json.dumps(status),flush=True)

if __name__=='__main__':main()
