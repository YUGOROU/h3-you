"""39 temporal quality scalars from existing uniform previews, CPU only.
No identity/biometric embeddings. Same strict proposal gates as v1.
"""
import math
from hf_quality_worker import aggregate_metrics,frame_metrics,main,FEATURE_NAMES as LEGACY_NAMES
from quality_temporal import summarize,FEATURE_NAMES as TEMPORAL_NAMES
FEATURE_NAMES=LEGACY_NAMES+TEMPORAL_NAMES


def extract(path,landmarker):
    import cv2
    import mediapipe as mp
    import numpy as np
    strip=cv2.imread(str(path))
    if strip is None or strip.shape[1]%320 or not 4<=strip.shape[1]//320<=5:
        raise ValueError('invalid uniform 320px filmstrip')
    n=strip.shape[1]//320;frames=[];multiple=0
    for i in range(n):
        rgb=cv2.cvtColor(strip[:,i*320:(i+1)*320],cv2.COLOR_BGR2RGB)
        result=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
        multiple+=int(len(result.face_landmarks)>1)
        metric=frame_metrics(rgb,result)
        if metric is not None:
            p=result.face_landmarks[0];h,w=rgb.shape[:2]
            distance=lambda a,b:math.hypot((p[a].x-p[b].x)*w,(p[a].y-p[b].y)*h)
            metric['mouth_open']=distance(13,14)/(distance(61,291)+1e-6)
        frames.append(metric)
    result=aggregate_metrics([f for f in frames if f is not None],n,multiple)
    result['features']+=summarize(frames)
    result['feature_recipe']='temporal-v2'
    if len(result['features'])!=len(FEATURE_NAMES):raise ValueError('feature dimensions')
    return result

if __name__=='__main__':main(extractor=extract,feature_names=FEATURE_NAMES)
