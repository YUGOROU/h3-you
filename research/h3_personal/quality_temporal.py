"""Stdlib temporal quality summaries. No identity embeddings or model execution.
Missing detections stay missing; motion is measured only on adjacent valid frames.
"""
import math
import statistics

FEATURE_NAMES = ['valid_longest_run_fraction','detection_transition_fraction','face_area_p90',
    'face_area_cv','face_sharp_p25_log','face_sharp_p75_log','eyes_sharp_median_log',
    'mouth_sharp_median_log','eyes_to_face_sharp_log_ratio','mouth_to_face_sharp_log_ratio',
    'exposure_clip_median','landmarks_inside_median','eye_open_p10','eye_open_p90',
    'mouth_open_p90','yaw_range','pitch_range','adjacent_center_motion_median',
    'adjacent_center_motion_max','adjacent_area_change_max','adjacent_eye_open_change_max']


def percentile(values, fraction):
    values=sorted(values)
    if not values:return 0.
    index=(len(values)-1)*fraction;lo=int(index);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def summarize(frames):
    if not frames:raise ValueError('no sampled frames')
    n=len(frames);valid=[f for f in frames if f is not None]
    runs=[];run=0
    for f in frames:
        run=run+1 if f is not None else 0;runs.append(run)
    transitions=sum((a is None)!=(b is None) for a,b in zip(frames,frames[1:]))
    base=[max(runs)/n,transitions/max(1,n-1)]
    if not valid:return base+[0.]*(len(FEATURE_NAMES)-2)
    keys=('area','sharp','eyes_sharp','mouth_sharp','clipped','inside','eye_open','mouth_open','yaw','pitch')
    if any(not math.isfinite(f[k]) for f in valid for k in keys):raise ValueError('nonfinite metric')
    if any(not math.isfinite(v) for f in valid for v in f['center']):raise ValueError('nonfinite center')
    values=lambda k:[f[k] for f in valid]
    med=lambda k:statistics.median(values(k))
    log=lambda x:math.log1p(max(0.,x))
    pairs=[(a,b) for a,b in zip(frames,frames[1:]) if a is not None and b is not None]
    moves=[math.dist(a['center'],b['center']) for a,b in pairs]
    area=values('area');mean=statistics.mean(area)
    result=base+[percentile(area,.9),statistics.pstdev(area)/max(mean,1e-9),
        log(percentile(values('sharp'),.25)),log(percentile(values('sharp'),.75)),
        log(med('eyes_sharp')),log(med('mouth_sharp')),
        statistics.median(log(f['eyes_sharp'])-log(f['sharp']) for f in valid),
        statistics.median(log(f['mouth_sharp'])-log(f['sharp']) for f in valid),
        med('clipped'),med('inside'),percentile(values('eye_open'),.1),percentile(values('eye_open'),.9),
        percentile(values('mouth_open'),.9),max(values('yaw'))-min(values('yaw')),
        max(values('pitch'))-min(values('pitch')),statistics.median(moves) if moves else 0.,
        max(moves,default=0.),max((abs(a['area']-b['area']) for a,b in pairs),default=0.),
        max((abs(a['eye_open']-b['eye_open']) for a,b in pairs),default=0.)]
    if len(result)!=len(FEATURE_NAMES) or not all(math.isfinite(v) for v in result):raise ValueError('invalid summary')
    return result
