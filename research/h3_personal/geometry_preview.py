"""Local CPU-only fixed anisotropic previews. No ML, tracking, uploads or autoplay."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


def filters(width,height,ratio):
    if (type(width) is not int or type(height) is not int or width%2 or height%2 or
        min(width,height)<16 or max(width,height)>4096 or type(ratio) not in (int,float) or
        not math.isfinite(ratio) or not .5<=ratio<=1):
        raise ValueError('bounded even canvas and finite horizontal ratio .5..1 required')
    narrow=max(2,round(width*ratio/2)*2)
    tall=round(height/ratio/2)*2
    return dict(pad=dict(filter=f'scale={narrow}:{height}:flags=lanczos,setsar=1,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black',
                         actual_ratio=narrow/width,padding=True,source_crop_fraction=0),
                crop=dict(filter=f'scale={width}:{tall}:flags=lanczos,setsar=1,crop={width}:{height}:(iw-ow)/2:(ih-oh)/2',
                          actual_ratio=height/tall,padding=False,source_crop_fraction=1-height/tall))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
                     '-show_format','-of','json',str(path)],timeout=20))


def audio_signature(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0',
        '-show_packets','-show_entries','packet=data_hash','-show_data_hash','sha256',
        '-of','json',str(path)],timeout=20))
    hashes=[p['data_hash'] for p in data.get('packets',[])]
    return dict(packets=len(hashes),sha256=hashlib.sha256('\n'.join(hashes).encode()).hexdigest())


def render(source,output,ratios):
    source=Path(source).resolve(strict=True);output=Path(output)
    if source.stat().st_size>64*1024**2:raise ValueError('local preview input exceeds64MiB')
    metadata=probe(source);videos=[s for s in metadata['streams'] if s['codec_type']=='video']
    if len(videos)!=1:raise ValueError('exact one video stream required')
    stream=videos[0];width,height=stream['width'],stream['height']
    duration=float(metadata['format']['duration'])
    if not 0<duration<=15 or stream.get('sample_aspect_ratio') not in (None,'N/A','1:1'):
        raise ValueError('short square-pixel input required; audit SAR first')
    if any(x.get('rotation',0)!=0 for x in stream.get('side_data_list',[])) or stream.get('tags',{}).get('rotate','0')!='0':
        raise ValueError('rotation metadata requires explicit orientation audit')
    ratios=list(ratios)
    if not ratios or len(ratios)>8 or len(set(ratios))!=len(ratios):raise ValueError('1..8 unique ratios required')
    plans=[(r,filters(width,height,r)) for r in ratios]
    original_hash=digest(source);audio=audio_signature(source)
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    results=[]
    for index,(ratio,modes) in enumerate(plans):
        for mode,plan in modes.items():
            target=output/f'{index}-{ratio:.3f}-{mode}.mp4'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(source),
                '-map','0:v:0','-map','0:a:0?','-vf',plan['filter'],
                '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                '-c:a','copy','-movflags','+faststart',str(target)],check=True,timeout=120)
            subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(target),
                            '-f','null','-'],check=True,timeout=60)
            actual=probe(target);video=next(s for s in actual['streams'] if s['codec_type']=='video')
            if ((video['width'],video['height'])!=(width,height) or
                video.get('nb_frames')!=stream.get('nb_frames') or
                abs(float(actual['format']['duration'])-duration)>.05 or audio_signature(target)!=audio):
                raise ValueError('preview geometry/frame count/duration/audio payload changed unexpectedly')
            results.append(dict(file=target.name,requested_ratio=ratio,mode=mode,**plan,
                                sha256=digest(target),decoded=True,audio_payload_unchanged=True))
    if digest(source)!=original_hash:raise ValueError('original video changed')
    report=dict(schema='h3-fixed-geometry-preview-v1',source_sha256=original_hash,
        canvas=[width,height],duration_seconds=duration,frames=stream.get('nb_frames'),
        audio_packet_signature=audio,candidates=results,automatic_ratio_estimation=False,
        ratio_verified=False,identity_quality_verified=False,original_unchanged=True,
        policy='fixed whole-clip scale; no per-frame face warp; no ML/no upload/no autoplay')
    (output/'preview-report.json').write_text(json.dumps(report,indent=2))
    return report
