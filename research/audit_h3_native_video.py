"""CPU-only, local-only frame/provenance audit of the native685->683 export.

No automatic artifact detector or quality score. Labels describe pinned decode
control flow, NOT GPU pixel parity. Uses existing FFmpeg/Pillow, no installation.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

GROUPS={
 'first-clip':[0,1,2,3,4,5,8,12,16],
 'vae-seams':[15,16,17,18,19,20,21,22,23,24,32,33,34,35,36,37,38,39],
 'continuation':[252,254,255,260,270,271,272,276,277,278,288,300,510,526,527,531,532,533,548,650],
}


def frame_context(frame):
    """Pinned5-latent stride/17-frame bodies,5-frame overlap;202latents total."""
    if type(frame) is not int or not 0<=frame<685:raise ValueError('native frame0..684 required')
    body,phase=divmod(frame,17);tail=body==40;clip=min(body,39)
    slot=not tail and body>0 and phase<5
    return dict(frame=frame,seconds=str(Fraction(frame,24)),decode_clip=clip,
        decode_latents=[clip*5,clip*5+7],tail=tail,body_phase=phase,
        temporal_overlap_slot=slot,two_clip_blend=slot and phase>0,
        incoming_clip_weight=str(Fraction(phase,5)) if slot else None,
        before_first_continuation_influence=frame<272)


def first_decode_influence(latent_index):
    """Earliest7-latent decode clip containing a newly appended latent.

    Native cumulative append boundaries are82/157. This is a possible influence
    envelope (272/527), NOT proof of a seam artifact or exact pixel dependence.
    """
    if type(latent_index) is not int or not 0<=latent_index<202:raise ValueError('native latent0..201 required')
    return max(0,(latent_index-6+4)//5)*17


def audit(video,output,expected_sha256=None):
    from PIL import Image,ImageDraw
    video=Path(video).resolve();output=Path(output).resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('generated frames must remain outside repository')
    sha=hashlib.sha256(video.read_bytes()).hexdigest()
    if expected_sha256 is not None and sha!=expected_sha256:raise ValueError('source video SHA differs')
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)],timeout=30))
    stream=next(s for s in meta['streams'] if s['codec_type']=='video')
    if (stream['width'],stream['height'],int(stream['nb_frames']),Fraction(stream['avg_frame_rate']))!=(768,1344,683,24):
        raise ValueError('this audit requires native768x1344/683frames/24fps export')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    frames=sorted({n for group in GROUPS.values() for n in group});select='+'.join(f'eq(n,{n})' for n in frames)
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f"select='{select}',scale=256:448",
        '-fps_mode','passthrough','-frames:v',str(len(frames)),str(output/'selected-%03d.png')],check=True,timeout=120)
    images={n:output/f'selected-{i:03d}.png' for i,n in enumerate(frames,1)}
    for name,group in GROUPS.items():
        columns=3 if name=='first-clip' else 4;rows=(len(group)+columns-1)//columns
        canvas=Image.new('RGB',(columns*256,rows*488),'white');draw=ImageDraw.Draw(canvas)
        for i,n in enumerate(group):
            x=i%columns*256;y=i//columns*488;context=frame_context(n)
            with Image.open(images[n]) as image:canvas.paste(image,(x,y+40))
            draw.text((x+3,y+2),f'frame {n} / {n/24:.3f}s / clip {context["decode_clip"]}',fill='black')
            draw.text((x+3,y+18),f'temporal blend: {context["two_clip_blend"]}',fill='black')
        canvas.save(output/(name+'.png'))
    report=dict(schema='h3-native-visual-audit-v1',source=str(video),sha256=sha,
        groups=GROUPS,frames=[frame_context(n) for n in frames],
        native_append_latents=[82,157],earliest_append_decode_influence=[first_decode_influence(n) for n in (82,157)],
        automated_ghost_detection=False,gpu_pixel_parity_verified=False,autoplay=False,uploads=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('video',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-sha256');a=p.parse_args(argv)
    result=audit(a.video,a.output,a.expected_sha256)
    print(json.dumps(dict(audit=str(a.output/'audit.json'),source_sha256=result['sha256'],autoplay=False)))
    return 0

if __name__=='__main__':raise SystemExit(main())
