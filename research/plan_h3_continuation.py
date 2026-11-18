"""Local WAV/grid preflight only; never loads models, uploads or starts Jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import wave
from research.h3_personal.continuation_plan import request


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',required=True,type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--window-frames',type=int,default=277)
    p.add_argument('--overlap-frames',type=int,default=22)
    a=p.parse_args(argv)
    path=a.audio.resolve(strict=True)
    if path.stat().st_size>64*1024**2:raise ValueError('bounded local PCM WAV required')
    with wave.open(str(path),'rb') as w:
        if w.getcomptype()!='NONE':raise ValueError('PCM required; no automatic conversion')
        r=request(w.getnframes(),w.getframerate(),window_frames=a.window_frames,overlap_frames=a.overlap_frames)
        r['source_format']=dict(channels=w.getnchannels(),sample_width_bytes=w.getsampwidth())
    r.update(source_audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),plan_only=True,
             gpu_launched=False,source_unchanged=True,implementation_ready_for_launch=False)
    content=json.dumps(r,ensure_ascii=False,indent=2)+'\n'
    if a.output:
        os.umask(0o077)
        with a.output.open('x') as f:f.write(content)
    else:print(content,end='')
    return 0


if __name__=='__main__':raise SystemExit(main())
