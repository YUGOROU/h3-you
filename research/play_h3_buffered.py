"""Foreground opt-in buffered playback of already-existing private local assets.

Default validates only. --play opens an AI-labelled ffplay window; Ctrl-C or i+Enter
interrupts and discards the whole turn. No downloads/GPU/server/OBS/Discord.
"""
import argparse
import json
from pathlib import Path
import select
import sys
import time
from research.h3_personal.buffered_stream import BufferedStream, Clip, validate_clip
from research.h3_personal.playback_sink import LocalPlaybackSink, PlaybackPump


def load_manifest(path):
    path=Path(path)
    if path.stat().st_size>65536: raise ValueError('oversized playback manifest')
    value=json.loads(path.read_text())
    if set(value)!={'schema','clips'} or value['schema']!='h3-buffered-playback-v1':
        raise ValueError('explicit playback manifest required')
    if not isinstance(value['clips'],list) or not 1<=len(value['clips'])<=128:
        raise ValueError('bounded nonempty clip list required')
    fields={'video','original_tts','video_sha256','audio_sha256'}
    if any(not isinstance(c,dict) or set(c)!=fields for c in value['clips']):
        raise ValueError('unexpected clip fields')
    return [Clip(**c) for c in value['clips']]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--play',action='store_true')
    args=p.parse_args(argv);root=Path(args.root).resolve(strict=True)
    clips=load_manifest(args.manifest)
    durations=[validate_clip(root,c) for c in clips]
    if any(d>15 for d in durations):raise ValueError('clip exceeds bounded15s playback buffer')
    print(json.dumps({'validated_clips':len(clips),'playback_requested':args.play,
                      'ai_generated':True,'streaming':'buffered, not real-time generation'}))
    if not args.play:return 0
    if not sys.stdin.isatty():raise ValueError('--play requires interactive stop control')
    q=BufferedStream(root);turn=q.begin();index=0;done=False
    with LocalPlaybackSink(root,allow_playback=True) as sink:
        pump=PlaybackPump(q,sink)
        try:
            print('AI-generated avatar. Ctrl-C or i+Enter stops both tracks and discards pending clips.')
            while q.current(turn):
                if select.select([sys.stdin],[],[],0)[0] and sys.stdin.readline().strip().lower()=='i':
                    pump.interrupt();break
                if index<len(clips) and q.has_capacity(durations[index]):
                    try:
                        if q.enqueue(turn,clips[index]):index+=1
                    except BufferError:pass
                if index==len(clips) and not done:q.finish(turn);done=True
                pump.tick();time.sleep(.02)
        except KeyboardInterrupt:
            pump.interrupt()
        finally:
            pump.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
